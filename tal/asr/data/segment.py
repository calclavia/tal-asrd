import os
import math
import torch
import json
import pickle
import inspect
import random
import torchaudio
import numpy as np
from torch.utils.data import Dataset, IterableDataset
from librosa.core import get_duration
from mutagen import File
from joblib import Parallel, delayed
import time
import hashlib
import shutil
from .util import load_audio_segment, is_valid_utterance, tokenize_utterances, tokenize_utterances_word_align

# Default sampling rate
DEFAULT_SR = 16000


def build_index(data_dir: str, file_stub: str, utterances: list, ext: str):
    """
    Builds an index for a single audio file's utterances.
    """
    # File stub
    audio_file = os.path.join(data_dir, '{}{}'.format(
        file_stub, ext
    ))
    f_duration = get_duration(filename=audio_file)
    valid_utts = [u for u in utterances if is_valid_utterance(u, f_duration)]
    return file_stub, valid_utts, f_duration


def is_valid_segment(utterances: list, discontinuity_threshold: float = 3):
    """ Check if this list of utterances contain valid audio timestamps. """
    # Check if the discontinuity between utterances in this block are reasonable
    for u in range(len(utterances) - 1):
        this_u = utterances[u]
        next_u = utterances[u + 1]
        if next_u['utterance_start'] - this_u['utterance_end'] \
                > discontinuity_threshold:
            return False
    return True


"""
from wildspeech.asr.data import ASRSegmentDataset
from wildspeech.asr.tokenizers.sentencepiece import Tokenizer
tokenizer = Tokenizer('./cache/tokenizer/taltoken-cased.model')
dataset = ASRSegmentDataset('./data/tal/valid', tokenizer)
"""


class ASRSegmentDataset(Dataset):
    """
    Audio -> Text dataset.

    During training we obtain a random segment of audio and ask the model to predict all utterances
    within that chunk of audio. The utterances that are beyond the audio bounds are truncated
    proportional to the seconds missed in the audio segment.
    Invalid utterances are ignored.
    If no utterances exists, the model generates '[UNINTELLIGIBLE]'

    Two special tokens are used.

    Text format:
    Begin episode
    <EOS>
        <utterance tokens><speaker_A><EOS>
        <utterance tokens><speaker_B><EOS>
    ...
    <EOT>
    End episode

    where BOS appears at the beginning of the episode, and EOT appears at the end of the episode.
    Speaker tokens starting from the end of the encoder vocabulary.
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer,
        speaker_map_loc: str = None,
        min_segment_size: int = 10,
        segment_size: int = 30,
        segment_shift: int = 10,
        random_segment_shift: int = 5,
        max_tokens: int = 128,
        ext: str = '.wav',
        aligned_truncation: bool = False,
        tokenizer_speakers=False,
        return_spk_ids=False,
    ):
        """
        Args:
            data_dir (str): Path to directory containing data files.
                            Data folder is expected to contain a 'transcript.pkl' file, which stores
                            a dictionary of audio file stubs and maps to the corresponding transcripts of that audio file.
                            Transcripts are formatted as a list of utterance dictionaries.
            speaker_map_loc (str): Path to JSON containing map of lower case speaker name to ID
            tokenizer: Tokenizer object to encode text into IDs
            segment_size (int): Amount of seconds per segment.
            segment_shift (int): Amount of seconds to shift per segment.
            ext (str): Audio file format.
        """
        # TODO: Segment size and shift augmentation
        super().__init__()

        self.data_dir = data_dir
        self.ext = ext
        self.segment_size = segment_size
        self.min_segment_size = min_segment_size
        self.segment_shift = segment_shift
        self.random_segment_shift = random_segment_shift
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.unk_phrase = '[UNINTELLIGIBLE]'
        self.aligned_truncation = aligned_truncation
        self.tokenizer_speakers = tokenizer_speakers
        self.return_spk_ids = return_spk_ids

        if self.aligned_truncation:
            print('Using word-level alignments for truncation!')

        self.speaker_map = None
        if speaker_map_loc:
            # Get speaker map & speaker IDs
            with open(speaker_map_loc, 'r+') as rf:
                self.speaker_map = json.load(rf)
            # The starting ID for speakers.
            self.first_speaker_id = len(self.tokenizer)
            print('Loaded speakers', len(self.speaker_map))

        # Knows when to rebuild cache
        arghash = (self.segment_size, self.segment_shift, self.ext, 0)
        try:
            with open(os.path.join(data_dir, 'cache_segment.pkl'), 'rb') as f:
                cache = pickle.load(f)
                marker, self.index = cache
                assert marker == arghash
        except:
            # Load all transcripts {episode: }
            with open(os.path.join(data_dir, 'transcript.pkl'), 'rb') as f:
                transcripts = pickle.load(f)

            print('Caching {} with {:,} second segments'.format(
                self.__class__.__name__, self.segment_size))
            self.index = Parallel(n_jobs=8)(delayed(build_index)(
                data_dir, file_stub, utts, ext) for file_stub, utts in transcripts.items())

            with open(os.path.join(data_dir, 'cache_segment.pkl'), 'wb') as f:
                pickle.dump((arghash, self.index), f)

        self.total_seconds = sum(d for _, _, d in self.index)

        print('Dataset contains {} hours of audio'.format(
            self.total_seconds / 3600))

    def _get_speaker_id(self, speaker_name: str):
        speaker_name = speaker_name.lower().strip()
        if self.tokenizer_speakers:
            # Unknown speaker gets the last ID
            return self.first_speaker_id + (self.speaker_map[speaker_name] if speaker_name in self.speaker_map else len(self.speaker_map))
        return self.speaker_map[speaker_name] if speaker_name in self.speaker_map else len(self.speaker_map)

    def build_index(self, data_dir: str):
        """ Construct a chunk index for sampling. """
        indexes = Parallel(n_jobs=16)(delayed(build_index_single)(data_dir, file_stub, utterances,
                                                                  self.min_chunk_duration, self.ext) for file_stub, utterances in self.transcripts.items())
        return [x for y in indexes for x in y]

    def __len__(self):
        return int(self.total_seconds) // self.segment_shift

    def __getitem__(self, i: int):
        # Find the index position
        cur_len = 0
        for file_stub, utterances, f_duration in self.index:
            num_segments = f_duration // self.segment_shift
            if i < num_segments:
                # After breaking the loop, the index would be local to this file
                break
            i -= num_segments

        assert f_duration >= self.segment_size
        start_time = min(max(i * self.segment_shift + (random.random() - 0.5)
                             * 2 * self.random_segment_shift, 0), f_duration - self.segment_size)
        end_time = min(start_time + random.random() * (self.segment_size -
                                                       self.min_segment_size) + self.min_segment_size, f_duration)

        # Find all utterances that intersect the given start and end times
        intersecting_utts = []
        for i, utt in enumerate(utterances):
            if utt['utterance_end'] > start_time and utt['utterance_start'] <= end_time:
                intersecting_utts.append((i, utt))
            elif len(intersecting_utts) > 0:
                # Don't need to keep searching
                break

        # If it's all just music or a discontinuity or ads
        if not intersecting_utts:
            # No utterances found. Output IDK.
            utterance_tokens = self.tokenizer.encode(
                self.unk_phrase,
                bos_token=False,
                eos_token=False
            )

            spk_ids = torch.LongTensor(
                len(utterance_tokens) * [self._get_speaker_id('unknown')])
        else:
            # Aligned truncation
            if self.aligned_truncation:
                assert not self.return_spk_ids, 'Not supported'
                utterance_tokens = tokenize_utterances_word_align(
                    utterances=utterances,
                    filtered_utts=intersecting_utts,
                    start_time=start_time,
                    end_time=end_time,
                    segment_size=self.segment_size,
                    tokenizer=self.tokenizer,
                    tokenize_speaker=self.tokenizer_speakers,
                    speaker_to_id=self._get_speaker_id,
                    return_spk_ids=self.return_spk_ids
                )

            # Proportional Truncation
            else:
                # Get utterance tokens
                utterance_tokens, spk_ids = tokenize_utterances(
                    utterances,
                    intersecting_utts,
                    self.tokenizer,
                    tokenize_speaker=self.tokenizer_speakers,
                    speaker_to_id=self._get_speaker_id,
                    return_spk_ids=self.return_spk_ids
                )

                start_utt = intersecting_utts[0][1]
                end_utt = intersecting_utts[-1][1]

                utt_start = start_utt['utterance_start']
                utt_end = end_utt['utterance_end']

                if utt_end - utt_start > self.segment_size:
                    # Number of tokens corresponding to the starting utterance
                    num_start_tokens = len(
                        self.tokenizer.encode(start_utt['utterance'],
                                              bos_token=False,
                                              eos_token=False
                                              ))
                    num_end_tokens = len(
                        self.tokenizer.encode(end_utt['utterance'],
                                              bos_token=False,
                                              eos_token=False
                                              ))

                    # Amount of time OOB time at start and end
                    start_oob = start_time - utt_start
                    end_oob = utt_end - end_time
                    # % of OOB time for the start utterance
                    start_prct = start_oob / \
                        (start_utt['utterance_end'] -
                         start_utt['utterance_start'])
                    # % of OOB time for the end utterance
                    end_prct = end_oob / \
                        (end_utt['utterance_end'] - end_utt['utterance_start'])

                    # Truncate utterances proportional to audio truncation
                    # Edge cases (beginning/end of episode), the OOB may be negative
                    truncate_start = max(
                        round(start_prct * num_start_tokens), 0)
                    truncate_end = max(round(end_prct * num_end_tokens), 0)
                    assert truncate_start >= 0
                    assert truncate_end >= 0
                    utterance_tokens = utterance_tokens[truncate_start:len(
                        utterance_tokens) - truncate_end]
                    spk_ids = spk_ids[truncate_start:len(
                        spk_ids) - truncate_end]

        # Limit maximum tokens
        utterance_tokens = utterance_tokens[:self.max_tokens]
        spk_ids = spk_ids[:self.max_tokens]
        assert len(spk_ids) == len(utterance_tokens)

        # Display output
        # print()
        # print(self.tokenizer.decode_speakers(utterance_tokens))

        # Load audio
        audio_path = os.path.join(self.data_dir, '{}{}'.format(
            file_stub, self.ext
        ))
        x_wav = load_audio_segment(audio_path, start_time, end_time)
        text = torch.LongTensor(utterance_tokens)
        spk_ids = torch.LongTensor(spk_ids)

        # Also return dataset index for debugging purposes
        return x_wav, text, spk_ids, i


class ASRSegmentCollater:
    """ Collates data """

    def __init__(self,  padding_idx):
        self.padding_idx = padding_idx

    def __call__(self, samples):
        """ Creates a batch out of samples """
        raw_audio, text, idx = zip(*samples)

        # Process Audio
        max_audio_len = max(map(len, raw_audio))
        audio_lens = torch.LongTensor([len(x) for x in raw_audio])

        # Padded
        audio = torch.stack([
            torch.nn.functional.pad(x, (0, max_audio_len - len(x)), value=0) for x in raw_audio
        ])

        # Process Text
        max_text_len = max(map(len, text))
        # Binary mask
        text_mask = torch.BoolTensor(
            [[1] * len(x) + [0] * (max_text_len - len(x)) for x in text])
        # Padded
        text = torch.stack([torch.nn.functional.pad(
            x, (0, max_text_len - len(x)), value=self.padding_idx) for x in text])

        return audio, audio_lens, text, text_mask, idx
