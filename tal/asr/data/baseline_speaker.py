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
from librosa.effects import time_stretch
from librosa.core import get_duration
from mutagen import File
from joblib import Parallel, delayed
import time
import hashlib
import shutil
from .util import DEFAULT_SR, load_audio_segment, tokenize_utterances, is_valid_utterance


def build_index(data_dir: str, file_stub: str, utterances: list,
                min_chunk_duration: float, num_utterances: int, ext: str,
                discontinuity_threshold: float):
    """
    Builds an index for a single audio file's utterances.
    """
    # File stub
    audio_file = os.path.join(data_dir, '{}{}'.format(file_stub, ext))

    # Make sure the corresponding audio file exists
    assert os.path.exists(audio_file), file_stub

    # Find valid utterances within that stub
    try:
        f_duration = get_duration(filename=audio_file)
    except Exception as e:
        f_duration = File(audio_file).info.length

    valid_utts = [u for u in utterances if is_valid_utterance(u, f_duration)]

    # Set nan to end of audio
    for utt in valid_utts:
        if np.isnan(utt['utterance_end']):
            utt['utterance_end'] = f_duration

    index = []

    if num_utterances is None:
        # Provide the full episode
        index.append((file_stub, valid_utts, f_duration))
    else:
        for i in range(len(valid_utts) + 1 - num_utterances):
            segment = valid_utts[i:i + num_utterances]
            if is_valid_segment(
                    segment,
                    discontinuity_threshold=discontinuity_threshold):
                segment_duration = sum(
                    u['utterance_end'] - u['utterance_start'] for u in segment)
                index.append((file_stub, segment, segment_duration))
    return index


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


class SDUtteranceDataset(Dataset):
    """
    Audio -> Text dataset.
    Provides well aligned audio utterances.
    Two special tokens are used.

    Text format:
    Begin episode
    <EOS>
        <utterance tokens><speaker_A><EOS>
        <utterance tokens><speaker_B><EOS>
    ...
    <EOT>
    End episode

    Text format:
    <EOS>
    <utterance tokens><speaker_A><EOS>
    <utterance tokens><speaker_B><EOS>
    """

    def __init__(self,
                 data_dir: str,
                 speaker_map_loc: str = None,
                 ext: str = '.wav',
                 num_utterances: int = 1,
                 min_segment_duration: float = 3,
                 max_segment_duration: float = None,
                 discontinuity_threshold: float = 3):
        """
        Args:
            data_dir (str): Path to directory containing data files.
                            Data folder is expected to contain a 'transcript.pkl' file, which stores
                            a dictionary of audio file stubs and maps to the corresponding transcripts of that audio file.
                            Transcripts are formatted as a list of utterance dictionaries.
            speaker_map_loc (str): Path to JSON containing map of lower case speaker name to ID
            cache_dir (str): A local cache directory to store copies of audio (useful when using networked drives)
            tokenizer: Tokenizer object to encode text into IDs
            num_utterances (int): Number of utterances per sample. If None, then a full episode is provided as a sample.
            min_segment_duration (float): Minimum seconds of a single utterance to consider.
            max_segment_duration (float): Maximum seconds of a single utterance to consider.
            ext (str, optional): Audio file format. Defaults to '.wav'.
            discontinuity_threshold (float, optional): Maximum tolerance (s) for discontinuities in the triplet. Defaults to 0.01.
        """
        super().__init__()

        self.data_dir = data_dir
        self.ext = ext
        self.discontinuity_threshold = discontinuity_threshold
        self.min_segment_duration = min_segment_duration
        self.max_segment_duration = max_segment_duration
        self.num_utterances = num_utterances

        self.speaker_map = None
        if speaker_map_loc:
            # Get speaker map & speaker IDs
            with open(speaker_map_loc, 'r+') as rf:
                self.speaker_map = json.load(rf)
            print('Loaded speakers', len(self.speaker_map))

        # Knows when to rebuild cache
        arghash = (self.num_utterances, self.ext, 11)
        try:
            cache_loc = os.path.join(
                data_dir, 'cache_aligned_{}u.pkl'.format(self.num_utterances))
            with open(cache_loc, 'rb') as f:
                cache = pickle.load(f)
                cach_hash, self.index = cache
                assert cach_hash == arghash
            print('~~~~~~~~~ Loaded index cache from {}'.format(cache_loc))
        except:
            print('~~~~~~~~~ Caching {} with {} utterances'.format(
                self.__class__.__name__, self.num_utterances))
            # Load all transcripts
            with open(os.path.join(data_dir, 'transcript.pkl'), 'rb') as f:
                transcripts = pickle.load(f)

            self.index = Parallel(n_jobs=8)(
                delayed(build_index)(data_dir, file_stub, utts,
                                     min_segment_duration, self.num_utterances,
                                     ext, discontinuity_threshold)
                for file_stub, utts in transcripts.items())
            # Flatten
            self.index = [y for x in self.index for y in x]

            if not self.index:
                raise ValueError('Empty index created!! Bad.')

            with open(
                    os.path.join(
                        data_dir, 'cache_aligned_{}u.pkl'.format(
                            self.num_utterances)), 'wb') as f:
                pickle.dump((arghash, self.index), f)

        # Prune utterances
        self.index = [(stub, utts) for stub, utts, duration in self.index
                      if (self.min_segment_duration is None
                          or duration >= self.min_segment_duration) and (
                              self.max_segment_duration is None
                              or duration < self.max_segment_duration)]
        print('Created {} with {:,} uttterances'.format(
            self.__class__.__name__, len(self)))

    def _get_speaker_id(self, speaker_name: str):
        speaker_name = speaker_name.lower().strip()
        # Unknown speaker gets the last ID
        return self.speaker_map[speaker_name] if speaker_name in self.speaker_map else len(self.speaker_map)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i: int):
        file_stub, utterances = self.index[i]
        utterance = utterances[0]

        audio_path = os.path.join(self.data_dir, '{}{}'.format(
            file_stub, self.ext))

        # Load audio
        start_time = utterance['utterance_start']
        end_time = utterance['utterance_end']
        utterance_tokens = torch.LongTensor(
            [self._get_speaker_id(utterance['speaker'])])

        audio_path = os.path.join(self.data_dir, '{}{}'.format(
            file_stub, self.ext))
        x_wav = load_audio_segment(audio_path, start_time, end_time)

        assert x_wav.size(0) > 0

        # Display output
        # print(i, self.tokenizer.decode_speakers(utterance_tokens))
        # print(x_wav.size())
        # torchaudio.save('out/dump_audio_{}.wav'.format(i), x_wav, DEFAULT_SR)
        # print(x_wav.size())

        return x_wav, utterance_tokens, i


class SDUtteranceCollater:
    """ Collates data """

    def __init__(self,  padding_idx):
        self.padding_idx = padding_idx

    def __call__(self, samples):
        """ Creates a batch out of samples """
        raw_audio, speakers, idx = zip(*samples)

        # Process Audio
        max_audio_len = max(map(len, raw_audio))
        audio_lens = torch.LongTensor([len(x) for x in raw_audio])

        # Padded
        audio = torch.stack([
            torch.nn.functional.pad(x, (0, max_audio_len - len(x)), value=0) for x in raw_audio
        ])

        # Process Text
        max_speaker_len = max(map(len, speakers))
        # Binary mask
        speaker_mask = torch.BoolTensor(
            [[1] * len(x) + [0] * (max_speaker_len - len(x)) for x in speakers])
        # Padded
        speaker = torch.stack([torch.nn.functional.pad(
            x, (0, max_speaker_len - len(x)), value=self.padding_idx) for x in speakers])

        return audio, audio_lens, speaker, speaker_mask, idx
