import re
import torch
import torchaudio
import numpy as np
from typing import Callable
from nltk.tokenize import TweetTokenizer
import hashlib

WT = TweetTokenizer()
def tweet_tokenize(stuff):
    return WT.tokenize(stuff)

PUNCTUATOR = re.compile(r'\s+([?.,!\'])')

# Default sampling rate
DEFAULT_SR = 16000

def load_audio_segment(audio_path, start_s, end_s, cache_dir=None):
    """ Loads an audio segment and uses local cache """
    if cache_dir:
        hash_code = '{}_{}_{}'.format(audio_path, start_s, end_s)
        hash_str = hashlib.sha1(hash_code.encode()).hexdigest()
        cached_path = os.path.join(cache_dir, hash_str + '.pt')

        try:
            # Cache hit
            return torch.load(cached_path)
        except FileNotFoundError:
            pass

    # Cache miss
    si, ei = torchaudio.info(audio_path)
    audio_file_sr = si.rate
    offset = int(start_s * audio_file_sr)
    segment_duration = end_s - start_s if end_s is not None else 0.
    segment_frames = int(segment_duration * audio_file_sr)
    x_wav, sr = torchaudio.load(
        audio_path,
        offset=offset,
        num_frames=segment_frames
    )

    # Unlike the docs, x_wav will be normalized to [-1, 1] https://github.com/pytorch/audio/issues/98
    assert sr == audio_file_sr, sr
    if sr != DEFAULT_SR:
        resample = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=DEFAULT_SR)
        x_wav = resample(x_wav)

    x_wav = x_wav.squeeze(0)
    if cache_dir:
        torch.save(x_wav, cached_path)
    return x_wav

def is_valid_utterance(utterance: dict, file_max_duration: float):
    """ Checks if a single utterance is valid """
    # Check if this utterance claims to start out of bounds
    if utterance['utterance_start'] > file_max_duration:
        return False

    # Bad utterance bounds
    if utterance['utterance_start'] > utterance['utterance_end']:
        return False

    # Check if this utterance ends up out of bounds
    if utterance['utterance_end'] is not None \
            and not np.isnan(utterance['utterance_end']) \
            and utterance['utterance_end'] > file_max_duration:
        return False
    return True

def tokenize_utterances(utterances, filtered_utts, tokenizer, add_eot=True, tokenize_speaker=None, speaker_to_id=lambda x: 0, return_spk_ids=False):
    """
    Args:
        tokenize_speaker: Function to tokenize speaker
    """
    # Tokenize a list of contiguous utterances
    utterance_tokens = []
    spk_ids = []

    for i, utt in filtered_utts:
        # Format: <utterance tokens> <speaker token> <eos>
        is_first = utt == utterances[0]
        is_last = utt == utterances[-1]

        if is_first:
            # Start episode with <EOS>
            utterance_tokens.append(tokenizer.eos_token_id)
            if return_spk_ids:
                spk_ids.append(speaker_to_id(utt['speaker']))

        text = utt['utterance'].strip()
        encoded_text = tokenizer.encode(
            text,
            bos_token=False,
            eos_token=False,
        )
        utterance_tokens.extend(encoded_text)
        if return_spk_ids:
            spk_ids.extend([speaker_to_id(utt['speaker'])] * len(encoded_text))

        # Add speaker token
        if tokenize_speaker:
            utterance_tokens.append(speaker_to_id(utt['speaker']))
            if return_spk_ids:
                spk_ids.append(speaker_to_id(utt['speaker']))

        # End turn with an EOS (end turn) token
        utterance_tokens.append(tokenizer.eos_token_id)
        if return_spk_ids:
            spk_ids.append(speaker_to_id(utt['speaker']))

        if is_last and add_eot:
            # End episode with <EOT>
            utterance_tokens.append(tokenizer.eot_token_id)
            if return_spk_ids:
                spk_ids.append(speaker_to_id(utt['speaker']))
    
    if return_spk_ids:
        assert len(spk_ids) == len(utterance_tokens)
        return utterance_tokens, spk_ids

    return utterance_tokens, None

def tokenize_utterances_word_align(
    utterances: list,
    filtered_utts: list,
    start_time: float,
    end_time: float,
    segment_size: float,
    tokenizer,
    add_eot: bool = True,
    tokenize_speaker: Callable = None,
):
    # Start/end bounds for this chunk of utterances & how much longer it is compared to the max
    section_start = filtered_utts[0][1]['utterance_start']
    section_end = filtered_utts[-1][1]['utterance_end']

    # The bounds look like this:
    #   SECTION_START
    #   START_TIME
    #   END_TIME
    #   SECTION_END
    start_bound = max(section_start, start_time)
    end_bound = min(end_time, section_end)

    # Truncate start
    need_start_truncate = False
    first_u = filtered_utts[0][1]
    for w_start, w_end, start_t_ix in first_u['alignments']:
        if w_start >= start_bound:
            need_start_truncate = True
            break

    # Truncate end
    need_end_truncate = False
    last_u_ix = len(filtered_utts) - 1
    last_u = filtered_utts[last_u_ix][1]
    for w_start, w_end, end_t_ix in last_u['alignments'][::-1]:
        if w_end <= end_bound:
            need_end_truncate = True
            break

    # Tokenize
    utterance_tokens = []
    for ix, (i, utt) in enumerate(filtered_utts):
        # Truncate the text if it's the first or last & necessary
        if ix in {0, last_u_ix} and (need_start_truncate or need_end_truncate):
            text = PUNCTUATOR.sub(r'\1', ' '.join(tweet_tokenize(utt['utterance'])[
                (start_t_ix if (ix == 0 and need_start_truncate) else 0):
                (end_t_ix if (ix == last_u_ix and need_end_truncate) else None)
            ]))
        else:
            text = utt['utterance'].strip()

        # Format: <utterance tokens> <speaker token> <eos>
        is_first = (utt == utterances[0] and not need_start_truncate)
        is_last = (utt == utterances[-1] and not need_end_truncate)

        if is_first:
            # Start episode with <EOS>
            utterance_tokens.append(tokenizer.eos_token_id)

        utterance_tokens.extend(
            tokenizer.encode(
                text,
                bos_token=False,
                eos_token=False,
            )
        )

        # Add speaker token
        if tokenize_speaker is not None:
            utterance_tokens.append(tokenize_speaker(utt['speaker']))

        # End turn with an EOS (end turn) token
        utterance_tokens.append(tokenizer.eos_token_id)

        if is_last and add_eot:
            # End episode with <EOT>
            utterance_tokens.append(tokenizer.eot_token_id)

    return utterance_tokens
