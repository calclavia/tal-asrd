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
from .util import DEFAULT_SR

def build_segment_cache_single(fname, data_dir, max_duration):
    audio_file = os.path.join(data_dir, fname)
    # Get audio file metadata
    si, ei = torchaudio.info(audio_file)
    duration = si.length / si.rate

    if duration > max_duration:
        # A list of segments
        # TODO: Use VAD
        return fname, si.rate, [i * max_duration for i in range(int(duration / max_duration))]
    return None

class RandomSegmentDataset(Dataset):
    """
    A dataset that produces random audio segments from a directory
    """

    def __init__(self, data_dir: str, max_duration: float = 10, ext: str = '.wav'):
        """
        Args:
            data_dir: Path to dataset
            max_duration: Number of seconds for an audio segment
            ext: File extension of audio files to scan
        """
        super().__init__()

        self.data_dir = data_dir
        self.max_duration = max_duration

        arghash = (self.max_duration, ext, 3)
        try:
            with open(os.path.join(self.data_dir, 'cache_rs.pkl'), 'rb') as f:
                cach_hash, index = pickle.load(f)
                assert cach_hash == arghash
        except:
            print('Building cache with max seconds:', self.max_duration)
            results = Parallel(n_jobs=16)(delayed(build_segment_cache_single)(
                entry.name, data_dir, self.max_duration) for entry in os.scandir(self.data_dir) if entry.is_file() and entry.name.endswith(ext))
            assert len(results) > 0
            results = [r for r in results if r is not None]
            index = [(r[0], r[1], x) for r in results for x in r[2]]
            assert len(index) > 0
            print('Done. Index size:', len(index))

            with open(os.path.join(self.data_dir, 'cache_rs.pkl'), 'wb') as f:
                pickle.dump((arghash, index), f)

        # Reduce memory requirement.
        fnames, sampling_rates, starts = zip(*index)
        self.fnames = fnames
        self.sampling_rates = np.array(sampling_rates)
        self.starts = np.array(starts)

        self.failed_files = set()

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i: int):
        fname = self.fnames[i]
        sampling_rate = self.sampling_rates[i]
        start = self.starts[i]

        audio_file = os.path.join(self.data_dir, fname)
        end = start + self.max_duration

        si, ei = torchaudio.info(audio_file)
        try:
            x_wav, sr = torchaudio.load(audio_file, offset=int(
                start * sampling_rate), num_frames=int((end - start) * sampling_rate))
        except Exception as e:
            print('Failed to load audio file:', self.fnames[i])
            if self.fnames[i] not in self.failed_files:
                self.failed_files.add(self.fnames[i])
                with open('out/bad_wav.log', 'a+') as f:
                    f.write(self.fnames[i] + '\n')
            return self[random.randint(0, len(self) - 1)]

        # Unlike the docs, x_wav will be normalized to [-1, 1] https://github.com/pytorch/audio/issues/98
        assert sr == sampling_rate, sr

        if sr != DEFAULT_SR:
            resample = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=DEFAULT_SR)
            x_wav = resample(x_wav)

        x_wav = x_wav.squeeze(0)
        return x_wav


class AudioCollator:
    """ Collates data """

    def __call__(self, samples):
        """ Creates a batch out of samples """
        return torch.stack(samples)
