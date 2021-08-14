
import pandas as pd
from collections import defaultdict
import pickle
import json
import numpy as np
from librosa.core import get_duration
import os
from tqdm import tqdm
from joblib import Parallel, delayed

base_dir = './data/librispeech'

speakers = defaultdict(set)
for split in ['train', 'valid', 'test']:
    for noise in ['clean', 'other']:
        t = pd.read_pickle(os.path.join(
            base_dir, split, noise, 'transcript.pkl'))
        for k, v in tqdm(t.items()):
            speakers[split].add(v[0]['speaker'])

speakers = {k: sorted(v) for k, v in speakers.items()}
speaker_map = speakers['train']
speaker_map = dict(zip(speaker_map, range(len(speaker_map))))
print('{:,} total speakers'.format(len(speaker_map)))


for split in ['train', 'valid', 'test']:
    for noise in ['clean', 'other']:
        t_loc = os.path.join(base_dir, split, noise, 'transcript.pkl')
        t = pd.read_pickle(t_loc)
        new_transcripts = dict()
        for k, v in tqdm(t.items()):
            new_transcripts[k] = [{
                'speaker': v[0]['speaker'],
                'speaker_id': speaker_map.get(v[0]['speaker'], -1),
                'utterance': v[0]['utterance'],
                'has_q': v[0]['has_q'],
                'ends_q': v[0]['ends_q'],
                'utterance_start': 0.0,
                'utterance_end': v[0]['utterance_end'],
            }]
        with open(t_loc, 'wb') as wf:
            pickle.dump(new_transcripts, wf)
        print('Dumped {:,} transcripts to {} ({:.2f} MB)'.format(
            len(new_transcripts), t_loc, os.path.getsize(t_loc) / 1024 / 1024
        ))

for split in ['train', 'valid', 'test']:
    for noise in ['clean', 'other']:
        with open(os.path.join(base_dir, split, noise, 'speaker_map.json'), 'w+') as wf:
            json.dump(speaker_map, wf)
