"""
Because the directory structure is fixed...
"""
import os
import pandas as pd
from tqdm import tqdm
from shutil import copyfile

base_dir = './data/tal-new'
target_dir = './data/tal-spotcheck'

for split in ['train', 'test', 'valid']:
    print(split)
    spc = pd.read_pickle(os.path.join(
        base_dir, '{}-spot-check.pkl'.format(split)))
    for ep in tqdm(spc):
        # Copy over WAV file
        _ = copyfile(
            os.path.join(base_dir, '{}.wav'.format(ep)),
            os.path.join(target_dir, split, '{}.wav'.format(ep))
        )
    _ = copyfile(
        os.path.join(base_dir, '{}-spot-check.pkl'.format(split)),
        os.path.join(target_dir, split, 'transcript.pkl')
    )
