import os
from shutil import copy2

common_folder = './librispeech'

for split in ['train', 'valid', 'test']:
    for hygiene in ['clean', 'other']:
        o_location = os.path.join(
            common_folder, split, hygiene, 'transcript.pkl')
        temp_loc = '/workspace/{}-{}-transcript.pkl'
