import os
import json
import numpy as np
from dateutil import parser
from tqdm import tqdm

dd = '/workspace/librispeech-splits'
vocab_file = '/workspace/librispeech-vocab.txt'

# Create pkl files for train/test/valid and clean/other
big_pkl = '/workspace/librispeech_splits.pkl'

with open('vocab_file', 'w+') as wf:
    wf.write('')

def convert_time(t_s):
    if isinstance(t_s, str):
        d = parser.parse(t_s)
        return (d.hour or 0) * 3600 + (d.minute or 0) * 60 + (d.second or 0) + (d.microsecond) / 1e6
    elif isinstance(t_s, (int, float)):
        return float(t_s)
    raise Exception('Unexpected input type: {} {}'.format(type(t_s), t_s))

# Convert
for split in ['train', 'test', 'valid']:
    for d_type in ['clean', 'other']:
        path = os.path.join(dd, split, d_type)
        if not os.path.exists(path):
            continue
        for f in tqdm(os.listdir(path)):
            f_path = os.path.join(path, f)
            temp_path = os.path.join(path, 'TEMP')
            out_lines = []
            with open(f_path, 'r+') as rf:
                for line in rf:
                    line_json = json.loads(line.strip())
                    # Lower case
                    line_json['utterance'] = line_json['utterance'].lower()
                    # Convert start times
                    line_json['utterance_start'] = convert_time(line_json['utterance_start'])
                    # Convert end times
                    line_json['utterance_end'] = convert_time(line_json['utterance_end'])
                    out_lines.append(line_json)
                    # Write to total vocab file
                    with open('vocab_file', 'a+') as af:
                        af.write(line_json['utterance'])
                        af.write('\n')
            with open(temp_path, 'w+') as wf:
                for line in out_lines:
                    wf.write(json.dumps(line))
                    wf.write('\n')
            os.remove(f_path)
            os.rename(temp_path, f_path)

"""
mc mirror alexa/bernard-librispeech-splits librispeech-splits --exclude "*.wav"

run stuff

mc mirror librispeech-splits alexa/bernard-librispeech-splits --overwrite
"""