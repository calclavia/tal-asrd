import glob
import json
import os
from bs4 import BeautifulSoup
from tqdm import tqdm
import sentencepiece

uri = '/root/data4/bernard-tal-splits/*/*.jsonl'

for f in tqdm(glob.glob(uri)):
    with open(f, 'r+') as rf:
        uts = [json.loads(u.strip()) for u in rf.read().splitlines()]
    with open(f, 'w+') as wf:
        for u in uts:
            u['utterance'] = BeautifulSoup(u['utterance'], 'lxml').get_text()
            wf.write(json.dumps(u))
            wf.write('\n')

import os
import pickle
import numpy as np
import json
from tqdm import tqdm
from lxml import html
from librosa.core import get_duration
from itertools import chain

train_vocab = []
for split in ['train', 'valid', 'test']:
    with open('/root/data4/tal-data/{}/OLD_transcript.pkl'.format(split), 'rb') as rf:
        transcripts = pickle.load(rf)
    new_transcripts = dict()
    for e, turns in tqdm(transcripts.items()):
        wav_len = get_duration(filename='/root/data4/tal-data/{}/{}.wav'.format(
            split, e
        ))
        for s in ['train', 'valid', 'test']:
            try:
                with open('/root/data4/bernard-tal-splits/{}/{}.jsonl'.format(
                    s, e
                ), 'r+') as rf:
                    turns = [json.loads(l) for l in rf.read().splitlines()]
                    for t in turns:
                        t['utterance'] = html.fromstring(t['utterance']).text_content().strip()
                        if t['utterance_end'] is None or np.isnan(t['utterance_end']):
                            t['utterance_end'] = wav_len
                new_transcripts[e] = turns
                break
            except:
                continue
    if split == 'train':
        train_vocab.extend(list(chain.from_iterable([
            [t['utterance'] for t in turns] for e, turns in new_transcripts.items()
        ])))
    new_loc = '/root/data4/tal-data/{}/transcript.pkl'.format(split)
    with open(new_loc, 'wb') as wf:
        pickle.dump(new_transcripts, wf)
    print('Saved {:,} {} - {:.3f} MB'.format(len(new_transcripts), new_loc, os.path.getsize(new_loc) / 1024 / 1024))

len(train_vocab)

with open('/root/data4/tal-data/tal-vocab-cased.txt', 'w+') as wf:
    for u in train_vocab:
        _ = wf.write(u)
        _ = wf.write('\n')


from sentencepiece import SentencePieceTrainer

SentencePieceTrainer.Train(
    '--input=tal-vocab-cased.txt --model_prefix=taltoken-cased --vocab_size=10000 '
    '--bos_id=0 --eos_id=1 --pad_id=2 --unk_id=3 --character_coverage=1.0 --model_type=bpe'
)



