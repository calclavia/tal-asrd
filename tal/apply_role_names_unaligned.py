import torch
import torch.nn as nn
import os
import pandas as pd
from tqdm import tqdm
import random
import pickle

"""
INPUT:
[
    # One episode
    (
        # Reference utterances
        [
            (ref_turn_1, ref_speaker_id),
            (ref_turn_2, ref_speaker_id),
            ...
        ],
        # Hypothesis utterances
        [
            (hyp_turn_1, (hyp_speaker_emb, hyp_speaker_id)),
            (hyp_turn_2, (hyp_speaker_emb, hyp_speaker_id)),
            ...
        ]
    ),
    ...
]

OUTPUT:
[
    # One episode
    (
        # Reference utterances
        [
            (ref_turn_1, ref_speaker_id, ref_role),
            (ref_turn_2, ref_speaker_id, ref_role),
            ...
        ],
        # Hypothesis utterances
        [
            (hyp_turn_1, (hyp_speaker_emb, hyp_speaker_id), hyp_role),
            (hyp_turn_2, (hyp_speaker_emb, hyp_speaker_id), hyp_role),
            ...
        ]
    ),
    ...
]

# tal-tds-speaker-3 (Speaker)
python3 -u -m wildspeech.apply_role_names_unaligned --in-file /home/shuyang/data4/tal-new/speaker_unaligned.pkl --out-file /home/shuyang/data4/tal-new/speaker_unaligned_ordered.pkl --role-map /home/shuyang/data4/tal-new/role_id_map_TALTDS.json --speaker-map /home/shuyang/data4/tal-new/full_speaker_map_old.json

# tal-tds-libri-ft (LibriEncoder)
python3 -u -m wildspeech.apply_role_names_unaligned --in-file /home/shuyang/data4/tal-new/fusion_unaligned.pkl --out-file /home/shuyang/data4/tal-new/fusion_unaligned_ordered.pkl --role-map /home/shuyang/data4/tal-new/role_id_map.json --speaker-map /home/shuyang/data4/tal-new/ALL_speaker_map.json

# tal-tds-libri-ft (LM Decoding)
python3 -u -m wildspeech.apply_role_names_unaligned --in-file /home/shuyang/data4/tal-new/fusion_unaligned.pkl --out-file /home/shuyang/data4/tal-new/fusion_unaligned_ordered.pkl --role-map /home/shuyang/data4/tal-new/role_id_map.json --speaker-map /home/shuyang/data4/tal-new/ALL_speaker_map.json

# tal-libri-lm-ft (LM Fusion)
python3 -u -m wildspeech.apply_role_names_unaligned --in-file /home/shuyang/data4/tal-new/fusion_unaligned.pkl --out-file /home/shuyang/data4/tal-new/fusion_unaligned_ordered.pkl --role-map /home/shuyang/data4/tal-new/role_id_map.json --speaker-map /home/shuyang/data4/tal-new/ALL_speaker_map.json

# tal-libri-lm-sa (ShiftAug)
python3 -u -m wildspeech.apply_role_names_unaligned --in-file /home/shuyang/data4/tal-new/fusion_unaligned.pkl --out-file /home/shuyang/data4/tal-new/fusion_unaligned_ordered.pkl --role-map /home/shuyang/data4/tal-new/role_id_map.json --speaker-map /home/shuyang/data4/tal-new/ALL_speaker_map.json
"""
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser('Label roles for unaligned')
    parser.add_argument('--in-file', type=str, required=True)
    parser.add_argument('--out-file', type=str, required=True)
    parser.add_argument('--speaker-map', type=str, required=True)
    parser.add_argument('--role-map', type=str, required=True)
    parser.add_argument('--hyp-role-map', type=str, default=None)
    args = parser.parse_args()

    # Role map
    with open(args.role_map, 'r+') as rf:
        role_map = {int(k): v for k, v in json.load(rf).items()}
    print('Loaded role map from {} with {:,} speakers'.format(
        args.role_map, len(role_map)
    ))
    hyp_rloc = args.hyp_role_map or args.role_map
    with open(hyp_rloc, 'r+') as rf:
        hyp_role_map = {int(k): v for k, v in json.load(rf).items()}
    print('Loaded hypothesis role map from {} with {:,} speakers'.format(
        hyp_rloc, len(hyp_role_map)
    ))

    # Speaker map
    with open(args.speaker_map, 'r+') as rf:
        speaker_map = {int(v): k for k, v in json.load(rf).items()}
    print('Loaded speaker name map from {} with {:,} speakers'.format(
        args.speaker_map, len(speaker_map)
    ))

    # Load file
    in_u = pd.read_pickle(args.in_file)
    print('Loaded {:,} utterances from {} {:.2f} MB'.format(
        len(in_u), args.in_file, os.path.getsize(args.in_file) / 1024 / 1024
    ))

    # Process file
    for ep in tqdm(range(len(in_u))):
        # References
        for i in range(len(in_u[ep][0])):
            ref_u, ref_spk = in_u[ep][0][i]
            try:
                in_u[ep][0][i] = (ref_u, speaker_map.get(ref_spk), role_map.get(ref_spk, 'host'))
            except:
                print('{}:\n{}'.format(ref_spk, ref_u))
                raise
            assert len(in_u[ep][0][i]) == 3
        # Hypotheses
        for i in range(len(in_u[ep][1])):
            hyp_u, (hyp_emb, hyp_spk) = in_u[ep][1][i]
            in_u[ep][1][i] = (hyp_u, (hyp_emb, hyp_spk), hyp_role_map.get(hyp_spk))
            assert len(in_u[ep][1][i]) == 3

    # Save output
    with open(args.out_file, 'wb') as wf:
        pickle.dump(in_u, wf)
    print('Saved {:,} role-augmented speaker utterances to {} ({:.2f} MB)'.format(
        len(in_u), args.out_file, os.path.getsize(args.out_file) / 1024 / 1024
    ))
