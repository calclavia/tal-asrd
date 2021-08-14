"""
Input spec (from generation)
# List of examples (6775 or w/e, number of individual snippets)
[
    # Tuple of ref / hyp
    (
        # List of reference utterance dictionaries
        [
            {utterance 1},
            {utterance 2},
            ...
        ],
        # List of hypothesis utterance dictionaries
        [
            {
                'utterance': text,
                'attention': array [n_tokens x 357],
                'chunkstart': array [n_tokens],
                'speakerId': int / None,
            }
            ...
        ]
    ),
    ...
]

Output spec
# List of examples (# episodes)
[
    # Tuple of ref / hyp
    (
        # List of reference (utterance string, speaker ID)
        [
            (ref_turn_1, ref_speaker_1, role),
            (ref_turn_2, ref_speaker_2, role),
            ...
        ],
        # List of hypothesis (utterance string, speaker ID)
        [
            (hyp_turn_1, (hyp_speaker_emb_1, hyp_speaker_id_1), role),
            (hyp_turn_2, (hyp_speaker_emb_2, hyp_speaker_id_2), role),
            ...
        ]
    ),
    ...
]
"""
import argparse
import pickle
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import numpy as np
from datetime import datetime
import json
from itertools import chain
from collections import defaultdict, Counter
from wildspeech.asr.tokenizers.sentencepiece import Tokenizer
from nltk import word_tokenize
from joblib import Parallel, delayed
import torch


def get_hyp_dict_wder(i: int,
                      hyp_dict: dict,
                      role_map: dict,
                      tok: object,
                      ep_sd_features: dict,
                      ep_sd_ids: dict,
                      word_level: bool = False):
    hyps = []
    with torch.no_grad():
        # Device to perform the operation
        device = torch.device('cuda')
        # Convert to tensors
        this_ep_features = torch.from_numpy(
            ep_sd_features[ep]).half().to(device)
        try:
            speaker_id = hyp_dict['speakerId']
            role = role_map.get(speaker_id, 'subject')

            # Unaligned baseline (Separate) generation - word-level alignments of features and IDs
            if word_level:
                # Iterate over words
                words = hyp_dict['utterance'].split()
                u_tok = hyp_dict['utteranceTokens']
                buffer = []
                last_dump_ix = 0

                for i_u, u in enumerate(u_tok):
                    buffer.append(u)
                    if buffer and ' ' in tok.decode(buffer[last_dump_ix:i_u]):
                        # Store word tokens
                        start_word_ix = last_dump_ix
                        end_word_ix = i_u
                        word_tokens = buffer[start_word_ix:end_word_ix]
                        word_str = tok.decode(word_tokens)
                        # Start finding the next word
                        last_dump_ix = i_u
                        # Store the word
                        # for i_word, w in enumerate(words):
                        #     word_tokens = tok.encode(
                        #         w, bos_token=False, eos_token=False)
                        #     # Brute force alignment
                        #     start_word_ix = None
                        #     end_word_ix = None
                        #     for u_ix in range(start_ix, len(u_tok)):
                        #         found = False
                        #         for word_width in range(1, len(word_tokens) + 2):
                        #             if tok.decode(u_tok[u_ix:u_ix+word_width]).strip() == w.strip():
                        #                 start_word_ix = u_ix
                        #                 end_word_ix = u_ix + word_width
                        #                 start_ix = end_word_ix
                        #                 found = True
                        #                 break
                        #         if found:
                        #             break
                        if start_word_ix is None:
                            print('Word: "{}"'.format(w))
                            print('Word tokens: {}'.format(word_tokens))
                            print('start_ix: {}/{}'.format(start_ix, len(u_tok)))
                            print('Remaining tokens: {}'.format(
                                u_tok[start_ix:]))
                            print('Utterance tokens starting w/ first index ({}-):\n{}'.format(
                                start_ix,
                                '\n'.join(
                                    '({}) {} -> "{}"'.format(
                                        i,
                                        u_tok[i:i+len(word_tokens)],
                                        tok.decode(
                                            u_tok[i:i+len(word_tokens)]),
                                    ) for i in range(start_ix, len(u_tok)) if u_tok[i] == word_tokens[0]
                                )
                            ))
                            print('Utterance tokens starting w/ first index:\n{}'.format(
                                '\n'.join(
                                    '({}) {} -> "{}"'.format(
                                        i,
                                        u_tok[i:i+len(word_tokens)],
                                        tok.decode(
                                            u_tok[i:i+len(word_tokens)]),
                                    ) for i in range(len(u_tok)) if u_tok[i] == word_tokens[0]
                                )
                            ))
                            raise Exception(
                                'Word not found in utterance tokens!')

                        a_weights = hyp_dict['attention'][
                            start_word_ix:end_word_ix]
                        chunk_starts = hyp_dict['chunkStart'][
                            start_word_ix:end_word_ix]
                        # Find all weighted embeddings corresponding to this word
                        speaker_weights = defaultdict(float)
                        # TODO: Tensorize this
                        word_emb = []
                        try:
                            for aw, cs in zip(a_weights, chunk_starts):
                                # Get weighted sum of features
                                feature_chunk = this_ep_features[cs:cs + 357]
                                aw = aw.half().to(device)
                                word_emb.append((aw[:len(feature_chunk)],
                                                feature_chunk))

                                # Get accumulated attentions for speaker IDs
                                spk_id_chunk = ep_sd_ids[ep][cs:cs + 357]
                                for w, sid in zip(aw[:len(spk_id_chunk)],
                                                  spk_id_chunk):
                                    speaker_weights[sid] += w.item()

                            a, b = zip(*word_emb)
                            a = torch.stack(a, dim=0)
                            b = torch.stack(b, dim=0)
                            word_emb = torch.matmul(
                                a.unsqueeze(-2), b).squeeze(1)
                        except:
                            print('word_emb:\n{}'.format(word_emb))
                            print('w: {}'.format(w))
                            print('utterance: {}'.format(
                                hyp_dict['utterance']))
                            print('len(word_tokens): {}'.format(
                                len(word_tokens)))
                            print('a_weights: {}'.format(a_weights))
                            print('chunk_starts: {}'.format(chunk_starts))
                            print('attention size: {}'.format(
                                hyp_dict['attention'].shape))
                            print('chunkStart size: {}'.format(
                                hyp_dict['chunkStart'].shape))
                            print('start_ix: {}'.format(start_word_ix))
                            print('Word {}/{}'.format(start_word_ix, len(words)))
                            print('u_tok: {}'.format(u_tok))
                            raise

                        # Get most heavily weighted speaker ID across the word
                        word_spk_id = sorted(
                            speaker_weights.items(), key=lambda x: x[1])[-1][0]
                        word_role = role_map.get(speaker_id, 'subject')
                        # Insert the single word as an "utterance" block
                        hyps.append(
                            (word_str, (word_emb.cpu(), word_spk_id), word_role))
            # We assume segmentation / speaker change detected from EOS tokens
            else:
                hyp_emb = []
                for aw, cs in zip(hyp_dict['attention'],
                                  hyp_dict['chunkStart']):
                    feature_chunk = this_ep_features[cs:cs + 357]
                    aw = aw.half().to(device)
                    hyp_emb.append((aw[:len(feature_chunk)], feature_chunk))

                a, b = zip(*hyp_emb)
                a = torch.stack(a, dim=0)
                b = torch.stack(b, dim=0)
                hyp_emb = torch.matmul(a.unsqueeze(-2), b).squeeze(1)

                hyps.append((hyp_dict['utterance'], (hyp_emb.cpu(),
                                                     speaker_id), role))
        except:
            print('episode: {}'.format(ep))
            print('hyp_dict keys: {}'.format(hyp_dict.keys()))
            print('Utterance: {}'.format(hyp_dict['utterance']))
            print(hyp_dict)
            raise

    return i, hyps


"""
git clone https://github.com/calclavia/wild-speech.git

export MODEL=tal-tds-speaker-3
export MODEL=tal-tds-libri-ft
export MODEL=tal-libri-lm-ft
export MODEL=tal-tds-baseline-1

export MODEL_DIR=./models/wild-speech/${MODEL}

# ALIGNED
python -u -m wildspeech.utils.aligned_to_wder_format --in-file ${MODEL_DIR}/aligned/test_result.pkl --out-file ${MODEL_DIR}/aligned/wder_ready.pkl

# UNALIGNED
python -u -m wildspeech.utils.aligned_to_wder_format --in-file ${MODEL_DIR}/unaligned/test_result.pkl --out-file ${MODEL_DIR}/unaligned/wder_ready.pkl --unaligned --workers 10

python3 -u -m wildspeech.utils.aligned_to_wder_format --in-file /home/shuyang/data4/tal-new/test_result.pkl --out-file /home/shuyang/data4/tal-new/wder_ready.pkl --unaligned --workers 15 --role-map /home/shuyang/data4/tal-new/role_id_map_TALTDS.json --speaker-id-hyp /home/shuyang/data4/tal-new/hyp_speaker_ids.pkl --speaker-feat-hyp /home/shuyang/data4/tal-new/hyp_speaker_features.pkl --cache-path /home/shuyang/data4/tal-new/taltoken-cased.model
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-file', type=str, required=True)
    parser.add_argument(
        '--speaker-id-hyp',
        type=str,
        default='./data/tal-final/test/hyp_speaker_ids.pkl')
    parser.add_argument(
        '--speaker-feat-hyp',
        type=str,
        default='./data/tal-final/test/hyp_speaker_features.pkl')
    parser.add_argument('--out-file', type=str, required=True)
    parser.add_argument('--unaligned', action='store_true', default=False)
    parser.add_argument(
        '--cache-path',
        type=str,
        default='./cache/tokenizer/taltoken-cased.model')
    parser.add_argument(
        '--role-map',
        type=str,
        default='./data/tal-final/role_id_map.json')
    parser.add_argument('--word-level', action='store_true', default=False)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()

    start = datetime.now()

    ep_sd_features = pd.read_pickle(args.speaker_feat_hyp)
    print('{} - Loaded baseline SD features for {:,} episodes'.format(
        datetime.now() - start, len(ep_sd_features)))

    ep_sd_ids = pd.read_pickle(args.speaker_id_hyp)
    print('{} - Loaded baseline SD IDs for {:,} episodes'.format(
        datetime.now() - start, len(ep_sd_ids)))

    with open(args.role_map, 'r+') as rf:
        role_map = {int(k): v for k, v in json.load(rf).items()}

    tok = Tokenizer(cache_path=args.cache_path)

    with open(args.in_file, 'rb') as rf:
        utterances = pickle.load(rf)
    print('{} - Loaded {:,} utterances from {} ({:.3f} MB)'.format(
        datetime.now() - start, len(utterances), args.in_file,
        os.path.getsize(args.in_file) / 1024 / 1024))

    STRIDE = 0.08
    FRAMEW = 1.41

    with torch.no_grad():
        # ALIGNED
        if not args.unaligned:
            print('!!!!!!!! ALIGNED !!!!!!!!')
            # Group by episode
            episode_refs = defaultdict(list)
            episode_hyps = defaultdict(list)
            for ii, (ref_utterances, hyp_dicts) in enumerate(tqdm(utterances)):
                # Somehow the hyp now generates a dict on every EOS?
                ref_dict = ref_utterances[0]
                ep = ref_dict['episode']

                # Device to perform the operation
                device = torch.device('cuda')
                # Convert to tensors
                this_ep_features = torch.from_numpy(
                    ep_sd_features[ep]).half().to(device)

                # Get episode alignment information
                u_start = ref_dict['utterance_start']
                u_end = ref_dict['utterance_end']
                st_frame = int(u_start / 0.08)
                e_frame = max(int(max(0.0, u_end - 1.0) / 0.08), st_frame + 1)

                # Reference always well-aligned
                episode_refs[ep].append((
                    u_start,
                    ref_dict['utterance'],
                    ref_dict['speaker'],
                    ref_dict['role'],
                ))

                valid_hyp_dicts = [
                    h for h in hyp_dicts if h['utterance'].strip()
                ]
                if not valid_hyp_dicts:
                    continue
                if len(valid_hyp_dicts) != 1:
                    print('???? - {} | {}: {}'.format(ep, ii, hyp_dicts))
                else:
                    hyp_dict = valid_hyp_dicts[0]

                # Hypothesis?
                hyp_speaker_id = hyp_dict.get('speakerId')
                # Majority vote from separate SD system
                if hyp_speaker_id is None and ep_sd_ids:
                    pred_chunk_ids = ep_sd_ids[ep][st_frame:e_frame]
                    hyp_speaker_id = Counter(pred_chunk_ids).most_common(1)[0][
                        0]

                hyp_speaker_attn = hyp_dict.get('attention')  # [tokens x 357]
                hyp_speaker_chunk_start = hyp_dict.get(
                    'chunkstart')  # [tokens]
                # Full embeddings
                if hyp_speaker_attn is None and ep_sd_features:
                    hyp_speaker_emb = this_ep_features[st_frame:e_frame]
                # Weighted sum over tokens
                else:
                    hyp_speaker_emb = []
                    for aw, cs in zip(hyp_dict['attention'],
                                      hyp_dict['chunkStart']):
                        feature_chunk = this_ep_features[cs:cs + 357]
                        aw = aw.half().to(device)
                        hyp_speaker_emb.append((aw[:len(feature_chunk)],
                                                feature_chunk))
                    # Torchify
                    a, b = zip(*hyp_speaker_emb)
                    a = torch.stack(a, dim=0)
                    b = torch.stack(b, dim=0)
                    hyp_speaker_emb = torch.matmul(a.unsqueeze(-2),
                                                   b).squeeze(1)

                # Episode
                episode_hyps[ep].append((
                    u_start,
                    hyp_dict['utterance'],
                    (hyp_speaker_emb.cpu(), hyp_speaker_id),
                    ref_dict['role'],
                ))

            # Reformatting
            wder_input = []
            for e in episode_refs:
                ref_examples = [(r_u, r_s, r_r)
                                for st, r_u, r_s, r_r in sorted(
                                    episode_refs[e], key=lambda x: x[0])]
                hyp_examples = [(h_u, h_s, h_r)
                                for st, h_u, h_s, h_r in sorted(
                                    episode_hyps[e], key=lambda x: x[0])]
                wder_input.append((ref_examples, hyp_examples))

        # UNALIGNED
        else:
            print('~~~~~~ UNALIGNED ~~~~~~')
            # Group by episode
            episode_refs = defaultdict(list)
            episode_hyps = defaultdict(list)
            for ref_utterances, hyp_dicts in utterances:
                ep = ref_utterances[0]['episode']
                for ref_dict in ref_utterances:
                    # Get episode alignment information
                    ep = ref_dict['episode']
                    u_start = ref_dict['utterance_start']

                    # Reference always well-aligned
                    episode_refs[ep].append((
                        ref_dict['utterance'],
                        ref_dict['speaker'],
                        ref_dict['role'],
                    ))

                wder_hyp_dicts = Parallel(n_jobs=args.workers)(
                    delayed(get_hyp_dict_wder)(
                        i=ii,
                        hyp_dict=hyp_dict,
                        role_map=role_map,
                        tok=tok,
                        ep_sd_features=ep_sd_features,
                        ep_sd_ids=ep_sd_ids,
                        word_level=args.word_level,
                    ) for ii, hyp_dict in tqdm(
                        enumerate(hyp_dicts), total=len(hyp_dicts))
                    if hyp_dict['utterance'])
                wder_hyp_dicts = list(
                    chain.from_iterable([
                        h
                        for i, h in sorted(wder_hyp_dicts, key=lambda x: x[0])
                    ]))
                episode_hyps[ep].extend(wder_hyp_dicts)

            # Reformatting
            wder_input = []
            for e in episode_refs:
                wder_input.append((episode_refs[e], episode_hyps[e]))

        # Save it all!
        with open(args.out_file, 'wb') as wf:
            pickle.dump(wder_input, wf)
        print('Dumped {:,} wder inputs to {} ({:.3f} MB)'.format(
            len(wder_input), args.out_file,
            os.path.getsize(args.out_file) / 1024 / 1024))
