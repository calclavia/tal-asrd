from nltk.tokenize import word_tokenize, TweetTokenizer
import editdistance
import random
import numpy as np
from edit_distance import SequenceMatcher
from itertools import chain
import hdbscan
from collections import defaultdict
from joblib import Parallel, delayed

# Some segments of code taken from https://github.com/google/uis-rnn/blob/master/uisrnn/evals.py
from scipy import optimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize

WT = TweetTokenizer()


def tweet_tokenize(stuff):
    return WT.tokenize(stuff)


def get_list_inverse_index(unique_ids: list):
    """Get value to position index from a list of unique ids.
    Args:
        unique_ids: A list of unique integers of strings.
    Returns:
        result: a dict from value to position
    Raises:
        TypeError: If unique_ids is not a list.
    """
    if not isinstance(unique_ids, list):
        raise TypeError('unique_ids must be a list')
    result = dict()
    for i, unique_id in enumerate(unique_ids):
        result[unique_id] = i
    return result


def compute_sequence_match(sequence1: list, sequence2: list):
    """Compute the accuracy between two sequences by finding optimal matching.
    Args:
        sequence1: A list of integers or strings.
        sequence2: A list of integers or strings.
    Returns:
        accuracy: sequence matching accuracy as a number in [0.0, 1.0]
    Raises:
        TypeError: If sequence1 or sequence2 is not list.
        ValueError: If sequence1 and sequence2 are not same size.
    """
    if not isinstance(sequence1, list) or not isinstance(sequence2, list):
        raise TypeError('sequence1 and sequence2 must be lists')
    if not sequence1 or len(sequence1) != len(sequence2):
        raise ValueError(
            'sequence1 and sequence2 must have the same non-zero length')
    # get unique ids from sequences
    unique_ids1 = sorted(set(sequence1))
    unique_ids2 = sorted(set(sequence2))
    inverse_index1 = get_list_inverse_index(unique_ids1)
    inverse_index2 = get_list_inverse_index(unique_ids2)
    # get the count matrix
    count_matrix = np.zeros((len(unique_ids1), len(unique_ids2)))
    for item1, item2 in zip(sequence1, sequence2):
        index1 = inverse_index1[item1]
        index2 = inverse_index2[item2]
        count_matrix[index1, index2] += 1.0
    row_index, col_index = optimize.linear_sum_assignment(-count_matrix)
    optimal_match_count = count_matrix[row_index, col_index].sum()
    accuracy = optimal_match_count / len(sequence1)
    # rows = sequence 1 labels = reference speakers
    # cols = sequence 2 labels = hypothesis speakers
    return row_index, col_index, accuracy


def cluster_speakers(speaker_representations: list, **kwargs):
    # Set "None" to next speaker vector
    engine = hdbscan.HDBSCAN(**kwargs)
    speaker_labels = engine.fit_predict(speaker_representations)
    return speaker_labels


def convert_to_wder_format(speaker_utterances,
                           wer_only: bool,
                           tokenizer=word_tokenize,
                           should_cluster: bool = False,
                           **kwargs):
    """
    Input - list of tuples:
        (utterance string, speaker ID)
        OR
        (utterance string, speaker vectors)
    Return - list of tuples:
        (word, speaker)
    """
    # Determine whether to cluster
    to_cluster = True
    if wer_only:
        to_cluster = False
    # (Speaker embedding, speaker ID) provided
    elif isinstance(speaker_utterances[0][-1], tuple):
        speaker_utterances = [(utt, spk_e if should_cluster else spk_i)
                              for utt, (spk_e, spk_i) in speaker_utterances]
        to_cluster = should_cluster
    elif isinstance(speaker_utterances[0][-1], (int, str, type(None))):
        to_cluster = False
    # Fill in Nones
    s_u_filled = []
    for i, (u, s) in enumerate(speaker_utterances):
        curr_speaker = s
        if curr_speaker is None:
            for _, future_speaker in speaker_utterances[i + 1:]:
                if future_speaker is not None:
                    curr_speaker = future_speaker
                    break
        if curr_speaker is None:
            if to_cluster:
                curr_speaker = np.array([0.0] * len(s_u_filled[0][1])
                                        if s_u_filled else [0.0])
            else:
                curr_speaker = -1
        s_u_filled.append((u, curr_speaker))
    # Do clustering if necessary
    if to_cluster:
        sample = None
        for h in speaker_utterances:
            if not isinstance(h[-1], type(None)):
                sample = h[-1]
                break
        print('Clustering (e.g. {})'.format(sample))
        clusters = cluster_speakers([h[-1] for h in speaker_utterances],
                                    **kwargs)
        speaker_utterances = [(h_u, clusters[i])
                              for i, (h_u, _) in enumerate(speaker_utterances)]
    assert len(speaker_utterances) > 0
    # Convert to relative speaker format
    all_speakers = []
    w_s_tuples = []
    for u, speaker in speaker_utterances:
        try:
            spk_cluster = all_speakers.index(speaker)
        except ValueError:
            spk_cluster = len(all_speakers)
            all_speakers.append(speaker)
        w_s_tuples.extend([(w, spk_cluster) for w in tokenizer(u)])
    n_speakers = len(all_speakers)
    return w_s_tuples, n_speakers


def calculate_wer(ref, hyp):
    # Get words and aligned speaker tokens
    ref_words, ref_spk = zip(*ref)
    if len(hyp) > 0:
        hyp_words, hyp_spk = zip(*hyp)
    else:
        hyp_words = []
        hyp_spk = []
    # Word error rate - levenshtein between reference & hypothesis overall
    asr_dist = editdistance.eval(ref_words, hyp_words)
    n_ref = len(ref_words)
    wer = asr_dist / n_ref
    return wer, asr_dist, n_ref


def calculate_wder(seg_id, ref, hyp, wer_only: bool = False):
    """
    https://arxiv.org/pdf/1907.05337.pdf
    WDER = ( (# ASR substitutions, corect speakers) + (# correct ASR, correct speakers) )
           -------------------------------------------------------------------------------
            ( (# ASR substitutions) + (# correct ASR) )
    = (s_is + c_is) / (s + c)

    For corpus-level WDER:
    sum s_is, c_is, s, c across each document in the corpus
    """
    # Get words and aligned speaker tokens
    ref_words, ref_spk = zip(*ref)
    if len(hyp) > 0:
        hyp_words, hyp_spk = zip(*hyp)
    else:
        hyp_words = []
        hyp_spk = []
    # ref_spk_labels, hyp_spk_labels, match_accuracy = compute_sequence_match(
    #     list(ref_spk),
    #     list(hyp_spk)
    # )
    # hyp_speaker_map = dict(zip(hyp_spk_labels, ref_spk_labels))
    # Word error rate - levenshtein between reference & hypothesis overall
    asr_dist = editdistance.eval(ref_words, hyp_words)
    n_ref = len(ref_words)
    wer = asr_dist / n_ref
    # Short circuit for WER only case
    if wer_only:
        wder = 1e8
        ref_spk_labels = None
        hyp_spk_labels = None
    else:
        sm = SequenceMatcher(a=ref_words, b=hyp_words)
        # Find substitutions
        substitutions = [[r0, r1, h0, h1]
                         for e_type, r0, r1, h0, h1 in sm.get_opcodes()
                         if e_type == 'replace']
        sub_RvH = list(
            chain.from_iterable([
                list(zip(ref_spk[r0:r1], hyp_spk[h0:h1]))
                for r0, r1, h0, h1 in substitutions
            ]))
        correct = [[r0, r1, h0, h1]
                   for e_type, r0, r1, h0, h1 in sm.get_opcodes()
                   if e_type == 'equal']
        corr_RvH = list(
            chain.from_iterable([
                list(zip(ref_spk[r0:r1], hyp_spk[h0:h1]))
                for r0, r1, h0, h1 in correct
            ]))
        print('{} - Got {:,} total edit operations (S+C)'.format(
            seg_id, len(substitutions + correct)))
        # Values for WDER
        # s = len(sub_RvH)
        # c = len(corr_RvH)
        # s_is = len([r for r, h in sub_RvH if r != hyp_speaker_map.get(h)])
        # c_is = len([r for r, h in corr_RvH if r != hyp_speaker_map.get(h)])
        # wder = (s_is + c_is) / (s + c)
        # Optimal WDER
        sub_R, sub_H = map(list, zip(*sub_RvH))
        cor_R, cor_H = map(list, zip(*corr_RvH))
        ref_spk_labels, hyp_spk_labels, match_accuracy = compute_sequence_match(
            sub_R + cor_R,
            sub_H + cor_H,
        )
        wder = 1 - match_accuracy
    print('{} - {:,} hyp speakers, WDER: {:.2f}'.format(
        seg_id, len(set(hyp_spk)), wder * 100.0))
    return wer, asr_dist, n_ref, wder, ref_spk_labels, hyp_spk_labels


def wder_segment(seg_id,
                 ref_us,
                 hyp_us,
                 wer_only,
                 should_cluster: bool = False,
                 **kwargs):
    """
    Computes WDER for a single segment
    """
    ref, n_ref_speakers = convert_to_wder_format(ref_us, wer_only=True)
    hyp, n_hyp_speakers = convert_to_wder_format(
        hyp_us, wer_only=wer_only, should_cluster=should_cluster, **kwargs)
    print('{} - {:,} hypothesis speakers'.format(seg_id, n_hyp_speakers))
    wer, asr_dist, n_ref, wder, ref_spk_labels, hyp_spk_labels = calculate_wder(
        seg_id, ref, hyp, wer_only)
    # %TODO: Debug
    # print('wer ({}): {}\n'.format(type(wer), wer))
    # print('asr_dist ({}): {}\n'.format(type(asr_dist), asr_dist))
    # print('n_ref ({}): {}\n'.format(type(n_ref), n_ref))
    return [asr_dist, n_ref], [ref_spk_labels, hyp_spk_labels], wder


def corpus_wder(paired_results,
                wer_only: bool = False,
                workers: int = 1,
                should_cluster: bool = False,
                **kwargs):
    # Calculate stats for each hypothesis and reference
    # Empty reference = ill-formed speaker label
    # Empty hypothesis = nonterminated, thus 0 substitutions and 0 correct
    results = Parallel(n_jobs=workers)(
        delayed(wder_segment)
        (i, ref_us, hyp_us, wer_only, should_cluster=should_cluster, **kwargs)
        for i, (ref_us, hyp_us) in tqdm(
            enumerate(paired_results), total=len(paired_results))
        if ref_us and hyp_us)
    wer_components, wder_components, wders = zip(*results)
    # print('wer_components ({}): {}\n'.format(type(wer_components), wer_components))

    # WDER
    for i, wder in enumerate(wders):
        print('{} WDER:\t{:.2f}'.format(i, wder * 100.0))
    ref_spk_t, hyp_spk_t = zip(*wder_components)
    overall_wder = np.mean(wders)
    print('Overall WDER: {:.3f}%'.format(100.0 * overall_wder))
    # WER
    asr_dist_t, n_words_t = zip(*wer_components)
    # print('asr_dist_t ({}): {}\n'.format(type(asr_dist_t), asr_dist_t))
    # print('n_words_t ({}): {}\n'.format(type(n_words_t), n_words_t))
    overall_wer = sum(asr_dist_t) / sum(n_words_t)
    print('Overall WER: {:.3f}%'.format(100.0 * overall_wer))
    return ref_spk_t, hyp_spk_t, overall_wder, asr_dist_t, n_words_t, overall_wer


def cosine_similarity(x, y, **kwargs):
    dot_products = np.dot(x, y)
    norm_products = np.linalg.norm(x) * np.linalg.norm(y)
    return (dot_products / (norm_products + 1e-8))


def cosine_distance(x, y, **kwargs):
    return 1 - cosine_similarity(x, y)


def inverse_dot_product(x, y, **kwargs):
    return 1.0 / (np.dot(x, y) + 1e-8)


def neg_dot_product(x, y, **kwargs):
    return -np.dot(x, y)


"""
Usage:
    python -u -m wildspeech.wder --eval-file ./models/tal2tal_results.pkl --metric cos_dist

Expects a results pickle with the following structure:
# List of examples
[
    # Tuple of ref / hyp
    (
        # List of reference (utterance string, speaker ID)
        [
            (ref_turn_1, ref_speaker_1),
            (ref_turn_2, ref_speaker_2),
            ...
        ],
        # List of hypothesis (utterance string, speaker ID)
        [
            (hyp_turn_1, (hyp_speaker_emb_1, hyp_speaker_id_1)),
            (hyp_turn_2, (hyp_speaker_emb_2, hyp_speaker_id_2)),
            ...
        ]
    ),
    ...
]

e.g.
[
    (
        # Ref
        [
            ('banana', 'jack'),
            ('try', 'margaret'),
            ('garbage', 'jack'),
            ('barfagus', 'margaret'),
        ],
        # Hyp
        [
            ('bert', np.array([1.0, 2.0, 3.0, 4.0])),
            ('ernie', None),
            ('garage', np.array([1.0, 2.0, 3.0, 5.0])),
            ('bertfungus', None),
        ]
    )
]

python3 -u -m wildspeech.wder --eval-file /home/shuyang/data4/tal-new/synced_ordered_test.pkl --workers 10 --metric cos_dist

"""
if __name__ == "__main__":
    import argparse
    import os
    import pickle
    import time
    from tqdm import tqdm

    parser = argparse.ArgumentParser('WDER calculation')
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--wer-only', action='store_true', default=False)
    parser.add_argument('--grid-search', action='store_true', default=False)
    parser.add_argument(
        '--tokenizer', type=str, choices=['punkt', 'tweet'], default='punkt')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--cluster', action='store_true', default=False)
    parser.add_argument(
        '--metric',
        choices=['euclidean', 'cos_sim', 'cos_dist', 'idp', 'ndp'],
        required=True)
    args = parser.parse_args()

    print('--------\n{}{}Evaluating WER/WDER from {} with {} and {}\n--------'.
          format(
              '(WER ONLY) ' if args.wer_only else '',
              '(Grid Search) ' if args.grid_search else '',
              args.eval_file,
              '{} tokenization'.format(args.tokenizer),
              'clustering speaker embeddings with {} metric'.format(
                  args.metric) if args.cluster else 'speaker IDs',
          ))
    time.sleep(2)

    np.random.seed(2020)
    random.seed(2020)

    results_loc = args.eval_file
    results_stub = os.path.basename(results_loc).split('.', 1)[0]
    metrics_loc = os.path.join(
        os.path.dirname(results_loc), '{}-wder.pkl'.format(results_stub))
    with open(results_loc, 'rb') as rf:
        paired_results = pickle.load(rf)

    print('{:,} test utterance pairs loaded from {}'.format(
        len(paired_results), results_loc))

    # Tokenization
    tokenization_fn = word_tokenize if args.tokenizer == 'punkt' else tweet_tokenize

    # s_is_t, c_is_t, s_t, c_t, overall_wder, asr_dist_t, n_words_t, overall_wer
    # best WDER is [4]
    outputs = []
    if args.grid_search and args.cluster:
        wder_inputs = {
            'paired_results': paired_results,
            'wer_only': args.wer_only,
            'tokenizer': tokenization_fn,
            'workers': args.workers,
            'cluster': True,
        }
        if args.metric == 'cos_sim':
            wder_inputs['metric'] = cosine_similarity
        elif args.metric == 'cos_dist':
            wder_inputs['metric'] = cosine_distance
        elif args.metric == 'idp':
            wder_inputs['metric'] = inverse_dot_product
        elif args.metric == 'ndp':
            wder_inputs['metric'] = neg_dot_product

        # DBScan space to optimize
        space = [
            Integer(2, 10, name='min_cluster_size'),
            Integer(2, 10, name='min_sample_size'),
        ]

        @use_named_args(space)
        def hyperopt_obj(**params):
            params['min_cluster_size'] = int(params['min_cluster_size'])
            params['min_sample_size'] = int(params['min_sample_size'])
            print('Hyperparams:', params)
            hyperopt_inputs = {**wder_inputs, **params}
            return corpus_wder(**hyperopt_inputs)[2]

        res_gp = gp_minimize(hyperopt_obj, space, n_calls=10, random_state=0)
        print('Hyperopt results:', res_gp)
        best_outputs = res_gp
    else:
        wder_inputs = {
            'paired_results': paired_results,
            'wer_only': args.wer_only,
            'tokenizer': tokenization_fn,
            'workers': args.workers,
            'should_cluster': args.cluster,
        }
        if args.metric == 'cos_sim':
            wder_inputs['metric'] = cosine_similarity
        elif args.metric == 'cos_dist':
            wder_inputs['metric'] = cosine_distance
        elif args.metric == 'idp':
            wder_inputs['metric'] = inverse_dot_product
        elif args.metric == 'ndp':
            wder_inputs['metric'] = neg_dot_product
        best_outputs = corpus_wder(**wder_inputs)

    with open(metrics_loc, 'wb') as wf:
        pickle.dump(best_outputs, wf)
    print('Dumped metrics to {} ({:.3f} MB)'.format(
        metrics_loc,
        os.path.getsize(metrics_loc) / 1024 / 1024))
"""
import pandas as pd
results = pd.read_pickle('ordered.pkl')

for example in results:
    ref_list, hyp_list = example
    break

print('{:,} hypothesis utterances, {:,} reference utterances'.format(
    len(hyp_list), len(ref_list)
))

ref, n_ref_speakers = convert_to_wder_format(ref_list, wer_only=True)
hyp, n_hyp_speakers = convert_to_wder_format(hyp_list, wer_only=False)
print('{:,} words (ref), {:,} words (hyp)'.format(len(ref), len(hyp)))


results = pd.read_pickle('./models/wild-speech/tal-tds-speaker-3/ordered.pkl')

for example in results:
    ref_list, hyp_list = example
    break

print('{:,} hypothesis utterances, {:,} reference utterances'.format(
    len(hyp_list), len(ref_list)
))

ref, n_ref_speakers = convert_to_wder_format(ref_list, wer_only=True)
hyp, n_hyp_speakers = convert_to_wder_format(hyp_list, wer_only=False, metric=cosine_distance)
print('{:,} words (ref), {:,} words (hyp)'.format(len(ref), len(hyp)))

wer, asr_dist, n_ref, wder, ref_spk_labels, hyp_spk_labels = calculate_wder(ref, hyp)
print('WER: {:.2f}, WDER: {:.2f}'.format(100.0 * wer, 100.0 * wder))

hyp_speaker_map = dict(zip(range(n_hyp_speakers), list(range(n_ref_speakers)) + [None] * (n_hyp_speakers - n_ref_speakers)))

"""
