from nltk.tokenize import word_tokenize, TweetTokenizer
import editdistance
import numpy as np
import random
from edit_distance import SequenceMatcher
from itertools import chain
import hdbscan
from collections import defaultdict
from datetime import datetime
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics.pairwise import cosine_distances
from sklearn.utils.extmath import safe_sparse_dot
import traceback

# Some segments of code taken from https://github.com/google/uis-rnn/blob/master/uisrnn/evals.py
from scipy import optimize

WT = TweetTokenizer()


def tweet_tokenize(stuff):
    return WT.tokenize(stuff)


def pairwise_idp(X, **kwargs):
    return np.reciprocal(safe_sparse_dot(X, X.T, dense_output=True) + 1e-8)


def pairwise_ndp(X, **kwargs):
    return -safe_sparse_dot(X, X.T, dense_output=True)


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


def cluster(embeddings: list, params: tuple):
    # Get type of clustering, dimensionality reduction, etc.
    cluster_type = params[0]
    cluster_params = params[1:-2]
    pca_nc = params[-2]
    metric_name = params[-1]
    metric = METRIC_MAP[metric_name]

    if isinstance(embeddings[0], tuple):
        embeddings = [np.asarray(e_tup).reshape(e_shape)
                      for e_tup, e_shape in embeddings]

    # If need to PCA
    if pca_nc is not None:
        pca = PCA(n_components=pca_nc)
        final_emb = pca.fit_transform(embeddings)
    else:
        final_emb = embeddings

    # HDBSCAN engine
    if cluster_type == 'hdbscan':
        min_cluster_size, min_samples = cluster_params
        engine = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
        )
    # DBSCAN engine
    elif cluster_type == 'dbscan':
        epsilon, min_samples = cluster_params
        engine = DBSCAN(
            eps=epsilon,
            min_samples=min_samples,
            metric=metric,
        )
    # Variational GMM engine
    elif cluster_type == 'gmm':
        weight_concentration_prior = cluster_params[0]
        engine = BayesianGaussianMixture(
            n_components=100,  # Upper bound
            max_iter=400,
            # Default 1 / # comp.
            weight_concentration_prior=weight_concentration_prior,
            n_init=5,  # Try five different initializations
        )
    # Agglomerative clustering
    elif cluster_type == 'agg':
        linkage, distance_threshold = cluster_params
        if metric_name == 'cos_dist':
            affinity = cosine_distances
        elif metric_name == 'euclidean':
            affinity = metric_name
        elif metric_name == 'idp':
            affinity = pairwise_idp
        elif metric_name == 'ndp':
            affinity = pairwise_ndp

        engine = AgglomerativeClustering(
            n_clusters=None,
            affinity=affinity,
            linkage=linkage,
            distance_threshold=distance_threshold,
        )

    # Cluster
    cluster_ids = engine.fit_predict(final_emb)
    return cluster_ids


def get_word_speakers(speaker_utterances,
                      embeddings: bool = False,
                      is_ref: bool = False,
                      role_based: bool = False,
                      role_map: dict = None,
                      tokenizer=word_tokenize):
    raw_u, raw_s, raw_r = zip(*speaker_utterances)
    # Control for Nones
    if not is_ref:
        # If we use embeddings, pick the first index. Otherwise the second.
        raw_speakers = [
            x[0 if embeddings else 1] if isinstance(x, tuple) else None
            for x in raw_s
        ]
        raw_s = []
        # Examine all raw extracted speakers
        for i, x in enumerate(raw_speakers):
            curr_speaker = x
            # Look for speaker in all following
            if curr_speaker is None:
                for future_speaker in raw_speakers[i + 1:]:
                    if future_speaker is not None:
                        curr_speaker = future_speaker
                        break
            if curr_speaker is None:
                if embeddings:
                    curr_speaker = np.array([0.0] *
                                            len(raw_s[0]) if raw_s else [0.0])
                else:
                    curr_speaker = -1
            raw_s.append(curr_speaker)
        # Map roles if necessary
        if role_based:
            raw_r = []
            for s in raw_s:
                raw_r.append(role_map.get(s, None))
            print('Mapped roles for {:,} utterances'.format(len(raw_r)))
        # print(
        #     'Adjusted speakers for hypothesis case - {:,} Nones remain'.format(
        #         len([x for x in raw_s if x is None])))
    # Convert to relative speaker format
    all_speakers = []
    words = []
    speaker_ids = []
    roles = []
    for u, speaker, role in zip(raw_u, raw_s, raw_r):
        if isinstance(speaker, (str, int)):
            spk_repr = speaker
        else:
            spk_repr = (tuple(np.asarray(speaker).flatten()), speaker.shape)
        try:
            spk_cluster = all_speakers.index(spk_repr)
        except ValueError:
            spk_cluster = len(all_speakers)
            all_speakers.append(spk_repr)
        u_words = tokenizer(u)
        words.extend(u_words)
        speaker_ids.extend([spk_cluster] * len(u_words))
        roles.extend([role] * len(u_words))
    n_speakers = len(all_speakers)
    # words -> list of words in order
    # speaker_ids -> list of relative speaker IDs in order
    # roles -> List of roles in order
    # n_speakers -> number of total unique speakers provided
    # all_speakers -> ID mapping for unique speaker embeddings
    return words, speaker_ids, roles, n_speakers, all_speakers


def get_wder_edits(ref_words: list, hyp_words: list):
    """
    Given a list of reference words and hypothesis words, find the edit operations
    (substitution and no-ops only) between the two.
    """
    # Get edit operations
    sm = SequenceMatcher(a=ref_words, b=hyp_words)
    # Find substitutions
    substitutions = [[r0, r1, h0, h1]
                     for e_type, r0, r1, h0, h1 in sm.get_opcodes()
                     if e_type == 'replace']
    # Find correct ASR words
    correct = [[r0, r1, h0, h1] for e_type, r0, r1, h0, h1 in sm.get_opcodes()
               if e_type == 'equal']
    return substitutions + correct


def get_wder(edits: list,
             ref_spk: list,
             hyp_spk: list,
             ref_roles: list,
             optimize_assignments: bool = True):
    """
    Given edits, speakers, and roles, compute WDER (and attribute WDER to roles!)
    """
    # Get speaker ID associated with each reference and hypothesis
    edit_RvH = list(
        chain.from_iterable([
            list(zip(ref_spk[r0:r1], hyp_spk[h0:h1], ref_roles[r0:r1]))
            for r0, r1, h0, h1 in edits
        ]))
    edit_R, edit_H, edit_roles = map(list, zip(*edit_RvH))
    if optimize_assignments:
        edit_R_reindex = dict(zip(sorted(set(edit_R)), range(len(edit_R))))
        edit_H_reindex = dict(zip(sorted(set(edit_H)), range(len(edit_H))))
        edit_R = [edit_R_reindex[r] for r in edit_R]
        edit_H = [edit_H_reindex[h] for h in edit_H]
        # Compute optimal mapping of speakers
        ref_spk_labels, hyp_spk_labels, match_accuracy = compute_sequence_match(
            edit_R, edit_H)
        ref_map = dict(zip(ref_spk_labels, range(len(ref_spk_labels))))
        hyp_map = dict(zip(hyp_spk_labels, range(len(hyp_spk_labels))))
        # Compute WDER and attributions
        attributions = {ro: 0 for ro in ['host', 'interviewer', 'subject']}
        wder_val = 0
        for ref_speaker, hyp_speaker, role in zip(edit_R, edit_H, edit_roles):
            if ref_map.get(ref_speaker) != hyp_map.get(hyp_speaker):
                wder_val += 1
                attributions[role] += 1
        attributions = {k: v / wder_val for k, v in attributions.items()}
        wder_val /= len(edit_R)
        # Sanity check it
        wder = 1 - match_accuracy
        try:
            assert np.abs(wder_val - wder) <= 1e-6
        except:
            print('WDER_VAL {:.6f} VS WDER {:.6f}'.format(wder_val, wder))
            raise
    else:
        attributions = {ro: 0 for ro in ['host', 'interviewer', 'subject']}
        wder_val = 0
        for ref_speaker, hyp_speaker, role in zip(edit_R, edit_H, edit_roles):
            if ref_speaker != hyp_speaker:
                wder_val += 1
                attributions[role] += 1
        attributions = {k: v / wder_val for k, v in attributions.items()}
        wder_val /= len(edit_R)
    return wder_val, attributions


def wder_segment(seg_id,
                 ref_us,
                 hyp_us,
                 cluster_params,
                 tokenizer=word_tokenize,
                 role_based: bool = False,
                 role_map: dict = None):
    """
    Computes WDER for a single segment
    """
    start = datetime.now()
    # Separate words and speakers
    ref_words, ref_spk, ref_roles, n_ref_spk, _ = get_word_speakers(
        ref_us, embeddings=False, is_ref=True, tokenizer=tokenizer)
    # hyp_spk -> speaker IDs, where hyp_spk_map is speaker ID -> embedding
    _, hyp_ids, hyp_roles, n_hyp_ids, _ = get_word_speakers(
        hyp_us,
        embeddings=False,
        role_based=role_based,
        role_map=role_map,
        is_ref=False,
        tokenizer=tokenizer)
    # hyp_spk -> speaker IDs, where hyp_spk_map is speaker ID -> embedding
    hyp_words, hyp_spk, _, n_hyp_spk, hyp_spk_map = get_word_speakers(
        hyp_us, embeddings=True, is_ref=False, tokenizer=tokenizer)
    print(
        '{} - {:,} ref w, {:,} hyp w, {:,} ref speakers, {:,} uq hyp emb, {:,} uq hyp IDs'.
        format(seg_id, len(ref_words), len(hyp_words), n_ref_spk, n_hyp_spk,
               n_hyp_ids))
    # Calculate edits
    asr_dist = editdistance.eval(ref_words, hyp_words)
    n_ref = len(ref_words)
    wer = asr_dist / n_ref
    print('{} - {} - WER: {:.2f}'.format(datetime.now() - start, seg_id,
                                         wer * 100.0))
    edits = get_wder_edits(ref_words, hyp_words)
    print('{} - {} - Got {:,} total edit operations (S+C)'.format(
        datetime.now() - start, seg_id, len(edits)))
    wder_results = dict()
    for param_set in cluster_params:
        start_pset = datetime.now()
        # Handle using IDs
        if param_set[0] == 'id':
            hyp_spk_mapped = hyp_ids
            n_clusters = len(set(hyp_ids))
        else:
            # Array with speaker_labels[speaker ID] : speaker cluster ID
            try:
                speaker_labels = cluster(hyp_spk_map, param_set)
            except Exception as e:
                print('-----\nhyp_spk_map:\n-----\n{}\n------'.format(
                    hyp_spk_map))
                print('-----\nparam_set:\n-----\n{}\n------'.format(param_set))
                traceback.print_exception(None, e, e.__traceback__)
                raise
            n_clusters = len(set(speaker_labels))
            # Coalesce from speaker IDs to speaker cluster IDs
            hyp_spk_mapped = [speaker_labels[i] for i in hyp_spk]
        if role_based:
            wder, attributions = get_wder(
                edits,
                ref_roles,
                hyp_roles,
                ref_roles,
                optimize_assignments=False)
        else:
            wder, attributions = get_wder(edits, ref_spk, hyp_spk_mapped,
                                          ref_roles)
        print(
            '{} - {} - {} {:,} speakers WDER:\t{:.2f} ({:.2f}% H, {:.2f}% I, {:.2f}% S'.
            format(datetime.now() - start_pset, seg_id, param_set, n_clusters,
                   wder * 100.0, attributions['host'] * 100.0,
                   attributions['interviewer'] * 100.0,
                   attributions['subject'] * 100.0))
        # Save
        wder_results[param_set] = (wder, n_clusters, n_ref_spk, wer,
                                   attributions)

    return wder_results


def corpus_wder_map(paired_results,
                    cluster_params,
                    tokenizer,
                    workers: int = 1,
                    role_based: bool = False,
                    role_map: dict = None):
    # Calculate stats for each hypothesis and reference
    # Empty reference = ill-formed speaker label
    # Empty hypothesis = nonterminated, thus 0 substitutions and 0 correct
    result_dicts = Parallel(n_jobs=workers)(delayed(wder_segment)(
        i,
        ref_us,
        hyp_us,
        tokenizer=tokenizer,
        cluster_params=cluster_params,
        role_based=role_based,
        role_map=role_map) for i, (ref_us, hyp_us) in tqdm(
            enumerate(paired_results), total=len(paired_results))
        if ref_us and hyp_us)
    all_wders = defaultdict(list)
    all_wers = defaultdict(list)
    all_cluster_sizes = defaultdict(list)
    all_ref_spk = defaultdict(list)
    all_host_attr = defaultdict(list)
    all_int_attr = defaultdict(list)
    all_sub_attr = defaultdict(list)
    for result_dict in result_dicts:
        for param_set, (wder, n_clusters, n_ref_spk, wer,
                        attributions) in result_dict.items():
            all_wders[param_set].append(wder)
            all_cluster_sizes[param_set].append(n_clusters)
            all_ref_spk[param_set].append(n_ref_spk)
            all_wers[param_set].append(wer)
            all_host_attr[param_set].append(attributions['host'])
            all_int_attr[param_set].append(attributions['interviewer'])
            all_sub_attr[param_set].append(attributions['subject'])
    # Get aggregate results
    agg_tups = list([(
        param_set,
        np.mean(all_wders[param_set]),
        np.mean(all_cluster_sizes[param_set]),
        np.mean(all_ref_spk[param_set]),
        np.mean(all_wers[param_set]),
        np.mean(all_host_attr[param_set]),
        np.mean(all_int_attr[param_set]),
        np.mean(all_sub_attr[param_set]),
    ) for param_set in all_wders])
    print('\n')
    overall_wer = None
    print('{}WDER\tClusters\tRef Spk\tWER\t%Host\t%Intrvw\t%Subject'.format(
        'Params'.ljust(50)))
    for pset, wder, clust_size, nref, wer, hp, ip, sp in sorted(
            agg_tups, key=lambda x: x[1]):
        print(
            '{}{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'.format(
                '{}:'.format(pset).ljust(50), wder * 100.0, clust_size, nref,
                wer * 100.0, hp * 100.0, ip * 100.0, sp * 100.0))
        overall_wer = wer
    print('\n')
    print('Overall WER: {:.2f}%'.format(overall_wer * 100.0))
    return agg_tups


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


METRIC_MAP = {
    'euclidean': 'euclidean',
    'cos_sim': cosine_similarity,
    'cos_dist': cosine_distance,
    'idp': inverse_dot_product,
    'ndp': neg_dot_product,
}
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

# Validation
python3 -u -m wildspeech.wder_search --eval-file /home/shuyang/data4/tal-new/synced_ordered.pkl --workers 10

# Test
python3 -u -m wildspeech.wder_search --eval-file /home/shuyang/data4/tal-new/synced_ordered_test.pkl --workers 10

python3 -u -m wildspeech.wder_search --eval-file /home/shuyang/data4/tal-new/synced-metric_ordered.pkl --workers 15 --algorithms id,gmm,hdbscan,dbscan,agg --metrics euclidean,cos_dist

python3 -u -m wildspeech.wder_search --eval-file /home/shuyang/data4/tal-new/synced-metric3_ordered.pkl --workers 15 --algorithms id,gmm,hdbscan,dbscan,agg --metrics euclidean,cos_dist

python3 -u -m wildspeech.wder_search --eval-file /home/shuyang/data4/tal-new/lmfusion-synced_ordered.pkl --workers 15 --algorithms id,agg

python3 -u -m wildspeech.wder_search --eval-file /home/shuyang/data4/tal-new/lmfusion-synced_ordered.pkl --workers 15 --algorithms id --role --role-map /home/shuyang/data4/tal-new/role_id_map.json

python3 -u -m wildspeech.wder_search --eval-file /home/shuyang/data4/tal-new/synced_ordered.pkl --workers 15 --algorithms id,gmm,hdbscan,dbscan,agg --metrics euclidean,cos_dist

python3 -u -m wildspeech.wder_search --eval-file /home/shuyang/data4/tal-new/synced_ordered.pkl --workers 15 --algorithms id --role --role-map /home/shuyang/data4/tal-new/role_id_map_TALTDS.json

"""
if __name__ == "__main__":
    import argparse
    import os
    import pickle
    import time
    import json
    from tqdm import tqdm
    import pandas as pd
    from itertools import product

    parser = argparse.ArgumentParser('WDER grid search')
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--role-map', type=str, default=None)
    parser.add_argument('--role', action='store_true', default=False)
    parser.add_argument(
        '--tokenizer', type=str, choices=['punkt', 'tweet'], default='punkt')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--algorithms', type=str, default='')
    parser.add_argument('--metrics', type=str, default='')
    args = parser.parse_args()

    start_all = datetime.now()

    # HANDLE ROLES
    if args.role and not args.role_map:
        raise Exception(
            'You must specify an ID->Role mapping to do role-based WDER!')
    role_map = None
    if args.role:
        with open(args.role_map, 'r+') as rf:
            role_map = {int(k): v for k, v in json.load(rf).items()}

    grid_algorithms = set(filter(None, args.algorithms.split(',') + ['id']))
    grid_metrics = set(filter(None, args.metrics.split(',') + ['cos_dist']))

    print(
        '--------\nGrid Searching {}WDER from {} with {} and {} workers with:\n{}\nMetrics:\n{}\n--------'.
        format(
            'Role-based ' if args.role else '',
            args.eval_file,
            '{} tokenization'.format(args.tokenizer),
            args.workers,
            grid_algorithms,
            grid_metrics,
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

    cluster_params = [('id', 'id')]
    # Variational GMM
    if 'gmm' in grid_algorithms:
        for wcp, pca_nc in product(
            [0.01, 0.02, 0.05, 0.1],  # Weight concentration prior
            [None, 5, 10],  # PCA components
        ):
            cluster_params.append(('gmm', wcp, pca_nc, 'euclidean'))

    for dist in grid_metrics:
        # HDBScan - min cluster size, min samples, dimensionality reduction (PCA)
        if 'hdbscan' in grid_algorithms:
            for mcs, ms, pca_nc in product(
                [3, 5],  # Min cluster size
                [1, 2, 3],  # Min samples
                [None, 5, 10],  # PCA components
            ):
                if ms > mcs:
                    continue
                cluster_params.append(('hdbscan', mcs, ms, pca_nc, dist))

        # DBScan
        if 'dbscan' in grid_algorithms:
            for epsilon, ms, pca_nc in product(
                # Epsilon (max distance bound)
                [0.1, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 1],
                [1, 2, 3],  # Min samples
                [None, 5, 10],  # PCA components
            ):
                cluster_params.append(('dbscan', epsilon, ms, pca_nc, dist))

        # Agglomerative Clustering
        if 'agg' in grid_algorithms:
            for lk, dt, pca_nc in product(
                ['complete', 'average', 'single'],
                [0.25],
                [None],  # PCA components
            ):
                if lk == 'ward' and dist != 'euclidean':
                    continue
                cluster_params.append(('agg', lk, dt, pca_nc, dist))

    print('Grid searching over {:,} parameter sets'.format(
        len(cluster_params)))
    agg_tups = corpus_wder_map(
        paired_results,
        cluster_params,
        role_based=args.role,
        role_map=role_map,
        tokenizer=tokenization_fn,
        workers=args.workers)

    df = pd.DataFrame([{
        'params': p,
        'wder': w,
        'wer': we,
        'clusters': c,
        'n_ref_spk': rs,
        'host_pct': hp,
        'interviewer_pct': ip,
        'subject_pct': sp,
    } for p, w, c, rs, we, hp, ip, sp in agg_tups])
    df = df.sort_values('wder')
    print(df)

    print(
        '--------------------------------------------\nGrid Searched {}WDER from {} with {} and {} workers with:\n{}\nMetrics:\n{}'.
        format(
            'Role-based ' if args.role else '',
            args.eval_file,
            '{} tokenization'.format(args.tokenizer),
            args.workers,
            grid_algorithms,
            grid_metrics,
        ))

    df.to_pickle(metrics_loc)
    print('Dumped metrics to {} ({:.3f} MB)'.format(
        metrics_loc,
        os.path.getsize(metrics_loc) / 1024 / 1024))

    print('{} - DONE'.format(datetime.now() - start_all))
