import argparse
import os
import json


def prune_bad_utterances(index: list, score_data: list, threshold: float = 9.1):
    """
    Prunes away "bad utterances" by thresholding all utterances
    above a certain loss threshold, then removing them from index.

    Args:
        score_data: A list of [filename, utterance index, score] which contains
                    each valid index and its score.
    """
    # Set of index data to prune
    prune_set = {tuple(s[:2]) for s in score_data if s[-1] > threshold}
    return [x for x in index if all((x['file'], i) not in prune_set for i in range(x['block_start'], x['block_end']))]


if __name__ == '__main__':
    """
    Example: python -m bernard.asr.util.prune_bad_utterances ./data/tal/test/index_block.json ./data/tal/libri_test_loss.json
    """
    parser = argparse.ArgumentParser(
        description='Prunes all the bad utterances from index file')
    parser.add_argument('in_file', type=str)
    parser.add_argument('score_data', type=str)

    args = parser.parse_args()
    print(args)

    with open(args.in_file) as f:
        index = json.load(f)

    with open(args.score_data) as f:
        score_data = json.load(f)

    print('Before', len(index))
    index = prune_bad_utterances(index, score_data)
    print('After', len(index))

    with open(args.in_file, 'w') as f:
        json.dump(index, f)
