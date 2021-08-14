import os
import argparse
import torch
import editdistance
from wildspeech.asr.transcribe import splice_strings
import string

"""
python -u -m wildspeech.asr.util.eval_transcripts ./data/tal-lex/inference/ref-tal-test.txt ./data/tal-lex/inference/hypo-tal-test.txt
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compares the transcripts from a reference transcript and a hypothesis.')
    parser.add_argument('ref', type=str)
    parser.add_argument('hyp', type=str)

    args = parser.parse_args()
    print(args)

    with open(args.ref) as f:
        ref_text = f.read()
    with open(args.hyp) as f:
        hyp_text = f.read()

    punc_table = str.maketrans('', '', string.punctuation)

    hyp = splice_strings(hyp_text.split('<EOT><EOT>'))
    hyp = hyp.replace('<|endoftext|>', ' <|endoftext|>').replace(
        '<EOT>', '').strip()

    print('Computing edit distance')
    ref = ref_text.lower().strip().translate(punc_table).split(' ')
    hyp = hyp.lower().strip().translate(punc_table).split(' ')

    word_edit_distance = editdistance.eval(ref, hyp)
    print('ED', word_edit_distance)
    print('WER', word_edit_distance / len(ref))

    # TODO:
    """
    # Create turns
    ref_turns = ref_text.split('<|endoftext|>')
    selected_hyp_turns = []
    hyp_turns = hyp.split('<|endoftext|>')

    # We want to try to best align the hypothesis with a reference.
    # Sometimes a turn can be completely missed or new turns could be added.
    reach = 5
    cur_turn = 0

    print('Trying to align hyps to refs...')
    for ref in ref_turns:
        ref = ref.lower().strip().translate(punc_table).split(' ')
        best_id = None
        best_score = float('inf')
        for i, candidate in enumerate(hyp_turns[cur_turn:cur_turn+reach]):
            candidate = candidate.lower().strip().translate(punc_table).split(' ')
            score = editdistance.eval(ref, candidate)

            if score < best_score:
                best_id = i
                best_score = score
        
        selected_hyp_turns.append(hyp_turns[best_id])
        cur_turn = i

    with open(args.ref + '.format', 'w') as f:
        for x in ref_turns:
            f.write(x + '\n')
    with open(args.hyp + '.format', 'w') as f:
        for x in selected_hyp_turns:
            f.write(x + '\n')
    """
