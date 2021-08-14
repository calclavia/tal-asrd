import os
import re
import string
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Strips the output file for WER calculation.')
    parser.add_argument('in_file', type=str)
    parser.add_argument('out_file', type=str)
    parser.add_argument('--strip-punct', action='store_true', default=False, help='Remove punctuation.')
    parser.add_argument('--strip-speakers', action='store_true', default=False, help='Remove speaker IDs.')
    parser.add_argument('--strip-eos', action='store_true', default=False, help='Remove EOS tokens.')
    
    args = parser.parse_args()
    print(args)
    
    with open(args.in_file) as f:
        lines = [l for l in f]

    if args.strip_punct:    
        table = str.maketrans('', '', string.punctuation)
        lines = [l.strip().translate(table) for l in lines]

    if args.strip_speakers:
        lines = [re.sub(r'\<S[0-9]+\>', '', l) for l in lines]

    if args.strip_eos:
        lines = [l.replace('<|endoftext|>', '') for l in lines]

    with open(args.out_file, 'w') as f:
        for l in lines:
            f.write(l + '\n')