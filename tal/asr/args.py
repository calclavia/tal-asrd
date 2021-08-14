import argparse


def get_argparser(is_train=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default=None,
                        help='Model weights to load')
    parser.add_argument('--load-encoder', type=str, default=None,
                        help='Model weights to load to only the encoder')
    parser.add_argument('--load-decoder', type=str, default=None,
                        help='Model weights to load to only the decoder')
    parser.add_argument('--train-data', type=str,
                        action='append', required=True)
    parser.add_argument('--valid-data', type=str,
                        action='append', required=True)
    parser.add_argument('--test-data', type=str, action='append')

    parser.add_argument('--cache-path', type=str, default='./cache')
    parser.add_argument('--tokenizer', type=str, required=True)
    parser.add_argument('--model-type', type=str, required=True)
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--val-batch-size', type=int, default=None)
    parser.add_argument('--beam-size', type=int, default=5)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--grad-acc', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Base learning rate per batch')
    parser.add_argument('--max-secs', type=float, default=20,
                        help='Max amount of seconds in a batch')
    parser.add_argument('--no-strict', action='store_true',
                        default=False, help='Disable strict loading')
    parser.add_argument('--num-speakers', type=int, default=0,
                        help='Amount of speakers to predict as part of the vocabulary. Default = no speaker prediction')
    parser.add_argument('--quick-test', action='store_true', default=False)
    parser.add_argument('--unaligned', action='store_true', default=False,
                        help='Perform unaligned (full episode) evaluation.')
    parser.add_argument('--shiftaug', action='store_true',
                        default=False, help='Performs random shift augmentation.')
    parser.add_argument('--alignaug', action='store_true', default=False,
                        help='Performs random aligned shift augmentation.')
    parser.add_argument('--spk-weight', type=float, default=0,
                        help='Multi-task speaker learning weight. This turns of speaker tokenization')
    parser.add_argument('--val-check-interval', type=int, default=1.0,
                        help='Number of iterations per validation step.')
    parser.add_argument('--lm-weight', type=float, default=0,
                        help='Weight of the language model while decoding')
    parser.add_argument('--smoothing', type=float,
                        default=0, help='Label smoothing amount')
    parser.add_argument('--lm-path', type=str, default=None,
                        help='Path to language model checkpoint')

    if is_train:
        parser.add_argument('--name', type=str, required=True)
        parser.add_argument('--project', type=str, default='asr')
        parser.add_argument('--checkpoint-path', type=str,
                            default='./models/wild-speech', help='Path to save model checkpoints')
        parser.add_argument('--overfit-pct', type=float, default=0,
                            help='Percetange of training set to use. 0 to disable.')

    return parser


def get_lm_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', type=str, required=True)
    parser.add_argument('--valid-data', type=str, required=True)
    parser.add_argument('--epoch-len', type=int, default=5000)

    parser.add_argument('--cache-path', type=str, default='./cache')
    parser.add_argument('--tokenizer', type=str, required=True)

    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--val-batch-size', type=int, default=None)
    parser.add_argument('--hidden-size', type=int, default=768)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--attn-heads', type=int, default=4)
    parser.add_argument('--max-len', type=int, default=512)

    parser.add_argument('--beam-size', type=int, default=5)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--grad-acc', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Base learning rate per batch')

    parser.add_argument('--no-strict', action='store_true',
                        default=False, help='Disable strict loading')

    parser.add_argument('--quick-test', action='store_true', default=False)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--project', type=str, default='asr')
    parser.add_argument('--checkpoint-path', type=str,
                        default='./models/wild-speech', help='Path to save model checkpoints')
    parser.add_argument('--load', type=str, default=None,
                        help='Model weights to load')
    parser.add_argument('--overfit-pct', type=float, default=0,
                        help='Percetange of training set to use. 0 to disable.')
    parser.add_argument('--load-gpt-weights',
                        action='store_true', default=False)
    parser.add_argument('--train-tal', action='store_true', default=False)
    parser.add_argument('--deterministic', action='store_true', default=False)
    parser.add_argument('--skip-gen', action='store_true', default=False)

    return parser
