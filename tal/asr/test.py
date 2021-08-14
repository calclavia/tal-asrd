import torch, os
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

from pytorch_lightning import Trainer
from .system import System
from .args import get_argparser
from wildspeech.lm.model import DecoderLMModel

if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    print(args)

    assert args.load is not None, 'Specify path to weights.'

    if not args.val_batch_size:
        args.val_batch_size = args.batch_size
        
    # TODO: Update to use Lightning's official loading
    # load on CPU only to avoid OOM issues
    # then its up to user to put back on GPUs
    checkpoint = torch.load(args.load, map_location=lambda storage, loc: storage)
    state_dict = checkpoint['state_dict']
    # TODO: A hack to remove the tokens
    # state_dict['model.token_embedding.weight'] = state_dict['model.token_embedding.weight'][:10000]
    # state_dict['model.lm_head.weight'] = state_dict['model.lm_head.weight'][:10000]
    
    # load the state_dict on the model automatically
    model = System(args)
    model.load_state_dict(state_dict, strict=not args.no_strict)

    # give model a chance to load something
    model.on_load_checkpoint(checkpoint)

    if args.lm_path:
        print('Loading language model weights...', args.lm_path)
        # TODO: Custom loading because lightning seems to be broken.
        checkpoint = torch.load(args.lm_path, map_location=lambda storage, loc: storage)
        state_dict = checkpoint['state_dict']
        state_dict = {k.replace('model.', ''):v for k, v in state_dict.items()}

        print('Loading parameters from', list(state_dict.keys()))
        lm = DecoderLMModel(
                 vocab_size=10000,
                 hidden_size=512,
                 attn_heads=4,
                 dropout=0.1,
                 decoder_layers=6,
                 activation='relu',
                 max_len=512
        )
        # For FT model
        lm.load_state_dict(state_dict, strict=False)
        # lm.load_state_dict(state_dict)
        lm.eval()
        model.lm = lm
        
    # most basic trainer, uses good defaults
    trainer = Trainer(
        gpus=torch.cuda.device_count(),
        use_amp=True,
        default_save_path='out/checkpoints',
        distributed_backend='ddp',
    )

    model.eval()

    print('Removing existing hyp and ref files...')
    if os.path.exists('out/hyp.txt'):
        os.unlink('out/hyp.txt')
    if os.path.exists('out/ref.txt'):
        os.unlink('out/ref.txt')
    if os.path.exists('out/hyp.jsonl'):
        os.unlink('out/hyp.jsonl')
    if os.path.exists('out/ref.jsonl'):
        os.unlink('out/ref.jsonl')
    trainer.test(model)
