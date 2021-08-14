from wildspeech.asr.args import get_argparser
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from wildspeech.asr.logger import WandbLoggerWrapper
import os
import time
from wildspeech.baseline.speaker_system import System
from pytorch_lightning import Trainer
import torch
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


if __name__ == '__main__':
    parser = get_argparser(is_train=True)
    args = parser.parse_args()
    print(args)

    if not args.val_batch_size:
        args.val_batch_size = args.batch_size

    # Handle CUDA things
    vis_devices_os = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if not vis_devices_os:
        gpus = torch.cuda.device_count()
        print('\n===========')
        print('No $CUDA_VISIBLE_DEVICES set, defaulting to {:,}'.format(gpus))
        print('===========\n')
    else:
        gpus = list(map(int, vis_devices_os.split(',')))
        print('Visible devices as specified in $CUDA_VISIBLE_DEVICES: {}'.format(gpus))

    model = System(args)

    if args.load:
        print('Loading existing model weights...', args.load)
        checkpoint = torch.load(
            args.load, map_location=lambda storage, loc: storage)
        state_dict = checkpoint['state_dict']

        model.load_state_dict(state_dict, strict=not args.no_strict)
        # give model a chance to load something
        model.on_load_checkpoint(checkpoint)

    if args.load_encoder:
        print('Loading encoder weights...', args.load_encoder)
        checkpoint = torch.load(
            args.load_encoder, map_location=lambda storage, loc: storage)
        state_dict = checkpoint['state_dict']

        # Filter out only encoder parts
        print('!!! WARNING: Only loading encoder parts of the model!')
        state_dict = {k: v for k, v in state_dict.items() if '.encoder' in k}
        print('Loading encoder parameters from', list(state_dict.keys()))
        model.load_state_dict(state_dict, strict=not args.no_strict)
        # give model a chance to load something
        model.on_load_checkpoint(checkpoint)

    # most basic trainer, uses good defaults
    trainer = Trainer(
        checkpoint_callback=ModelCheckpoint(
            filepath=os.path.join(args.checkpoint_path,
                                  args.name, 'checkpoints'),
            save_top_k=-1
        ),
        early_stop_callback=EarlyStopping(
            'val_loss', patience=10) if args.overfit_pct == 0 else None,
        logger=WandbLoggerWrapper(
            name=args.name, project=args.project, args=args),
        gpus=gpus,
        use_amp=True,
        default_save_path=os.path.join(
            args.checkpoint_path, args.name, 'checkpoints'),
        distributed_backend='ddp',
        accumulate_grad_batches=args.grad_acc,
        max_epochs=args.max_epochs,
        fast_dev_run=args.quick_test,
        overfit_pct=args.overfit_pct,
        val_check_interval=args.val_check_interval,
    )
    trainer.fit(model)
