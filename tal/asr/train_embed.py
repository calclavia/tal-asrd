
import os
import math
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from wildspeech.optimizers import Lamb
from .args import get_argparser
from pytorch_lightning import Trainer
from .system import System
import time
import os
from .logger import WandbLoggerWrapper
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from .args import get_argparser
from .data import ContrastiveDataset, ContrastiveCollator


class System(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.num_gpus = torch.cuda.device_count()
        # Will scale the learning rate by batch size
        self.train_bsz = args.batch_size
        self.val_bsz = args.batch_size
        self.num_workers = args.num_workers

        # MLP
        self.model = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch
        # Data in the format of [batch, n-way, features]
        # The first two examples in the n-way should match, and the rest are negative examples

        features = self.forward(x)

        # Compute cosine distances between all examples within a batch
        # Normalize
        features = features / features.norm(dim=-1, keepdim=True)
        # [batch, 1, features]
        examples = features[:, :1]
        # [batch, n-way, features]
        support = features[:, 1:]
        # Inner product
        logits = torch.matmul(examples, support.transpose(1, 2)).squeeze(1)

        logprobs = torch.log_softmax(logits, dim=-1)
        # Minimize
        loss = -logprobs[:, 0].float().mean()
        acc = (logits.argmax(dim=-1) == 0).float().mean()

        return {'loss': loss, 'log': {
            'loss': loss,
            'acc': acc
        }}

    def validation_step(self, batch, batch_idx):
        # No difference in train vs validation
        metrics = self.training_step(batch, batch_idx)
        return {'val_' + k: v for k, v in metrics['log'].items()}

    def validation_end(self, outputs):
        keys = list(outputs[0].keys())
        metrics = {k: torch.stack([x[k] for x in outputs]).mean()
                   for k in keys}

        print('val metrics', metrics)
        return {**metrics, 'log': metrics}

    def test_step(self, batch, batch_idx):
        data = self.validation_step(batch, batch_idx)
        return {k.replace('val_', 'test_'): v for k, v in data.items()}

    def test_end(self, outputs):
        keys = list(outputs[0].keys())
        metrics = {k: torch.stack([x[k] for x in outputs]).mean()
                   for k in keys}
        print('Test results', metrics)
        return {**metrics, 'log': metrics}

    def configure_optimizers(self):
        # Linear scaling rule
        # Learning rate is per batch
        effect_bsz = self.num_gpus * self.train_bsz * self.args.grad_acc
        scaled_lr = self.args.lr * \
            math.sqrt(effect_bsz) if self.args.lr else None
        print('Effective learning rate:', scaled_lr)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=scaled_lr)
        return optimizer
        # scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer, lambda e: max(
        #         1 - e / self.args.max_steps, scaled_lr / 1000)
        # )
        # return [optimizer], [scheduler]

    @pl.data_loader
    def train_dataloader(self):
        speakers, ids = torch.load(self.args.train_data[0])
        dataset = ContrastiveDataset(speakers, ids)
        dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        return DataLoader(
            dataset,
            sampler=dist_sampler,
            num_workers=self.num_workers,
            batch_size=self.train_bsz,
            collate_fn=ContrastiveCollator(),
            pin_memory=True)

    @pl.data_loader
    def val_dataloader(self):
        speakers, ids = torch.load(self.args.valid_data[0])
        dataset = ContrastiveDataset(speakers, ids)
        dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        return DataLoader(
            dataset,
            sampler=dist_sampler,
            num_workers=self.num_workers,
            batch_size=self.val_bsz,
            collate_fn=ContrastiveCollator(),
            pin_memory=True)


"""
python -m wildspeech.asr.train_embed --train-data ./models/wild-speech/tal-tds-speaker-3/train_spk_embeds.train.pt --valid-data ./models/wild-speech/tal-tds-speaker-3/train_spk_embeds.valid.pt --test-data ./models/wild-speech/tal-tds-speaker-3/train_spk_embeds.valid.pt --tokenizer taltoken-cased.model --model-type 2x --batch-size 64 --lr 5e-4 --num-workers 1
"""
if __name__ == '__main__':
    parser = get_argparser(is_train=True)
    args = parser.parse_args()
    print(args)

    if not args.val_batch_size:
        args.val_batch_size = args.batch_size

    # Create system
    model = System(args)

    gpus = torch.cuda.device_count()
    # most basic trainer, uses good defaults
    trainer = Trainer(
        checkpoint_callback=ModelCheckpoint(
            filepath=os.path.join(args.checkpoint_path,
                                  args.name, 'checkpoints'),
            save_top_k=-1
        ),
        early_stop_callback=EarlyStopping(
            'val_loss', patience=20) if args.overfit_pct == 0 else None,
        logger=WandbLoggerWrapper(
            name=args.name, project=args.project,  args=args),
        gpus=gpus,
        use_amp=True,
        default_save_path=os.path.join(
            args.checkpoint_path, args.name, 'checkpoints'),
        distributed_backend='ddp',
        accumulate_grad_batches=args.grad_acc,
        max_epochs=args.max_epochs,
        fast_dev_run=args.quick_test,
        overfit_pct=args.overfit_pct,
    )
    trainer.fit(model)
