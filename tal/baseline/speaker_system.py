import pytorch_lightning as pl
from wildspeech.asr.data.baseline_speaker import SDUtteranceDataset, SDUtteranceCollater
from wildspeech import count_parameters, debug_log
from wildspeech.optimizers import Adafactor, Lamb
from wildspeech.schedules import triangle_schedule
from wildspeech.asr.models import SDModel, time_mask, freq_mask
import wandb
import torchaudio
import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from wildspeech.asr.logger import WandbLogger
from wildspeech.asr.data import DEFAULT_SR
from wildspeech.asr.util import *
import random
import pickle
from tqdm import tqdm


class System(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.num_gpus = torch.cuda.device_count()
        # Will scale the learning rate by batch size
        self.train_bsz = args.batch_size
        self.val_bsz = args.val_batch_size
        self.num_workers = args.num_workers
        self.train_ce_loss = LabelSmoothLoss(
            smoothing=self.args.smoothing) if self.args.smoothing > 0 else nn.CrossEntropyLoss(reduction='none')
        self.ce_loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.debug_mode = getattr(args, 'debug', False)

        self.model = SDModel(num_speakers=args.num_speakers)
        self.pad_speaker = args.num_speakers

        # Set validationa and test batches for generation. Format them as batch size 1
        gen_collater = SDUtteranceCollater(self.pad_speaker)
        self.gen_v_batch = gen_collater([self.get_dataloader('valid', return_dataset=True)[0]])

        print('Trainable Params: {:,}'.format(count_parameters(self)))
        print('Encoder Params: {:,}'.format(
            count_parameters(self.model.encoder)))

        # Holds the test outputs so far
        self.test_outputs = []

    def debug(self, x: object, msg: str = ''):
        debug_log(x, msg, debug=self.debug_mode)

    def on_epoch_start(self):
        if isinstance(self.logger, WandbLogger):
            self.logger.init(self.model)

    def forward(self, x, audio_lens):
        return self.model.forward(x, audio_lens)

    def training_step(self, batch, batch_nb):
        x, audio_lens, y, *_ = batch
        
        spk_logits, _ = self.forward(x, audio_lens)
        num_logits = spk_logits.size(-1)
        y_target = y.repeat_interleave(spk_logits.size(1), dim=1)

        if self.training:
            # Only count loss on unpadded positions
            spk_loss = self.train_ce_loss(spk_logits.view(-1, num_logits), y_target.reshape(-1))\
                .float().mean()
        else:
            # Don't use label smoothing for validation
            spk_loss = self.ce_loss_fn(spk_logits.view(-1, num_logits), y_target.reshape(-1))\
                .float().mean()

        with torch.no_grad():
            pred_ids = spk_logits.argmax(dim=-1)
            acc = (pred_ids == y).float().mean()

        loss = spk_loss
        log = {
            'spk_loss': spk_loss,
            'spk_acc': acc,
        }

        log['loss'] = loss
        return {'loss': loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        # No difference in train vs validation
        metrics = self.training_step(batch, batch_idx)
        return {'val_' + k: v for k, v in metrics['log'].items()}

    def validation_end(self, outputs):
        self.generate_single(self.gen_v_batch)

        keys = list(outputs[0].keys())
        metrics = {k: torch.stack([x[k] for x in outputs]).mean()
                   for k in keys}
        return {**metrics, 'log': metrics}

    def generate_single(self, batch):
        # Try generating
        x, audio_lens, y, *_ = batch

        if self.on_gpu:
            x = x.cuda()
            audio_lens = audio_lens.cuda()
            y = y.cuda()
        
        spk_logits, _ = self.forward(x[:1].half(), audio_lens)
        speaker_ids = spk_logits.argmax(dim=-1)

        print('\n=== CANDIDATE GENERATION ===\n')
        print('Speaker IDs: {}'.format(speaker_ids.detach().cpu().numpy().tolist()))
        print('Gold label: {}'.format(y[:1].detach().cpu().numpy().tolist()))

        if isinstance(self.logger, WandbLogger):
            self.logger.init(self.model)
            self.logger.log_generation(x[0].cpu().numpy(), '', '')

    def configure_optimizers(self):
        # Linear scaling rule
        # Learning rate is per batch
        effect_bsz = self.num_gpus * self.train_bsz * self.args.grad_acc
        scaled_lr = self.args.lr * \
            math.sqrt(effect_bsz) if self.args.lr else None
        print('Effective learning rate:', scaled_lr)
        optimizer = Lamb(self.model.parameters(), lr=scaled_lr)

        if self.args.max_steps is not None:
            total_steps = self.args.max_steps
            print('Scheduler total steps:', total_steps)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda e: max(
                    1 - e / total_steps, scaled_lr / 1000)
            )
            return [optimizer], [scheduler]
        return optimizer
    
    def get_dataloader(self, split, return_dataset=False):
        data_files = getattr(self.args, '{}_data'.format(split))
        dataset = []
        for p in data_files:
            if split == 'test':
                dataset.append(SDUtteranceDataset(
                    p,
                    num_utterances=None if self.args.unaligned else 1,
                    min_segment_duration=None if self.args.unaligned else 3,
                    max_segment_duration=None if self.args.unaligned else self.args.max_secs,
                    speaker_map_loc=os.path.join(
                        p, 'speaker_map.json'),
                ))
            else:
                dataset.append(SDUtteranceDataset(
                    p,
                    num_utterances=1,
                    max_segment_duration=self.args.max_secs,
                    speaker_map_loc=os.path.join(
                        p, 'speaker_map.json'),
                ))
        dataset = torch.utils.data.ConcatDataset(dataset)
        if return_dataset:
            return dataset
        dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        return DataLoader(
            dataset,
            sampler=dist_sampler,
            num_workers=self.num_workers,
            batch_size=self.train_bsz if split == 'train' else self.val_bsz,
            collate_fn=SDUtteranceCollater(self.pad_speaker),
            pin_memory=True)

    @pl.data_loader
    def train_dataloader(self):
        return self.get_dataloader('train', return_dataset=False)
    
    @pl.data_loader
    def val_dataloader(self):
        return self.get_dataloader('valid', return_dataset=False)

    @pl.data_loader
    def test_dataloader(self):
        return self.get_dataloader('test', return_dataset=False)
