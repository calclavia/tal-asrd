import pytorch_lightning as pl
from .data import ASRAlignedDataset, ASRAlignedCollater, RandomSegmentDataset, AudioCollator,  ASRSegmentDataset
from wildspeech import count_parameters, debug_log
from wildspeech.optimizers import Adafactor, Lamb
from wildspeech.schedules import triangle_schedule
from .models import ASRModel, time_mask, freq_mask
from wildspeech.asr.tokenizers.sentencepiece import Tokenizer
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
from .logger import WandbLogger
from .data import DEFAULT_SR
from .util import *
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

        self.model = ASRModel(args.model_type, args.num_speakers, use_speaker_head=args.spk_weight > 0)
        self.tokenizer = Tokenizer(cache_path=os.path.join(
            args.cache_path, 'tokenizer', self.args.tokenizer))

        # Set validationa and test batches for generation. Format them as batch size 1
        gen_collater = ASRAlignedCollater(self.tokenizer.pad_token_id)
        self.gen_v_batch = gen_collater([ASRAlignedDataset(
            self.args.valid_data[0],
            self.tokenizer,
        )[0]])

        print('Trainable Params: {:,}'.format(count_parameters(self)))
        print('Encoder Params: {:,}'.format(
            count_parameters(self.model.encoder)))

        # Language model
        self.lm = None
        # Holds the test outputs so far
        self.test_outputs = []

    def debug(self, x: object, msg: str = ''):
        debug_log(x, msg, debug=self.debug_mode)

    def on_epoch_start(self):
        if isinstance(self.logger, WandbLogger):
            self.logger.init(self.model)

    def generate(self,
                 audio_x: torch.Tensor,
                 generated: torch.LongTensor,
                 audio_lens: torch.LongTensor,
                 length: int,
                 beam_size: int = 1,
                 terminate_token: int = None,
                 force_half: bool = True,
                 force_output=False) -> list:
        """
        Generates text from encoder features.
        Args:
            audio_x: Input audio (wav). [batch, seq len]
            generated: The input to prime the generation. Should use [BOS] if no prime. [batch, seq_len]
            audio_lens: A long tensor containing the length of each audio input (so we can correctly pad them) [batch]
            length: The maximum output length supported. If decoding exceeds this length before generating EOS, it will return None.
            beam_size: Beam search width. Higher results in more computation but better decoding.
            terminate_token: The ID of the token to determine if the sequence terminated
            force_half: Force audio to half precision to address validation issues
            force_output: Spit out whatever the model generates even if it doesn't terminate.
        Return:
            A list of generated sequences. Each item in the list is a sequence of output decodings.
        """
        if force_half:
            audio_x = audio_x.half()

        # Note: Anything that tracks batches and beams will need to be handled accordingly w/ beam search
        encoder_out = self.model.encode(audio_x, audio_lens)
        batch_size = generated.size(0)

        # generated is currently [batch, seq_len] but will become [batch x beam size, seq_len]
        # Scores of each candidate sequence idx. Currently [batch], but will be [batch, beam]
        cur_beam_size = 1
        scores = torch.zeros(batch_size, 1, device=audio_x.device)

        # A list of finished sequences for each batch
        finished = [[] for _ in range(batch_size)]
        done = torch.BoolTensor([0 for _ in range(batch_size * beam_size)])
        # A tensor of speaker embeddings
        spk_embeds = None

        for i in tqdm(range(length)):
            model_input = generated
            model_memory = encoder_out

            logits = self.model.decode(
                model_input, model_memory, causal_mask=False)

            vocab_size = logits.size(-1)

            # Select last logit
            if self.args.spk_weight > 0:
                pred_speaker = self.model.decode_spk(
                    model_input, model_memory, causal_mask=False)
                pred_speaker = pred_speaker[:, -1]

            logits = logits[:, -1, :]
            logprobs = F.log_softmax(logits.float(), dim=-1)

            if self.lm is not None and self.args.lm_weight > 0:
                # Filter out speaker tokens, which the LM shouldn't see
                lm_input = torch.min(model_input, torch.tensor(
                    len(self.tokenizer) - 1).to(model_input))
                lm_logits = self.lm(lm_input, causal_mask=False)

                # Select last logit
                lm_logits = lm_logits[:, -1, :]

                lm_logprobs = F.log_softmax(lm_logits.float(), dim=-1)
                logprobs[:, :lm_logprobs.size(-1)] += lm_logprobs[:,
                                                                  :logprobs.size(-1)] * self.args.lm_weight

            # Total log probability of the sequence
            total_logprobs = logprobs + scores.view(-1, 1)
            if cur_beam_size == beam_size:
                # Mask out beams that are done
                total_logprobs.masked_fill_(
                    done.view(-1, 1).to(total_logprobs.device), float('-inf'))
            # Turn tensor into [batch, beams, vocab_size]
            total_logprobs = total_logprobs.view(-1,
                                                 cur_beam_size, vocab_size)
            # Flatten all beam tokens into vocab dimension so the top value can be calculated
            total_logprobs = total_logprobs.flatten(start_dim=1)
            # Within each batch, get the top tokens across the beam
            # indices = [batch, k]
            scores, indices = torch.topk(total_logprobs, k=beam_size)
            # self.debug(scores, 'Scores after topK')

            # We get each batch and the top indices of the batch
            # [batch x beam_size]
            best_tokens = (indices % vocab_size)
            # [batch x beam_size] (which beam should we pick here)
            best_beams = (indices // vocab_size)

            if cur_beam_size != beam_size:
                # First time we only have "batch" number of sequences, which we need to repeat.
                assert beam_size % cur_beam_size == 0
                generated = generated.repeat_interleave(
                    beam_size // cur_beam_size, dim=0)
                # Repeat encoder output
                encoder_out['encoder_out'] = encoder_out['encoder_out'].repeat_interleave(
                    beam_size // cur_beam_size, dim=0)
                encoder_out['encoder_padding_mask'] = encoder_out['encoder_padding_mask'].repeat_interleave(
                    beam_size // cur_beam_size, dim=0)

            def select_from_beam(x, best_beams):
                # Concat the best tokens into the beams they originated from
                # Convert to [batch, beam, seq_len]
                x = x.view(batch_size, beam_size, -1)
                x = x.gather(
                    1, best_beams.unsqueeze(-1).expand(-1, -1, x.size(-1)))
                x = x.view(batch_size * beam_size, -1)
                return x

            generated = select_from_beam(generated, best_beams)
            generated = torch.cat((generated, best_tokens.view(-1, 1)), dim=-1)

            if self.args.spk_weight > 0:
                if spk_embeds is None:
                    spk_embeds = pred_speaker.unsqueeze(1).repeat_interleave(
                        beam_size // cur_beam_size, dim=0)
                else:
                    spk_embeds = select_from_beam(spk_embeds, best_beams).view(-1, spk_embeds.size(-2), spk_embeds.size(-1))
                    spk_embeds = torch.cat((spk_embeds, pred_speaker.unsqueeze(1)), dim=1)
                assert generated.size(1) == spk_embeds.size(1) + 1, (generated.size(), spk_embeds.size())

            # EOS check
            if terminate_token is not None:
                end_indices = (
                    best_tokens.view(-1) == terminate_token
                ).nonzero().squeeze(1).cpu()
            else:
                end_indices = np.array([])

            # Store every sequence that ended.
            for index in end_indices.tolist():
                if not done[index].item():
                    # finished[batch] = (generated tensor, scores scalar) x K
                    assert len(generated[index].size()) == 1
                    finished[index // beam_size].append(
                        (
                            generated[index].cpu(),
                            spk_embeds[index].cpu() if spk_embeds is not None else None,
                            scores.view(-1)[index]
                        )
                    )
                    # Once a beam is finished, it will be ignored
                    done[index] = 1

            cur_beam_size = beam_size
            if done.sum() >= batch_size * beam_size:
                break

        # Add any additional unfinished sequences
        if terminate_token is None or force_output:
            # B x K x T
            generated = generated.view(batch_size, beam_size, -1)
            if spk_embeds is not None:
                spk_embeds = spk_embeds.view(batch_size, beam_size, spk_embeds.size(1), -1)
            # B x K
            scores = scores.view(batch_size, beam_size)

            # finished[batch] = (generated tensor, scores scalar) x K
            for batch_n in range(batch_size):
                for candidate, spk, score in zip(
                    generated[batch_n].cpu(),
                    spk_embeds[batch_n].cpu() if spk_embeds is not None else None,
                    scores[batch_n].cpu()
                ):
                    assert len(candidate.size()) == 1, candidate.size()
                    finished[batch_n].append((candidate, spk, score.item()))

        # Return the highest probability beam sequences per batch
        # Length normalize
        finished = [[(candidate, spk, score / len(candidate))
                     for candidate, spk, score in batch] for batch in finished]
        output_seq = [
            max(batch, key=lambda x: x[-1])[0] if len(batch) > 0 else None
            for batch in finished
        ]
        output_spk = [
            max(batch, key=lambda x: x[-1])[1] if len(batch) > 0 else None
            for batch in finished
        ]
        return output_seq, output_spk

    def generate_unaligned(self,
                           audio_x: torch.Tensor,
                           generated: torch.LongTensor,
                           audio_lens: torch.LongTensor,
                           chunk_size=357,
                           max_iters=1000000,
                           max_positions=None,
                           thresh_prct=0.5,
                           shift_prct=0.25,
                           stall_patience=25,
                           rep_n=5,
                           skip_prct=0.1) -> list:
        """
        Generates text from encoder features.
        Args:
            audio_x: Input audio (wav). [batch, seq len]
            generated: The input to prime the generation. Should use [BOS] if no prime. [batch, seq_len]
            audio_lens: A long tensor containing the length of each audio input (so we can correctly pad them) [batch]
            length: The maximum output length supported. If decoding exceeds this length before generating EOS, it will return None.
            beam_size: Beam search width. Higher results in more computation but better decoding.
            terminate_token: The ID of the token to determine if the sequence terminated
            force_half: Force audio to half precision to address validation issues

            thresh_prct: Percent to reach before a shift occurs
            shift_prct: Amount to shift when shift occurs
            stall_patience: Iterations of no increase in attention to deem a stall
            rep_n: N-gram repetition check
            skip_prct: Percent of chunk to skip when repetition is detected
        Return:
            A list of generated sequences. Each item in the list is a sequence of output decodings.
        """
        audio_x = audio_x.half()

        max_positions = self.model.max_positions if max_positions is None else max_positions

        # Note: Anything that tracks batches and beams will need to be handled accordingly w/ beam search
        encoder_out = self.model.encode(audio_x, audio_lens)
        encoder_lens = (~encoder_out['encoder_padding_mask']).sum(dim=-1).cpu()
        batch_size = generated.size(0)

        # A list of finished sequence indicators for each batch
        done = torch.BoolTensor([0 for _ in range(batch_size)])
        # (chunk_start, attention weights)
        alignments = []

        # Current chunk starting position
        chunk_start = torch.LongTensor(
            [0 for _ in range(batch_size)])
        history_start = torch.LongTensor(
            [0 for _ in range(batch_size)])

        # A list of progress %
        highest_progress = 0
        num_no_improve = 0
        # Time spent in this window
        window_time = 0

        def slice_tensor(x, start: torch.LongTensor, end: torch.LongTensor = None):
            if end is not None:
                jagged_tensors = [x[i, start[i]:end[i]]
                                  for i in range(x.size(0))]
            else:
                jagged_tensors = [x[i, start[i]:] for i in range(x.size(0))]

            max_len = max(map(len, jagged_tensors))
            # right-padding
            jagged_tensors = [torch.nn.functional.pad(
                x, (0, max_len - len(x)), value=0) for x in jagged_tensors]

            return torch.stack(jagged_tensors)

        def slice_encoder_out(encoder_out, start, end):
            return {
                'encoder_out': slice_tensor(encoder_out['encoder_out'], start, end),
                'encoder_padding_mask': slice_tensor(encoder_out['encoder_padding_mask'], start, end),
            }

        with tqdm(total=encoder_lens.item()) as pbar:
            for i in range(max_iters):
                model_input = generated
                model_memory = encoder_out

                # We're feeding in different input sizes via right padding, so we must retrieve the correct index.
                pad_amount = history_start - history_start.min().long()
                model_input = slice_tensor(model_input, history_start)
                select_index = [model_input.size(
                    1) - 1 - pad for pad in pad_amount.tolist()]
                assert model_input.size(
                    1) <= max_positions, 'Cannot exceed max context length'

                assert model_input.size(1) > pad_amount.max(
                ), 'Not supposed to pad more than the model input'

                model_memory = slice_encoder_out(
                    model_memory, chunk_start, chunk_start + chunk_size)

                logits = self.model.decode(
                    model_input, model_memory, causal_mask=False)

                vocab_size = logits.size(-1)

                if i > 0:
                    # Select last logit accounted for padding
                    logits = torch.stack([logits[i, s]
                                          for i, s in enumerate(select_index)], dim=0)
                else:
                    # Select last logit
                    logits = logits[:, -1, :]

                    if torch.isnan(logits).any():
                        raise Exception('Logits contain nans!')

                logprobs = F.log_softmax(logits, dim=-1)

                if self.lm is not None and self.args.lm_weight > 0:
                    # Filter out speaker tokens, which the LM shouldn't see
                    lm_input = torch.min(model_input, torch.tensor(
                        len(self.tokenizer) - 1).to(model_input))
                    lm_logits = self.lm(lm_input, causal_mask=False)

                    if i > 0:
                        # Select last logit accounted for padding
                        lm_logits = torch.stack([lm_logits[i, s]
                                                 for i, s in enumerate(select_index)], dim=0)
                    else:
                        # Select last logit
                        lm_logits = lm_logits[:, -1, :]

                    lm_logprobs = F.log_softmax(lm_logits.float(), dim=-1)
                    logprobs[:, :lm_logprobs.size(-1)] += lm_logprobs[:,
                                                                      :logprobs.size(-1)] * self.args.lm_weight

                generated = torch.cat(
                    (generated, logprobs.argmax(dim=-1).unsqueeze(1)), dim=-1)

                # Average attention across all layers and all heads
                # Attention weight per layer is [batch, tgt_size, src_size]
                # Attention weights are B x Target (text) x Source (audio)
                attn_weights = torch.stack(
                    [l.src_attn_weights for l in self.model.decoder.layers], dim=0).mean(dim=0)

                # Select the index of the attention corresponding to the last token position
                attn_weights = torch.stack(
                    [attn_weights[i, s] for i, s in enumerate(select_index)], dim=0)

                # Store the alignments
                alignments.append((chunk_start, attn_weights.cpu()))
                assert len(alignments) == generated.size(1) - 1, (len(alignments), generated.size(1))

                # Assumes a unimodal attention
                # attn_range is a vector that goes from 0 to 1 [0, ..., 1]
                attn_range = torch.arange(attn_weights.size(1)).to(
                    attn_weights) / attn_weights.size(1)
                prct_progress = (
                    attn_weights * attn_range.unsqueeze(0)).sum(dim=-1).cpu()

                # Did the progress improve?
                progress_improved = prct_progress.item() > highest_progress
                # Amount of iterations with no improvement
                if progress_improved:
                    num_no_improve = 0
                    if window_time > 5:
                        # Don't increase for first 5 steps of a window
                        highest_progress = prct_progress.item()
                else:
                    num_no_improve += 1

                is_stalling = torch.tensor(num_no_improve >= stall_patience)

                """ Edge case to reset history. """

                # Measures how much repetition
                rep_mask = ngram_repeat_mask(model_input, rep_n)
                # Count the amount of consecutive 1s from the back
                rep_count = rep_mask.sum(dim=-1).cpu()
                is_repeating = rep_count > rep_n * 2

                # Are we at the last chunk?
                is_last_chunk = encoder_lens - chunk_start <= chunk_size

                # Reset history when model generates EOS or if it repeats too often
                reset_window = is_stalling | is_repeating
                window_changed = False

                if (~is_last_chunk).any():
                    if reset_window.any():
                        # Skips forward in time.
                        chunk_start += int(chunk_size * skip_prct)
                                        
                        if is_repeating.any():
                            # Trim away the repetitions
                            rollback_amount = 2 * rep_n
                            generated = generated[:, :-(rollback_amount-1)]
                            alignments = alignments[:-(rollback_amount-1)]

                        # Put an EOS in last token position
                        generated[:, -1] = self.tokenizer.eos_token_id
                        # Reset history to EOS
                        history_start[:] = generated.size(1) - 1

                        # Reset previous progress
                        highest_progress = 0
                        window_time = 0
                        window_changed = True
                    else:                        
                        # Don't advance again if we just reset
                        """ Regular case to advance window forward. """

                        # Whether or not we should advance the window. [batch_size]
                        # 1 for a sample in the batch that should advance forward. 0 otherwise.
                        if prct_progress > thresh_prct:
                            history_size = generated.size(1) - history_start

                            # Advance attention window forward
                            chunk_start += int(chunk_size * shift_prct)
                            # We have generated text corresponding to "thresh_prct" amount of audio and we're shifting by "shift_prct" of audio
                            # So we should remove the first "shift_prct" of text
                            del_prct = shift_prct / thresh_prct
                            history_start += (del_prct * (history_size - 1)).floor().long()
                            # Reset previous progress
                            highest_progress = 0
                            window_time = 0
                            window_changed = True

                """ Bounds certain values """
                # Prevent chunk start from exceeding maximum length
                chunk_start = torch.min(chunk_start, encoder_lens - chunk_size)
                # Truncate text to max positions
                history_start = torch.max(history_start, torch.tensor(
                    max(generated.size(1) - max_positions, 0)))

                assert history_start.max() < generated.size(1), ('Invalid history start index',
                                                                 history_start, generated.size(1))
                assert generated.size(1) - history_start.min() <= max_positions, ('Exceed max positions',
                                                                                  history_start, generated.size(1))
                window_time += 1

                # print('Progress %', prct_progress, highest_progress, num_no_improve, is_stalling)

                # if window_changed:
                #     print('\nWindow changed')
                #     print('Output so far', self.tokenizer.decode_speakers(
                #         generated[0].cpu().tolist()))
                #     print('Output IDS last', generated[0].cpu().tolist()[:10])
                #     print('History set to', self.tokenizer.decode_speakers(
                #         generated[0, history_start[0]:].cpu().tolist()))
                #     print('Was repeating?', is_repeating)
                #     print('Progress %', prct_progress)
                #     # print('Del %', del_prct)
                #     print('chunk_start', chunk_start, 'of', encoder_lens)
                #     print('New History Size', generated.size(1) - history_start)
                #     print('is_last_chunk?', is_last_chunk)
                #     print('is_stalling?', is_stalling)
                #     print('rep_mask', rep_mask)

                # If we're in the last chunk and we detect a window reset, we're done.
                end_indices = (
                    reset_window & is_last_chunk).nonzero().squeeze(1)

                # Store every sequence that ended.
                for index in end_indices.tolist():
                    if not done[index].item():
                        done[index] = 1

                if done.sum() >= batch_size:
                    break

                pbar.n = chunk_start.item()
                pbar.refresh()

        return generated, alignments

    def forward(self, x, y_prev, audio_lens):
        return self.model.forward(x, y_prev, audio_lens)

    def training_step(self, batch, batch_nb):
        x, audio_lens, y, y_mask, spk_ids, *_ = batch

        # Remove unknown speakers
        if self.args.num_speakers > 0 and self.args.spk_weight == 0:
            y_known = torch.min(y, torch.tensor(
                len(self.tokenizer) + self.args.num_speakers - 1).to(y))
            y_prev = y_known[:, :-1]
            y_target = y_known[:, 1:]
        else:
            y_prev = y[:, :-1]
            y_target = y[:, 1:]

        if self.training:
            # Following TDS paper, randomly replace 1% of input with random tokens
            rand_mask = torch.rand_like(
                y_prev, dtype=torch.half) < 0.01
            y_prev = torch.where(rand_mask, torch.randint_like(
                y_prev, high=len(self.tokenizer)), y_prev)

        (y_hat, spk_pred), _ = self.forward(x, y_prev, audio_lens)
        num_logits = y_hat.size(-1)

        # Don't use label smoothing for validation
        ce_fn = self.train_ce_loss if self.training else self.ce_loss_fn
        
        lm_loss = ce_fn(y_hat.view(-1, num_logits), y_target.reshape(-1))\
            .masked_select(y_mask[:, 1:].reshape(-1)).float().mean()

        spk_loss = torch.tensor(0.).to(lm_loss)
        if self.args.spk_weight > 0:
            # Directly predict the speaker ID
            spk_loss = ce_fn(spk_pred.view(-1, spk_pred.size(-1)), spk_ids[:, 1:].reshape(-1))\
                .masked_select(y_mask[:, 1:].reshape(-1)).float().mean()

        loss = lm_loss + self.args.spk_weight * spk_loss
        log = {
            'lm_loss': lm_loss,
            'spk_loss': spk_loss,
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
        x, audio_lens, y, y_mask, *_ = batch

        if self.on_gpu:
            x = x.to('cuda')
            audio_lens = audio_lens.to('cuda')
            y = y.to('cuda')
            y_mask = y_mask.to('cuda')

        generated, _ = self.generate(
            x[:1].half(), y[:1, :1], audio_lens, int(y.size(1) * 1.5),
            terminate_token=self.tokenizer.eos_token_id,
        )
        ref_text = None
        hyp_text = None
        # Generate
        for batch_idx, (hyp, ref) in enumerate(zip(generated, y.cpu())):
            # Remove the masks
            target_len = sum(y_mask[batch_idx].cpu().tolist())

            if hyp is not None and len(hyp) > 1:
                hyp_text = self.tokenizer.decode(
                    [x for x in hyp.tolist() if x < len(self.tokenizer)])
            ref_text = self.tokenizer.decode(
                [x for x in ref[:target_len - 1].tolist() if x < len(self.tokenizer)])

        print('\n=== CANDIDATE GENERATION ===\n')
        print('Tokens:', hyp, ref)
        if ref_text is not None:
            print('GOLD:\n{}\n'.format(ref_text))
        if hyp_text is not None:
            print('GENERATED:\n{}\n'.format(hyp_text))
        print('Done.')

        if isinstance(self.logger, WandbLogger):
            self.logger.init(self.model)
            self.logger.log_generation(x[0].cpu().numpy(), ref_text, hyp_text)

    def test_step(self, batch, batch_nb):
        # MLE Loss score
        x, audio_lens, y, y_mask, _, sample_id = batch
        batch_size = x.size(0)

        if self.args.unaligned:
            loss = torch.tensor(0.0)
        else:
            # Remove unknown speakers
            if self.args.num_speakers > 0 and self.args.spk_weight == 0:
                y_known = torch.min(y, torch.tensor(
                    len(self.tokenizer) + self.args.num_speakers - 1).to(y))
                y_prev = y_known[:, :-1]
                y_target = y_known[:, 1:]
            else:
                y_prev = y[:, :-1]
                y_target = y[:, 1:]

            (y_hat, _), _ = self.forward(x, y_prev, audio_lens)
            num_logits = y_hat.size(-1)

            # Only count loss on unpadded positions
            loss = self.ce_loss_fn(y_hat.view(-1, num_logits), y_target.reshape(-1))\
                .masked_select(y_mask[:, 1:].reshape(-1)).float().mean()

        # A batch of sequence of speaker embeddings.
        # Each sample consists of a list of speaker embeddings corresponding to the number of utterances
        batch_speaker_embeds = None
        try:
            # First token is always EOS
            if self.args.unaligned:
                generated, alignments = self.generate_unaligned(
                    x,
                    y[:, :1],
                    audio_lens,
                    chunk_size=357 if self.args.unaligned else None)
            else:
                alignments = None
                generated, spk_embeds = self.generate(
                    x,
                    y[:, :1],
                    audio_lens,
                    length=int(y.size(1) * 1.1),
                    terminate_token=self.tokenizer.eos_token_id,
                    beam_size=self.args.beam_size)

                # We generate single utterances, so we simply average all embeddings
                # batch_speaker_embeds = [x.mean(dim=0, keepdim=True) if x is not None else None for x in spk_embeds]
                batch_speaker_embeds = spk_embeds
        except Exception as e:
            print('Failed to generate for sample IDs', sid)
            raise e

        hyp_dec = []
        ref_dec = []
        failed_seq = 0

        for batch_idx, (hyp, ref) in enumerate(zip(generated, y.cpu())):
            # Remove the masks
            target_len = sum(y_mask[batch_idx].cpu().tolist())

            if hyp is not None and len(hyp) > 1:
                # Decodes h into utterances
                hyp = hyp[:-1].tolist()
                
                # Convert to list of (utterance, speakerId)
                utts, split_indices = self.tokenizer.decode_speakers(hyp)

                # Add metadata for each utterance
                utts = [{'utterance': utt_str, 'speakerId': sid} for utt_str, sid in utts]

                if alignments is not None:
                    last_split_i = 0
                    for utt, split_i in zip(utts, split_indices):
                        assert split_i - last_split_i > 0
                        relevant_aligns = alignments[last_split_i:split_i + 1]
                        
                        assert len(relevant_aligns) > 0
                        chunk_starts, weights = zip(*relevant_aligns)
                        utt['attention'] = torch.cat(weights, dim=0)
                        utt['chunkStart'] = torch.cat(chunk_starts, dim=0)
                        utt['utteranceTokens'] = hyp[last_split_i:split_i + 1]
                        last_split_i = split_i

                hyp_dec.append(utts)
            else:
                hyp_dec.append([])
                failed_seq += 1

            # A hack to retrieve the original utterance
            _, utts = self.test_index[sample_id[batch_idx]]
            ref_dec.append(utts)

        if failed_seq > 0:
            print('Warning! Sequences did not terminate:', failed_seq)

        self.test_outputs += list(zip(ref_dec, hyp_dec))
        with open('out/test_result.pkl', 'wb') as f:
            pickle.dump(self.test_outputs, f)

        # TODO: Should store this in checkpoint directory.
        # Write the outputs
        with open('out/hyp.txt', 'a+') as ftxt:
            for utts in hyp_dec:
                # ASR-only output
                ftxt.write(' '.join([u['utterance'] for u in utts]) + '\n')

        with open('out/ref.txt', 'a+') as ftxt:
            for utts in ref_dec:
                # ASR-only output
                ftxt.write(' '.join([u['utterance'] for u in utts]) + '\n')

        # Write audio files
        # bsz = x.size(0)
        # for i in range(bsz):
        #     torchaudio.save('out/dump_audio_{}.wav'.format(i + batch_nb * bsz), x[i, :audio_lens[i]].cpu(), DEFAULT_SR)

        return {'loss': loss, 'log': {'test_loss': loss}}

    def test_end(self, outputs):
        """
        Called at the end of test to aggregate outputs
        :param outputs: list of individual outputs of each test step
        :return:
        """
        test_loss_mean = 0
        for output in outputs:
            test_loss_mean += output['loss']

        test_loss_mean /= len(outputs)
        tqdm_dict = {'test_loss': test_loss_mean.item()}
        print('Loss', tqdm_dict['test_loss'])

        # Only log test_loss
        return {
            'progress_bar': tqdm_dict,
            'log': {'test_loss': test_loss_mean.item()}
        }

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

    @pl.data_loader
    def train_dataloader(self):
        dataset = []
        for p in self.args.train_data:
            dataset.append(ASRAlignedDataset(
                p,
                self.tokenizer,
                num_utterances=1,
                max_segment_duration=self.args.max_secs,
                speaker_map_loc=os.path.join(
                    p, 'speaker_map.json'),
                tokenizer_speakers=self.args.num_speakers > 0 and self.args.spk_weight == 0,
                return_spk_ids=True
            ))
            if self.args.shiftaug or self.args.alignaug:
                dataset.append(ASRSegmentDataset(
                    p,
                    self.tokenizer,
                    segment_size=self.args.max_secs,
                    speaker_map_loc=os.path.join(
                        p, 'speaker_map.json'),
                    tokenizer_speakers=self.args.num_speakers > 0 and self.args.spk_weight == 0,
                    aligned_truncation=self.args.alignaug,
                    return_spk_ids=True
                ))
        dataset = torch.utils.data.ConcatDataset(dataset)

        dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        return DataLoader(
            dataset,
            sampler=dist_sampler,
            num_workers=self.num_workers,
            batch_size=self.train_bsz,
            collate_fn=ASRAlignedCollater(
                self.tokenizer.pad_token_id),
            pin_memory=True)

    @pl.data_loader
    def val_dataloader(self):
        dataset = []
        for p in self.args.valid_data:
            dataset.append(ASRAlignedDataset(
                p,
                self.tokenizer,
                num_utterances=1,
                max_segment_duration=self.args.max_secs,
                speaker_map_loc=os.path.join(
                    p, 'speaker_map.json'),
                tokenizer_speakers=self.args.num_speakers > 0 and self.args.spk_weight == 0,
                return_spk_ids=True
            ))
        dataset = torch.utils.data.ConcatDataset(dataset)
        dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        return DataLoader(
            dataset,
            sampler=dist_sampler,
            num_workers=self.num_workers,
            batch_size=self.val_bsz,
            collate_fn=ASRAlignedCollater(
                self.tokenizer.pad_token_id),
            pin_memory=True)

    @pl.data_loader
    def test_dataloader(self):
        dataset = []
        for p in self.args.test_data:
            dataset.append(ASRAlignedDataset(
                p,
                self.tokenizer,
                num_utterances=None if self.args.unaligned else 1,
                min_segment_duration=None if self.args.unaligned else 3,
                max_segment_duration=None if self.args.unaligned else self.args.max_secs,
                speaker_map_loc=os.path.join(
                    p, 'speaker_map.json'),
                tokenizer_speakers=self.args.num_speakers > 0 and self.args.spk_weight == 0,
                return_spk_ids=True
            ))
            self.test_index = dataset[-1].index
        dataset = torch.utils.data.ConcatDataset(dataset)
        dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        return DataLoader(
            dataset,
            sampler=dist_sampler,
            num_workers=self.num_workers,
            batch_size=self.val_bsz,
            collate_fn=ASRAlignedCollater(
                self.tokenizer.pad_token_id),
            pin_memory=True)
