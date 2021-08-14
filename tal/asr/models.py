import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import random
import json
from torchaudio import transforms
from .data import DEFAULT_SR
from rezero.transformer import RZTXDecoderLayer
from fairseq.models.wav2vec import Wav2VecModel
from wildspeech.modules import PositionalEncoding, weight_init


class LogMelSpec(nn.Module):
    """
    Performs log melspec transformation on raw audio
    Adapted from https://github.com/zcaceres/spec_augment/blob/master/SpecAugment.ipynb
    Parameters based on https://arxiv.org/pdf/1904.11660.pdf
    """

    def __init__(self, sr=DEFAULT_SR, n_mels=80, eps=1e-6):
        super().__init__()
        self.mel_transform = transforms.MelSpectrogram(
            sample_rate=sr,
            n_mels=n_mels,
            n_fft=int(25/1000 * sr),
            # 400 win length = 25ms window
            win_length=int(25/1000 * sr),
            # 160 steps = 10ms hop
            hop_length=int(10/1000 * sr)
        )
        self.eps = eps

    @torch.jit.ignore
    def forward(self, audio: torch.Tensor):
        """
        Args:
            Audio: Tensor of [batch, audio_len]
        """
        with torch.no_grad():
            # TODO: A hack to fix something wrong with this parameter not going on CUDA automatically
            # self.mel_transform.mel_scale.fb = self.mel_transform.mel_scale.fb.to(audio.device)
            # Melsepc accepts [channel, time]
            mel = self.mel_transform(audio)
            # Outputs [channel, n_mels, time]
            # swap dimension, mostly to look sane to a human.
            mel = mel.permute(0, 2, 1)
            # Becomes [channel, time, n_mels]
            mel = torch.log(mel + self.eps)
            # Apply zero mean Transform
            mel -= mel.mean()
        return mel


class ASRModel(nn.Module):
    def __init__(self,
                 model_type='2x',
                 num_speakers=0,
                 n_mels=80,
                 vocab_size=10000,
                 n_head=4,
                 max_positions=512,
                 dropout=0.2,
                 embed_size=64,
                 spk_embed=128,
                 use_speaker_head=False,
                 ):
        super().__init__()
        self.embed_size = embed_size
        self.num_speakers = num_speakers
        self.max_positions = max_positions
        self.use_speaker_head = use_speaker_head
        self.model_type = model_type

        tds_sizes = [n_mels, 10 * n_mels, 14 * n_mels, 18 * n_mels]
        tds_depths = [2, 3, 6]

        if model_type == '1x':
            d_decoder_hidden = 256
            n_decoder_layers = 4
        elif model_type == '2x':
            d_decoder_hidden = 512
            n_decoder_layers = 4
        else:
            raise Exception('Invalid model type')

        self.pos_dec_encoder = PositionalEncoding(
            d_decoder_hidden, max_len=max_positions, dropout=dropout)

        # 1 second of audio (15999 frames) => 100 frames. Each frame is 0.01s.
        # Each frame has a window of 25ms with 10ms stride.
        self.logmelspec = LogMelSpec(n_mels=n_mels)

        # 141 frames => 1 feature. 1 feature ~1.41 seconds receptive field
        self.encoder = TDS(n_mels, tds_sizes, tds_depths, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        # Projects encoder size to decoder size
        self.decoder_proj = nn.Linear(tds_sizes[-1], d_decoder_hidden)

        # Decoder related
        num_tokens = vocab_size if use_speaker_head else vocab_size + num_speakers
        if self.embed_size:
            # Factorized embedding
            self.embedding = nn.Embedding(num_tokens, embed_size)
            self.embedding_proj = nn.Linear(
                embed_size, d_decoder_hidden, bias=False)
            # Decoder language modeling head
            self.lm_head = nn.Linear(embed_size, num_tokens, bias=False)
        else:
            self.embedding = nn.Embedding(num_tokens, d_decoder_hidden)
            # Decoder language modeling head
            self.lm_head = nn.Linear(d_decoder_hidden, num_tokens, bias=False)

        # Tie input and output token embeddings
        self.lm_head.weight = self.embedding.weight

        self.decoder = nn.TransformerDecoder(
            ModRZTXDecoderLayer(
                d_model=d_decoder_hidden,
                dim_feedforward=d_decoder_hidden * 4,
                nhead=n_head,
                dropout=dropout,
                activation='relu'
            ),
            n_decoder_layers
        )

        if self.use_speaker_head:
            self.spk_enc_proj = nn.Linear(tds_sizes[-1], d_decoder_hidden)

            self.spk_decoder = nn.TransformerDecoder(
                ModRZTXDecoderLayer(
                    d_model=d_decoder_hidden,
                    dim_feedforward=d_decoder_hidden * 4,
                    nhead=n_head,
                    dropout=dropout,
                    activation='relu'
                ),
                n_decoder_layers // 2
            )
            self.speaker_head = nn.Sequential(
                nn.Linear(d_decoder_hidden, spk_embed),
                nn.Linear(spk_embed, num_speakers),
            )

        # Initialize weights
        self.apply(weight_init())

    def get_encoder_params(self):
        return list(self.encoder.parameters())

    def extract_features(self, x, specaug=True):
        # Convert raw audio to log melspectrograms
        x = self.logmelspec(x)
        # [batch, time, features]

        if self.training and specaug:
            # Spec augmentation
            x = time_mask(freq_mask(x))
        return x

    def encode_features(self, x: torch.Tensor, audio_lens: torch.LongTensor = None):
        # Pass through TDS convolution
        # [batch, time, features] => [batch, features, time]
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        # [batch, features, time] => [batch, time, features]
        x = x.permute(0, 2, 1)

        # Project to decoder dimensionality
        spk_h = self.spk_enc_proj(x) if self.use_speaker_head else None
        x = self.decoder_proj(x)

        # 1 for ignore
        encoder_padding_mask = None
        if audio_lens is not None:
            # The audio length indicator needs to be scaled down by a factor after CNN striding
            scaled_lens = audio_lens // (audio_lens.max() // x.size(1))

            # compute padding mask
            encoder_padding_mask = torch.zeros(
                x.size(0), x.size(1), dtype=torch.bool)
            for i, l in enumerate(scaled_lens.cpu().tolist()):
                encoder_padding_mask[i, l:] = 1
            encoder_padding_mask = encoder_padding_mask.to(x.device)

        return {
            'speaker_out': spk_h,
            'encoder_out': x,
            'encoder_padding_mask': encoder_padding_mask
        }

    def encode(self, x: torch.LongTensor, audio_lens: torch.LongTensor = None):
        """
        Args:
            x - Tensor of raw waveform [batch, length]
        """
        x = self.extract_features(x)
        return self.encode_features(x, audio_lens)

    def decode(self, y_prev, encoder_out, past=None, causal_mask=True):
        """
        Args:
            y_prev: [batch, seq_len]
        """
        # Handle encoder output
        x = encoder_out['encoder_out']
        x = self.dropout(x)
        # [batch, seq_len, features]-> [seq_len, batch, features]
        x = x.permute(1, 0, 2)

        memory_key_padding_mask = encoder_out['encoder_padding_mask']

        # Decode audio features using Transformer
        # [batch, seq_len, features]
        y_prev = self.embedding(y_prev)

        if self.embed_size:
            y_prev = self.embedding_proj(y_prev)

        y_prev = self.pos_dec_encoder(y_prev)

        # Pytorch's native Transformer works with [SEQ, BATCH, FEATURES]
        # => [seq_len, batch, features]
        y_prev = y_prev.permute(1, 0, 2)

        tgt_mask = None
        if causal_mask:
            # Generate causal mask
            n_ctx = y_prev.size(0)
            mask = torch.triu(torch.ones(n_ctx, n_ctx), 1)
            mask = mask.masked_fill(mask == 1, float('-inf'))
            tgt_mask = mask.to(y_prev.device)

        h = self.decoder(y_prev, x, tgt_mask=tgt_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        # Change back to [batch, seq_len, features]
        h = h.permute(1, 0, 2).contiguous()
        
        if self.embed_size:
            h = F.linear(h, self.embedding_proj.weight.t())

        h = self.lm_head(h)
        return h

    def decode_spk(self, y_prev, encoder_out, causal_mask=True):
        """
        Args:
            y_prev: [batch, seq_len]
        """
        # Handle encoder output
        x = encoder_out['speaker_out']
        x = self.dropout(x)
        # [batch, seq_len, features]-> [seq_len, batch, features]
        x = x.permute(1, 0, 2)

        memory_key_padding_mask = encoder_out['encoder_padding_mask']

        # Decode audio features using Transformer
        # [batch, seq_len, features]
        y_prev = self.embedding(y_prev)

        if self.embed_size:
            y_prev = self.embedding_proj(y_prev)

        y_prev = self.pos_dec_encoder(y_prev)

        # Pytorch's native Transformer works with [SEQ, BATCH, FEATURES]
        # => [seq_len, batch, features]
        y_prev = y_prev.permute(1, 0, 2)

        tgt_mask = None
        if causal_mask:
            # Generate causal mask
            n_ctx = y_prev.size(0)
            mask = torch.triu(torch.ones(n_ctx, n_ctx), 1)
            mask = mask.masked_fill(mask == 1, float('-inf'))
            tgt_mask = mask.to(y_prev.device)

        h = self.spk_decoder(y_prev, x, tgt_mask=tgt_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        # Change back to [batch, seq_len, features]
        h = h.permute(1, 0, 2).contiguous()
        h = self.speaker_head(h)
        return h

    def forward(self, x, y_prev, audio_lens):
        encoder_out = self.encode(x, audio_lens)
        lm_out = self.decode(y_prev, encoder_out)
        spk_out = self.decode_spk(y_prev, encoder_out) if self.use_speaker_head else None
        return (lm_out, spk_out), encoder_out


class TDSBlock(nn.Module):
    def __init__(self, hidden, kernel_size, groups, dropout=0.1):
        super().__init__()

        # Depth-wise Grouped Convolution
        self.conv = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=kernel_size, stride=1,
                      groups=groups, padding=kernel_size // 2
                      ),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Pointwise FF
        self.fc = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=1, stride=1),
            nn.Dropout(dropout)
        )

        # Residual weights
        self.resweight = nn.Parameter(torch.Tensor([0]))

    def forward(self, x):
        """
        Args:
            x: Melspec features [batch, features, time]
        """
        # Residual connections with residual dropout
        x = x + self.resweight * self.conv(x)
        x = x + self.resweight * self.fc(x)
        return x


"""
Receptive field test:

import torch
from wildspeech.asr.models import ASRModel
m = ASRModel().cuda()
x = torch.randn(1,80,1000,requires_grad=True, device=torch.device('cuda'))
y = m.encoder(x)
grad = torch.zeros_like(y)
grad[0, :, y.size(-1)//2]=1
y.backward(grad)
len(x.grad[0,0].nonzero())
"""


class TDS(nn.Module):
    """
    Acoustic model
    """

    def __init__(self, input_size, sizes, depths, kernel_size=21, dropout=0.1):
        super().__init__()
        self.extract_block_id = 1
        # Pad edges by receptive field fixes the sensitivity to edges
        # Receptive field, in number of frames
        self.sizes = sizes
        self.blocks = nn.Sequential(*[
            nn.Sequential(
                # Resize the dimensionality
                nn.Conv1d(
                    sizes[i - 1], sizes[i], kernel_size=kernel_size, stride=2, groups=input_size),
                # Residual operations
                nn.Sequential(*[
                    TDSBlock(sizes[i], kernel_size,
                             input_size, dropout=dropout)
                    for _ in range(depths[i - 1])])
            )
            for i in range(1, len(sizes))
        ])

    def extract(self, x):
        """
        Feature extraction network.
        RF: 21 frames (0.21s)
        """
        for i in range(self.extract_block_id):
            x = self.blocks[i](x)
        return x

    def aggregate(self, x):
        """
        Feature aggregation network
        RF: 141 frames (1.41s) or 61 extracted features.
        """
        for i in range(self.extract_block_id, len(self.sizes) - 1):
            x = self.blocks[i](x)
        return x

    def forward(self, x):
        """
        Args:
            x: Melspec features [batch, features, time]
        """
        return self.aggregate(self.extract(x))


class SDModel(nn.Module):
    def __init__(self,
                 num_speakers=6008,
                 n_mels=80,
                 dropout=0.2,
                 embed_size=128):
        super().__init__()
        self.num_speakers = num_speakers

        tds_sizes = [n_mels, 10 * n_mels, 14 * n_mels, 18 * n_mels]
        tds_depths = [2, 3, 6]

        # 1 second of audio (15999 frames) => 100 frames. Each frame is 0.01s.
        # Each frame has a window of 25ms with 10ms stride.
        self.logmelspec = LogMelSpec(n_mels=n_mels)

        # 141 frames => 1 feature. 1 feature ~1.41 seconds receptive field
        self.encoder = TDS(n_mels, tds_sizes, tds_depths, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        # Projects encoder size to speaker ID
        self.spk_embed_proj = nn.Linear(tds_sizes[-1], embed_size)
        self.spk_logit_proj = nn.Linear(embed_size, num_speakers)

        # Initialize weights
        self.apply(weight_init())

    def get_encoder_params(self):
        return list(self.encoder.parameters())

    def extract_features(self, x, specaug=True):
        # Convert raw audio to log melspectrograms
        x = self.logmelspec(x)
        # [batch, time, features]

        if self.training and specaug:
            # Spec augmentation
            x = time_mask(freq_mask(x))
        return x

    def encode_features(self, x: torch.Tensor, audio_lens: torch.LongTensor = None):
        # Pass through TDS convolution
        # [batch, time, features] => [batch, features, time]
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        # [batch, features, time] => [batch, time, features]
        x = x.permute(0, 2, 1)

        encoder_padding_mask = None
        if audio_lens is not None:
            # The audio length indicator needs to be scaled down by a factor after CNN striding
            scaled_lens = audio_lens // (audio_lens.max() // x.size(1))

            # compute padding mask
            encoder_padding_mask = torch.zeros(
                x.size(0), x.size(1), dtype=torch.bool)
            for i, l in enumerate(scaled_lens.cpu().tolist()):
                encoder_padding_mask[i, l:] = 1
            encoder_padding_mask = encoder_padding_mask.to(x.device)

        return {
            'encoder_out': x,
            'encoder_padding_mask': encoder_padding_mask
        }

    def encode(self, x: torch.LongTensor, audio_lens: torch.LongTensor = None):
        """
        Args:
            x - Tensor of raw waveform [batch, length]
        """
        x = self.extract_features(x)
        return self.encode_features(x, audio_lens)

    def decode(self, encoder_out, past=None, causal_mask=True):
        # Handle encoder output [batch, seq_len, features]
        x = encoder_out['encoder_out']
        x = self.dropout(x)

        # [batch, seq_len, num_speakers]
        logits = self.spk_logit_proj(self.spk_embed_proj(x))

        return logits

    def forward(self, x, audio_lens):
        encoder_out = self.encode(x, audio_lens)
        return self.decode(encoder_out), encoder_out


class ModRZTXDecoderLayer(nn.Module):
    """ Modified version of decoder layer that caches the attention weights for analysis """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.resweight = nn.Parameter(torch.Tensor([0]))
        self.resweight_src = nn.Parameter(torch.Tensor([0]))

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                 key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2) * self.resweight
        tgt2, src_attn_weights = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                                     key_padding_mask=memory_key_padding_mask)
        self.src_attn_weights = src_attn_weights.detach()
        tgt = tgt + self.dropout2(tgt2) * self.resweight_src

        if hasattr(self, "activation"):
            tgt2 = self.linear2(self.dropout(
                self.activation(self.linear1(tgt))))
        else:  # for backward compatibility
            tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2) * self.resweight
        return tgt


def freq_mask(spec, F=27, num_masks=2):
    # https://github.com/zcaceres/spec_augment/blob/master/SpecAugment.ipynb
    cloned = spec.clone()
    num_mel_channels = cloned.shape[2]

    for b in range(cloned.size(0)):
        for _ in range(num_masks):
            f = random.randrange(0, F)
            f_zero = random.randrange(0, num_mel_channels - f)

            # avoids randrange error if values are equal and range is empty
            if (f_zero == f_zero + f):
                return cloned

            mask_end = random.randrange(f_zero, f_zero + f)
            cloned[b, :, f_zero:mask_end] = 0
    return cloned


def time_mask(spec, T=100, num_masks=2):
    # https://github.com/zcaceres/spec_augment/blob/master/SpecAugment.ipynb
    cloned = spec.clone()
    len_spectro = cloned.shape[1]

    for b in range(cloned.size(0)):
        for _ in range(num_masks):
            t = random.randrange(0, T)
            t_zero = random.randrange(0, len_spectro - t)

            # avoids randrange error if values are equal and range is empty
            if (t_zero == t_zero + t):
                return cloned

            mask_end = random.randrange(t_zero, t_zero + t)
            cloned[b, t_zero:mask_end, :] = 0
    return cloned
