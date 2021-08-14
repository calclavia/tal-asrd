import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import numpy as np

from tqdm import tqdm
from glob import glob
from wildspeech.asr.system import System
from wildspeech import get_device, debug_log, count_parameters, SuppressPrint
from wildspeech.asr.speech_detect import get_speech_frames
from librosa.core import get_duration
from mutagen import File
from functools import partial
from typing import Sequence
from difflib import SequenceMatcher

USE_CUDA, DEVICE = get_device()
DEFAULT_SR = 16000

DEBUG = partial(debug_log, debug=False)


def overlap_ix(a: str, b: str, word_overlap: int = 5) -> Sequence[int]:
    overlap_a_ix = len(a) - len(' '.join(a.split()[-word_overlap:]))
    overlap_b_ix = len(' '.join(b.split()[:word_overlap+1]))  # EOS
    return overlap_a_ix, overlap_b_ix


def splice_ix(a: str, b: str, word_overlap: int = 5) -> str:
    # Overlap indices
    aix, bix = overlap_ix(a, b, word_overlap)

    # Match them!
    seq = SequenceMatcher(None, a, b)
    match = seq.find_longest_match(aix, len(a), 0, bix)
    if not match:
        return len(a), 0

    a_end_ix, b_start_ix, match_size = match
    # Return only a 1 or 2 word match
    if match_size < 5:  # less than 1 or 2 word match
        return len(a), 0

    # Splice them together
    return a_end_ix, b_start_ix


def splice_strings(strs: list, word_overlap: int = 20) -> str:
    # Initialize with first string
    first_end, a_start = splice_ix(
        strs[0],
        strs[1],
        word_overlap
    )
    output_str = strs[0][:first_end].strip()

    # Move through, clobbering as we go
    for i in range(1, len(strs) - 1):
        a_end, b_start = splice_ix(
            strs[i],
            strs[i+1],
            word_overlap
        )
        output_str += ' ' + strs[i][a_start:a_end].strip()
        # Moving on to next rolling batch
        a_start = b_start

    # Tack on final string
    output_str += ' ' + strs[-1][a_start:].strip()
    return output_str


def transcribe_file(
        audio_path: str,
        model: nn.Module,
        window_frames: int,
        stride_frames: int,
        batch_size: int = 15,
        log: bool = False,
        debug: bool = False,
        beam_width: int = 4,
        lm_weight: float = 0.,
        length: int = 60,
        truncate: float = -1,
        device: str = 'cpu',
        speech_only: bool = False,
        splice: bool = False,
        use_eot: bool = True,
) -> Sequence[str]:
    debug_fxn = partial(debug_log, debug=debug)

    # Load data, at the correct sample rate
    si, _ = torchaudio.info(audio_path)
    x_wav, sr = torchaudio.load(audio_path)
    assert sr == si.rate, sr

    if sr != DEFAULT_SR:
        resample = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=DEFAULT_SR)
        x_wav = resample(x_wav)
        print('Resampled!')

    x_wav = x_wav.squeeze(0)
    if truncate > 0.:
        x_wav = x_wav[:int(truncate * len(x_wav))]

    if speech_only:
        x_wav = get_speech_frames(
            audio_segment=x_wav.numpy(),
            frame_duration_ms=30,
            sample_rate=DEFAULT_SR,
            vad_level=3,
            padding_duration_ms=300,
            log=False
        )
    debug_fxn(x_wav, 'x_wav')

    # Construct a batch
    n_windows = int(np.ceil((len(x_wav) - window_frames) / stride_frames)) + 1

    gen_outputs = []
    current_batch = []

    # Log if necessary
    i_range = range(n_windows)
    if log:
        i_range = tqdm(i_range)
    for i in i_range:
        # Grab the window
        w_start = stride_frames * i
        w_end = w_start + window_frames
        window = x_wav[w_start:w_end]
        current_batch.append(window)
        debug_fxn(current_batch, 'Window bounds: {} --> {}'.format(
            w_start, w_end))

        # Possible truncation of last batch
        if len(current_batch) == batch_size or i == n_windows - 1:
            # Generate for current batch
            batch_outputs = transcribe_batch(
                batch=current_batch,
                model=model,
                beam_width=beam_width,
                lm_weight=lm_weight,
                length=length,
                device=device,
                use_eot=use_eot,
            )
            debug_fxn(batch_outputs, 'batch_outputs')
            gen_outputs.extend([
                model.tokenizer.decode(b) for b in batch_outputs
                if b is not None
            ])

            # Reset current batch
            current_batch = []

    if splice:
        # Speak around 2/3 words on average per second
        merge_window = 3 * ((window_frames - stride_frames) // DEFAULT_SR)
        return splice_strings(gen_outputs, merge_window)

    return gen_outputs


def transcribe_batch(batch: list,
                     model: nn.Module,
                     beam_width: int = 4,
                     lm_weight: float = 0.,
                     length: int = 60,
                     device: str = 'cpu',
                     use_eot: bool = True) -> Sequence[str]:
    with torch.no_grad():
        # Pad batches if necessary
        audio_lens = [len(a) for a in batch]
        max_audio_len = max(audio_lens)

        # B x T : Audio frames
        batch_audio = torch.stack([
            F.pad(a, (0, max_audio_len - len(a)), 'constant', 0.)
            for a in batch
        ]).to(device)

        # B : Length of audio
        batch_audio_lens = torch.LongTensor(audio_lens).to(device)

        # B x 1 : Generation priming token (BOS)
        batch_generated = torch.LongTensor(
            # HACK TO GET IT WORKING WITH THE EOT TOKEN USAGE FROM SYSTEM.PY
            [[model.tokenizer.bos_token_id if use_eot else model.tokenizer.eos_token_id]
             for _ in range(len(batch))]).to(device)

        # Create generated output
        gen_output = model.generate(
            audio_x=batch_audio,
            generated=batch_generated,
            audio_lens=batch_audio_lens,
            length=length,
            beam_width=beam_width,
            terminate_token=model.tokenizer.eot_token_id if use_eot else None,
            lm_weight=lm_weight,
        )

    return gen_output


class DefaultArgs(object):
    def __init__(
            self,
            weights_path:
            str = '/root/data4/bernard-asr-models/tal-tune2/avg_last_5.pt',
            cache_path: str = '/root/data4',
            batch_size: int = 32,
            val_batch_size: int = 32,
            beam_width: int = 4,
            lm_weight: float = 0.,
            half_precision: bool = True,
            **kwargs):
        self.weights_path = weights_path
        self.cache_path = cache_path
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.beam_width = beam_width
        self.lm_weight = lm_weight
        self.half_precision = half_precision

        # Any other arbitrary arguments!
        self.__dict__.update(kwargs)


def load_model(args: object = None, device: str = DEVICE) -> nn.Module:
    if args is None:
        args = DefaultArgs()

    # TODO: Custom loading because lightning seems to be broken.
    # load on CPU only to avoid OOM issues
    # then its up to user to put back on GPUs
    checkpoint = torch.load(
        args.weights_path, map_location=lambda storage, loc: storage)

    # load the state_dict on the model automatically
    with SuppressPrint():
        model = System(args)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    # give model a chance to load something
    model.on_load_checkpoint(checkpoint)
    if args.half_precision:
        model.half()
    model.to(device)
    model.eval()
    print('Loaded model{}, in evaluation mode with {:,} parameters'.format(
        ' with half precision' if args.half_precision else '',
        count_parameters(model),
    ))

    return model


"""
MODEL: TAL_TUNE2
python3 -u -m bernard.asr.transcribe \
--data-path /root/data4/bernard-tal-splits \
--weights-path /root/data4/bernard-asr-models/tal-tune2/avg_last_5.pt \
--cache-path /root/data4 \
--batch-size 15 \
--window-size 20 \
--beam-width 3 \
--half-precision \
--splice \
--exp tal-tune2 \
--debug \
--out-path /root/data4/bernard-asr-generation

nohup python3 -u -m bernard.asr.transcribe \
--data-path /root/data4/bernard-tal-splits \
--weights-path /root/data4/bernard-asr-models/tal-tune2/avg_last_5.pt \
--cache-path /root/data4 \
--batch-size 15 \
--window-size 20 \
--beam-width 3 \
--half-precision \
--splice \
--exp tal-tune2 \
--out-path /root/data4/bernard-asr-generation > /root/data4/tal-transcribe-tt2.log &

tail -f /root/data4/tal-transcribe-tt2.log

MODEL: TAL-SPEAKERS-FIX2
nohup python3 -u -m bernard.asr.transcribe \
--data-path /root/data4/bernard-tal-splits \
--weights-path /root/data4/bernard-asr-models/tal-speakers-fix2/checkpoints/_ckpt_epoch_9.ckpt \
--cache-path /root/data4 \
--batch-size 8 \
--window-size 40 \
--beam-width 3 \
--half-precision \
--splice \
--window-overlap 0.25 \
--exp tal-speakers-fix2 \
--out-path /root/data4/bernard-asr-generation > /root/data4/tal-transcribe-tsf2.log &

tail -f /root/data4/tal-transcribe-tsf2.log
"""
if __name__ == '__main__':
    import argparse

    # Create parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-path', type=str, default=None)
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--out-path', type=str, default='./generation')
    parser.add_argument('--cache-path', type=str, default='./cache')
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--val-batch-size', type=int, default=None)
    parser.add_argument('--beam-width', type=int, default=3)
    parser.add_argument(
        '--truncate', type=float, default=-1, help='Truncate file')
    parser.add_argument(
        '--debug', action='store_true', default=False, help='Debug mode')
    parser.add_argument(
        '--window-size', type=int, default=10, help='Window size in seconds')
    parser.add_argument(
        '--use-speakers',
        action='store_true',
        default=False,
        help='Speaker model')
    parser.add_argument(
        '--half-precision',
        action='store_true',
        default=False,
        help='Run at half precision')
    parser.add_argument(
        '--lm-weight', type=float, default=0., help='LM weight')
    parser.add_argument(
        '--overwrite', action='store_true', default=False, help='Overwrite existing output'
    )
    parser.add_argument(
        '--splice', action='store_true', default=False, help='Splice together sentences'
    )
    parser.add_argument(
        '--window-overlap', type=float, default=0.10, help='Window overlap size'
    )
    parser.add_argument(
        '--exp', type=str, required=True, help='Experiment name'
    )

    # Parse arguments
    args = parser.parse_args()
    data_dir = args.data_path
    batch_size = args.batch_size
    beam_width = args.beam_width
    weights_path = args.weights_path
    window_s = args.window_size
    truncate = args.truncate
    lm_weight = args.lm_weight
    out_dir = args.out_path
    overwrite = args.overwrite
    splice = args.splice
    exp_name = args.exp
    window_overlap = args.window_overlap
    if args.debug:
        DEBUG = partial(debug_log, debug=True)
        DEBUG([], 'Debugging!!!')

    # Create the output path in case it's necessary
    os.makedirs(out_dir, exist_ok=True)

    # Load model
    model = load_model(args, DEVICE)

    # Window sizes in seconds
    window_f = window_s * DEFAULT_SR
    # K% overlap of T end / T+1 beignning
    stride_s = int(window_s * (1 - window_overlap))
    stride_f = stride_s * DEFAULT_SR
    print(
        'Transcribing with {:,}-second windows, stride {:,} s & batch size {:,}'.
        format(window_s, stride_s, batch_size))

    # Transcribe audio
    audio_files = glob(
        os.path.join(data_dir, 'test', '*.wav'))  # WAV files in each split
    if truncate > 0:
        audio_files = audio_files[:1]

    for ix, audio_path in enumerate(audio_files, 1):
        filestub = os.path.basename(audio_path).rsplit('.', 1)[0]
        output_path = os.path.join(
            out_dir, '{}__{}.txt'.format(exp_name, filestub))
        print('Transcribing file {:,}/{:,}'.format(
            ix, len(audio_files)
        ))

        if not overwrite and os.path.exists(output_path):
            print('{} Already exists! Skipping...'.format(output_path))
            continue

        # Transcribe one piece of audio
        try:
            dur = get_duration(filename=audio_path)
        except Exception as e:
            dur = File(audio_path).info.length
        print('Transcribing file: {} with length {}'.format(
            audio_path, time.strftime('%H:%M:%S', time.gmtime(dur))))
        gen_output = transcribe_file(
            audio_path,
            model,
            window_frames=window_f,
            stride_frames=stride_f,
            batch_size=batch_size,
            log=True,
            debug=args.debug,
            beam_width=beam_width,
            lm_weight=lm_weight,
            length=window_s * 6,  # Approx. density = 6 words/s
            truncate=truncate,
            device=DEVICE,
            splice=splice,
            use_eot=not 'tal-tune2' in weights_path  # Hacky as hell
        )

        # Save the outputs
        with open(output_path, 'w+') as wf:
            n_lines = 0
            if isinstance(gen_output, list):
                for line in gen_output:
                    wf.write(line)
                    print(line)
                    n_lines += 1
            else:
                wf.write(gen_output)
                lines = gen_output.split('<|endoftext|>')
                n_lines = len(lines)
                print('\n'.join(lines))
        print('Transcribed {:,} lines to {}'.format(n_lines, output_path))

        # Save gold
        gold_path = audio_path.replace('.wav', '.jsonl')
        with open(gold_path, 'r+') as rf:
            gold_txt = ''
            last_speaker = None
            for line in rf:
                data = json.loads(line)
                gold_txt += data['utterance']
                if last_speaker != data['speaker']:
                    gold_txt += ' <|endoftext|> '
                    last_speaker = data['speaker']
        gold_out_path = os.path.join(out_dir, 'gold_{}.txt'.format(filestub))
        with open(gold_out_path, 'w+') as wf:
            wf.write(gold_txt)

    print('DONE TRANSCRIBING {}'.format(audio_path))
