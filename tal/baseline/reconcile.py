from wildspeech.wder_search import compute_sequence_match
from collections import defaultdict
import re
from unidecode import unidecode
import torch.nn.functional as F
from wildspeech.asr.data.aligned import ASRAlignedDataset, ASRAlignedCollater
from wildspeech.asr.tokenizers.sentencepiece import Tokenizer
import pickle
from librosa.core import get_duration
import torchaudio
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
from wildspeech.baseline.speaker_system import System
from apex import amp
import os
import torch
import argparse
from argparse import Namespace
from wildspeech.asr.args import get_argparser

args = '--project asr-tal --max-secs 30 --train-data ./data/tal-final/train/ --valid-data ./data/tal-final/valid/ --test-data ./data/tal-final/test/ --model-type 2x --batch-size 35 --max-epochs 100 --max-steps 40000 --lr 3e-4 --tokenizer taltoken-cased.model --smoothing 0.05 --num-speakers 6008 --name tal-diarize-baseline-1'.split(
    ' ')

t_argparser = get_argparser(is_train=True)
args = t_argparser.parse_args(args)

model = System(args)
model.cuda()
model = amp.initialize(model, opt_level='O1')
model.eval()

mdir = './models/wild-speech'
model_name = 'tal-diarize-ft'
model_folder = os.path.join(mdir, model_name)
wloc = os.path.join(model_folder, 'checkpoints', '_ckpt_epoch_28.ckpt')
print('Loading existing model weights...', wloc)
checkpoint = torch.load(wloc, map_location=lambda storage, loc: storage)
state_dict = checkpoint['state_dict']

model.load_state_dict(state_dict, strict=not args.no_strict)
# give model a chance to load something
model.on_load_checkpoint(checkpoint)


stride = 0.08
fwidth = 1.41


def get_speaker_frames(ep_turns, file_duration):
    # Get gold labels
    speakers = [u['speaker'] for u in ep_turns]
    turn_starts = [u['utterance_start'] for u in ep_turns]
    turn_ends = [u['utterance_end'] for u in ep_turns]
    # Create speaker frames
    current_turn = 0
    speaker_frames = []
    for frame_start in tqdm(np.arange(0, file_duration + 0.01 - fwidth, stride)):
        frame_end = frame_start + fwidth
        frame_mid = frame_start + 0.5 * fwidth
        try:
            # Determine whether to advance the turn
            if frame_start > turn_ends[current_turn] or frame_mid > turn_ends[current_turn]:
                current_turn += 1
            # Determine whether we're too far behind
            if frame_end < turn_starts[current_turn] or frame_mid < turn_starts[current_turn]:
                speaker_frames.append(-1)
                continue
            speaker_frames.append(speakers[current_turn])
        except:
            speaker_frames.append(-1)
    return speaker_frames


def get_speaker_ids(wav_loc, model):
    x_wav, _ = torchaudio.load(wav_loc)
    x_wav = x_wav.cuda().half()
    with torch.no_grad():
        feat = model.model.encode(x_wav, None)
        feat_mat = np.matrix(model.model.spk_embed_proj(
            feat['encoder_out']
        ).detach().cpu().numpy())
        pred_ids = torch.argmax(model.model.decode(feat), dim=-1)
    return feat_mat, pred_ids.squeeze().detach().cpu().numpy().tolist()


for split in ['test', 'valid']:
    fdir = './data/tal-final/{}/'.format(split)
    tloc = os.path.join(fdir, 'transcript.pkl')
    transcripts = pd.read_pickle(tloc)
    # Get frames and features
    gold_speaker_frames = dict()
    pred_speaker_ids = dict()
    pred_speaker_features = dict()
    for ep, turns in transcripts.items():
        wav_loc = os.path.join(fdir, '{}.wav'.format(ep))
        wav_s = get_duration(filename=wav_loc)
        gold_speaker_frames[ep] = get_speaker_frames(turns, wav_s)
        feat_mat, pred_ids = get_speaker_ids(wav_loc, model)
        pred_speaker_ids[ep] = pred_ids
        pred_speaker_features[ep] = feat_mat
    gold_loc = os.path.join(
        model_folder, 'ref_speaker_ids_{}.pkl'.format(split))
    with open(gold_loc, 'wb') as wf:
        pickle.dump(gold_speaker_frames, wf)
    print('Dumped {:,} speaker frames to {} ({:.2f})'.format(
        sum(len(s) for s in gold_speaker_frames.values()),
        gold_loc, os.path.getsize(gold_loc) / 1024 / 1024
    ))
    pred_loc = os.path.join(
        model_folder, 'hyp_speaker_ids_{}.pkl'.format(split))
    with open(pred_loc, 'wb') as wf:
        pickle.dump(pred_speaker_ids, wf)
    print('Dumped {:,} predicted speaker frames to {} ({:.2f})'.format(
        sum(len(s) for s in pred_speaker_ids.values()),
        pred_loc, os.path.getsize(pred_loc) / 1024 / 1024
    ))
    pf_loc = os.path.join(
        model_folder, 'hyp_speaker_features_{}.pkl'.format(split))
    with open(pf_loc, 'wb') as wf:
        pickle.dump(pred_speaker_features, wf)
    print('Dumped {:,} predicted speaker features to {} ({:.2f})'.format(
        sum(len(s) for s in pred_speaker_features.values()),
        pf_loc, os.path.getsize(pf_loc) / 1024 / 1024
    ))

# Force exit
exit(0)


def get_relative_ids(speakers):
    all_speakers = []
    ids = []
    for s in speakers:
        try:
            spk_id = all_speakers.index(s)
        except:
            spk_id = len(all_speakers)
            all_speakers.append(s)
        ids.append(spk_id)
    return ids, all_speakers


ders = []
for ep in transcripts:
    gold_ids = get_relative_ids(gold_speaker_frames[ep])
    pred_ids = get_relative_ids(pred_speaker_frames[ep])
    ref_spk_labels, hyp_spk_labels, match_accuracy = compute_sequence_match(
        gold_ids, pred_ids)
    der = 1 - match_accuracy
    print('{} DER: {:.2f}'.format(ep, der * 100.0))
    ders.append(der)

print('{:.2f} DER'.format(np.mean(ders) * 100.0))


gold_loc = os.path.join(fdir, 'ref_speaker_frames.pkl')
with open(gold_loc, 'wb') as wf:
    pickle.dump(gold_speaker_frames, wf)

print('Dumped {:,} speaker frames to {} ({:.2f})'.format(
    sum(len(s) for s in gold_speaker_frames.values()),
    gold_loc, os.path.getsize(gold_loc) / 1024 / 1024
))

# ALIGNED

tokenizer = Tokenizer(cache_path='./cache/tokenizer/taltoken-cased.model')
ds = ASRAlignedDataset(
    fdir,
    tokenizer=tokenizer,
    num_utterances=1,
    min_segment_duration=3,
    max_segment_duration=args.max_secs,
    speaker_map_loc=os.path.join(fdir, 'speaker_map.json'),
    tokenizer_speakers=True,
    return_spk_ids=True
)

generated = []
for i in tqdm(range(len(ds))):
    u = ds.index[i]
    x_wav, text, spk_ids, _ = ds[i]
    x_wav = x_wav.unsqueeze(0).cuda().half()
    with torch.no_grad():
        feat = model.model.encode(x_wav, None)
        feat_mat = np.matrix(model.model.spk_embed_proj(
            feat['encoder_out']
        ).detach().cpu().numpy())
        pred_id = torch.mode(torch.argmax(
            model.model.decode(feat), dim=-1))[0][0].item()
    generated.append((u, (feat_mat, pred_id)))

loc = os.path.join(mdir, model_name, 'speakers_baseline_aligned.pkl')
with open(loc, 'wb') as wf:
    pickle.dump(generated, wf)

print('Dumped {:,} generations to {} ({:.2f} MB)'.format(
    len(generated), loc, os.path.getsize(loc) / 1024 / 1024))

# UNALIGNED
utterance_breaks = dict()
for ep, turns in transcripts.items():
    ep_wavloc = os.path.join(fdir, '{}.wav'.format(ep))
    x_wav, sr = torchaudio.load(wavloc)
    audio_lens = torch.LongTensor([len(x) for x in x_wav])
    # Convert
    x_wav = x_wav.cuda().half()
    audio_lens = audio_lens.cuda()
    with torch.no_grad():
        # 1 x n_features x 1440
        features = model.model.spk_embed_proj(
            model.model.encode(x_wav, audio_lens)['encoder_out']).squeeze(0)
    break


threshold = 1.0
for i in range(len(features) - 1):
    cdist = (
        1.0 - F.cosine_similarity(features[i], features[i+1], dim=0)).item()
    break


# Align stuff
speaker_features = pd.read_pickle(os.path.join(
    './models/wild-speech/tal-diarize-baseline-1/speakers_baseline_aligned.pkl'))
speaker_map = {
    (u[1][0]['episode'], u[1][0]['utterance_start']): (
        re.sub(' +', ' ', unidecode(u[1][0]['utterance']).strip()),
        emb,
        voted_id
    ) for u, (emb, voted_id) in speaker_features
}
with open('./data/tal-final/valid/transcript.pkl', 'rb') as f:
    transcripts = pickle.load(f)
    # Clean some utterances
    utt_to_transcript = {re.sub(' +', ' ', unidecode(utt['utterance']).strip()): (
        k, utt) for k, utts in transcripts.items() for utt in utts}

# transcripts = pd.read_pickle('./data/tal-final/valid/transcript.pkl')
# utt_to_transcript = {re.sub(' +', ' ', unidecode(utt['utterance']).strip()): (k, utt) for k, utts in transcripts.items() for utt in utts}
test_results = pd.read_pickle(
    './models/wild-speech/tal-tds-baseline-1/aligned/test_result.pkl')

episode_refs = defaultdict(list)
episode_hyps = defaultdict(list)

no_speaker_emb = []
for refs, hyps in test_results:
    for ref, hyp in zip(refs, hyps):
        ref_utt, _ = ref
        hyp_utt, _ = hyp
        if ref_utt.strip() not in utt_to_transcript:
            continue
        ep_name, utt = utt_to_transcript[ref_utt]
        try:
            ref_u, hyp_emb, hyp_id = speaker_map[(
                ep_name, utt['utterance_start'])]
        except KeyError:
            ref_u = ref_utt.strip()
            hyp_emb = None
            hyp_id = None
            no_speaker_emb.append((ep_name, utt['utterance_start']))
        ref_id = utt['speaker']
        episode_refs[ep_name].append((
            utt['utterance_start'],
            ref_u,
            ref_id,
            utt['role']
        ))
        episode_hyps[ep_name].append((
            utt['utterance_start'],
            hyp_utt,
            (hyp_emb, hyp_id),
            utt['role'],
        ))

wder_input = []
for e in episode_refs:
    ref_examples = [
        (r_u, r_s, r_r) for st, r_u, r_s, r_r in sorted(episode_refs[e], key=lambda x: x[0])
    ]
    hyp_examples = [
        (h_u, h_s, h_r) for st, h_u, h_s, h_r in sorted(episode_hyps[e], key=lambda x: x[0])
    ]
    wder_input.append((
        ref_examples, hyp_examples
    ))

fl = './models/wild-speech/tal-diarize-baseline-1/speakers_synced_aligned.pkl'
with open(fl, 'wb') as wf:
    pickle.dump(wder_input, wf)

print('{:,} inputs dumped to {} ({:.2f} MB)'.format(
    len(wder_input), fl, os.path.getsize(fl) / 1024 / 1024))


skipped = 0
aligned = []
print('Utterances in transcript.pkl:', len(utt_to_transcript))
print('Trying to match utterances:', len(data))
for refs, hyps in data:
    for ref, hyp in zip(refs, hyps):
        ref_utt = ref[0]

        if ref_utt.strip() in utt_to_transcript:
            ep_name, utt = utt_to_transcript[ref_utt]

            ref_words = word_tokenize(ref_utt)
            hyp_words = word_tokenize(hyp[0])

            aligned.append({
                'episode': ep_name,
                'utterance': utt,
                'hypothesis': hyp[0],
                'hypothesis_speaker': hyp[1],
                'reference': ref[0],
                'reference_speaker': ref[1],
                'wer': editdistance.eval(ref_words, hyp_words) / len(ref_words)
            })
        else:
            print('Missing:', ref_utt)
            skipped += 1

# Sort by WER
aligned = sorted(aligned, key=lambda x: x['wer'], reverse=True)
with open(args.out_file, 'wb') as f:
    pickle.dump(aligned, f)

print('Done. Skipped {} utterances because they cannot be found.'.format(skipped))
