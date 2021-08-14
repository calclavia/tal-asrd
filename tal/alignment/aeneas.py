from aeneas.executetask import ExecuteTask
from aeneas.task import Task
import re
import sys
import os
import requests
import json
import pickle
import pandas as pd
import numpy as np
import wave
import torchaudio
import multiprocessing
import string
from itertools import chain
from tqdm import tqdm
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from functools import partial
from nltk import word_tokenize

from nltk.tokenize import TweetTokenizer
WT = TweetTokenizer()


def tokenize(stuff):
    return WT.tokenize(stuff)


def full_force_align(ep_turns_tuple: tuple, data_dir: str, out_dir: str):
    # Logistics
    start = datetime.now()
    episode, turns = ep_turns_tuple
    wav_loc = os.path.join(data_dir, '{}.wav'.format(episode))
    alignment_file = os.path.join(out_dir, '{}-alignment.json'.format(episode))
    if os.path.exists(alignment_file):
        print('{} exists! Skipping.'.format(alignment_file))
        return

    # Set up plaintext files
    t_tokenized = [w for w in word_tokenize(
        ' '.join(t['utterance'] for t in turns)
    ) if w not in string.punctuation]
    print('{:,} turns ({:,} words)'.format(len(turns), len(t_tokenized)))
    plaintext_file = os.path.join(out_dir, '{}.txt'.format(episode))
    with open(plaintext_file, 'w+') as wf:
        for w in t_tokenized:
            _ = wf.write('{}\n'.format(w))

    print('Written plaintext words in order to {} {:.3f} MB'.format(
        plaintext_file, os.path.getsize(plaintext_file) / 1024 / 1024))

    # create Task object
    config_string = u"task_language=eng|is_text_type=plain|os_task_file_format=json"
    task = Task(config_string=config_string)
    task.audio_file_path_absolute = wav_loc
    task.text_file_path_absolute = plaintext_file
    task.sync_map_file_path_absolute = alignment_file

    # process Task
    ExecuteTask(task).execute()
    print('{} - Aligned words for {}'.format(datetime.now() - start, episode))

    # output sync map to file
    task.output_sync_map_file()
    print('{} - Alignment map saved to {} ({:.3f} MB)'.format(
        datetime.now() - start,
        alignment_file, os.path.getsize(alignment_file) / 1024 / 1024
    ))


def align_episode(ep_turns_tuple: tuple, data_dir: str, align_dir: str, log: bool = False):
    # Logistics
    start = datetime.now()
    e, turns = ep_turns_tuple

    ep_alignments = []
    for ii, t in tqdm(enumerate(turns), total=len(turns)):
        # Dump to wav file
        wav_loc = os.path.join(data_dir, '{}.wav'.format(e))
        u_start = t['utterance_start']
        duration = t['utterance_end'] - u_start

        # Using load_wav here will normalize, but then save as normalized values
        try:
            offset = int(u_start * 16000)
            n_frames = int(duration * 16000)
            audio, sr = torchaudio.load(
                wav_loc, offset=offset, num_frames=n_frames)
        except:
            print('Start: {}'.format(u_start))
            print('End: {}'.format(t['utterance_end']))
            print('Duration: {}'.format(duration))
            print('Offset: {}'.format(offset))
            print('Frames: {}'.format(n_frames))
            print(
                '--------------\nEpisode {}, turn {}:\n{}\n-----------------'.format(e, ii, t))
            raise
        wav_temp = os.path.join(align_dir, 'tmp_{}.wav'.format(e))
        torchaudio.save(wav_temp, audio, sample_rate=16000,
                        precision=16, channels_first=True)
        if log:
            print('{} - Saved {:,.2f} s of audio to {} ({:.3f} KB)'.format(
                datetime.now() - start, duration, wav_temp, os.path.getsize(wav_temp) / 1024
            ))

        # Get words
        txt_temp = os.path.join(align_dir, 'tmp_{}.txt'.format(e))
        t_tokenized = [(i, t) for i, t in enumerate(
            tokenize(t['utterance'])) if t not in string.punctuation]
        indices, words = zip(*t_tokenized)
        with open(txt_temp, 'w+') as wf:
            _ = wf.write('\n'.join(words))
        if log:
            print('{} - Saved {:,} words to {} ({:.3f} KB)'.format(
                datetime.now() - start, len(words), txt_temp, os.path.getsize(txt_temp) / 1024
            ))

        # create Task object
        alignment_file = os.path.join(
            align_dir, 'tmp_{}_alignment.json'.format(e))
        config_string = u"task_language=eng|is_text_type=plain|os_task_file_format=json"
        task = Task(config_string=config_string)
        task.audio_file_path_absolute = wav_temp
        task.text_file_path_absolute = txt_temp
        task.sync_map_file_path_absolute = alignment_file

        # process Task
        ExecuteTask(task).execute()
        if log:
            print('{} - Aligned {:,} words'.format(datetime.now() - start, len(words)))

        # output sync map to file
        task.output_sync_map_file()
        if log:
            print('{} - Alignment map saved to {} ({:.3f} MB)'.format(
                datetime.now() - start,
                alignment_file, os.path.getsize(alignment_file) / 1024 / 1024
            ))

        # Get and format alignments
        with open(alignment_file, 'r+') as rf:
            raw_alignments = json.load(rf)['fragments']

        # Format: (begin, end, index in TweetTokenize'd utterance)
        alignments = [
            (float(r['begin']) + u_start,
             float(r['end']) + u_start, indices[i])
            for i, r in enumerate(raw_alignments)
        ]
        ep_alignments.append(alignments)
        if log:
            print('Processed {:,} alignments for the utterance'.format(
                len(alignments)))

        # Clean up
        os.remove(alignment_file)
        os.remove(wav_temp)
        os.remove(txt_temp)

    print('{} - Processed {:,} turn alignments for episode {}'.format(
        datetime.now() - start, len(turns), e
    ))

    return e, ep_alignments


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--utterance', action='store_true', default=False)
    parser.add_argument('--in-dir', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--pool', type=int, default=1)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--transcript-stub', type=str, default='transcript')
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    start_all = datetime.now()

    # Make out directory
    os.makedirs(args.out_dir, exist_ok=True)

    # wa<START>-<END>
    start_ix, end_ix = map(int, args.config[len('wa'):].split('-'))

    # Get transcripts
    t_loc = os.path.join(args.in_dir, '{}.pkl'.format(args.transcript_stub))
    transcripts = pd.read_pickle(t_loc)
    print('Loaded {:,} episodes from {} ({:.3f} MB)'.format(
        len(transcripts), t_loc, os.path.getsize(t_loc) / 1024 / 1024
    ))
    candidate_episodes = sorted(transcripts.keys())[start_ix:end_ix]
    # already_aligned = {
    #     ep for ep in candidate_episodes if os.path.exists(
    #         os.path.join(args.out_dir, '{}-alignment.json'.format(ep)))}
    ep_turns = [
        (ep, trans) for ep, trans in transcripts.items()
        if ep in candidate_episodes  # and ep not in already_aligned
    ]
    print('Aligning {:,}/{:,} episodes from {} ({}) to {} ({})'.format(
        len(ep_turns), len(transcripts),
        candidate_episodes[0], start_ix, end_ix, candidate_episodes[-1],
    ))

    # Align them
    if args.utterance:
        mapped_fn = partial(
            align_episode,
            data_dir=args.in_dir,
            align_dir=args.out_dir,
            log=args.verbose,
        )
    else:
        mapped_fn = partial(
            full_force_align,
            data_dir=args.in_dir,
            out_dir=args.out_dir,
        )
    with multiprocessing.Pool(processes=args.pool) as pool:
        ep_alignment_tuples = pool.map(mapped_fn, ep_turns)

    # Trange
    aligned_t_loc = os.path.join(
        args.in_dir, '{}-aligned.pkl'.format(args.transcript_stub))
    for ep, ep_aligns in ep_alignment_tuples:
        for i, u_align in enumerate(ep_aligns):
            transcripts[ep][i]['alignments'] = u_align
    with open(aligned_t_loc, 'wb') as wf:
        pickle.dump(transcripts, wf)
    print('{} - Dumped {:,} episodes with alignments to {} ({:.3f} MB)'.format(
        datetime.now() - start_all,
        len(transcripts), aligned_t_loc, os.path.getsize(
            aligned_t_loc) / 1024 / 1024
    ))
