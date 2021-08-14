"""
Extract features (This American Life)

# Native Wav2Vec features

pkill -f diarization
nohup python3 -u -m bernard.diarization.features.wav2vec.extract_features_tal \
-g "/root/data4/bernard-tal-splits/*/*.wav" \
-w /root/data4/wav2vec \
-o /root/data4/tal-features \
> /root/data4/tal_w2v.log &

tail -f /root/data4/tal_w2v.log

# 400ms non-overlapping windows (mean of 36 overlapping windows, skip 5)

pkill -f diarization
nohup python3 -u -m bernard.diarization.features.wav2vec.extract_features_tal \
-g "/root/data4/bernard-tal-splits/*/*.wav" \
-w /root/data4/wav2vec \
-o /root/data4/tal-features-400 \
--window 400 \
> /root/data4/tal_w2v_400.log &

tail -f /root/data4/tal_w2v_400.log

"""
if __name__ == "__main__":
    import os
    import gc
    import glob
    import json
    import time
    import torch
    import pickle
    import torchaudio
    import argparse
    import numpy as np

    from tqdm import tqdm
    from librosa.core import get_duration
    from bs4 import BeautifulSoup
    from datetime import datetime
    from collections import defaultdict

    from wildspeech import get_device, count_parameters
    from wildspeech.utils.audio import convert_audio, get_audio_info, TARGET_SAMPLE_RATE
    from wildspeech.diarization.features.wav2vec import get_trained_wav2vec, download_model

    start = datetime.now()
    USE_GPU, DEVICE = get_device()

    # Set up parser
    parser = argparse.ArgumentParser(
        description='Extract Wav2Vec features for audio')
    parser.add_argument("--audio-glob", '-g', type=str,
        default='/root/data4/bernard-tal-splits/*/*.wav', help="Audio glob")
    parser.add_argument('--wav2vec-cache', '-w', type=str, default='/root/data4/wav2vec',
        help='Where to cache the Wav2Vec model.')
    parser.add_argument('--convert', '-C', action='store_true', default=False,
        help='Convert audio to proper WAV, 16bit, 16k sample rate.')
    parser.add_argument('--out-dir', '-o', type=str, default='/root/data4/tal-features',
        help='Directory to store features for TAL')
    parser.add_argument('--overwrite', action='store_true', default=False,
        help='Overwrite checkpoints (re-generate features)')
    parser.add_argument('--window', type=int, default=None,
        help='If specified, generates features for specified non-overlapping window sizes')
    args = parser.parse_args()

    # Parse arguments
    audio_glob = args.audio_glob
    base_dir = audio_glob.split('*')[0]
    cache_dir = args.wav2vec_cache
    convert = args.convert
    out_dir = args.out_dir
    overwrite = args.overwrite
    window = args.window
    os.makedirs(out_dir, exist_ok=True)

    # Valid windows are 50 + 10k
    if window < 50 or (window - 50) % 10 != 0:
        raise Exception('Window size must be W >= 50, (W - 50) % 10 == 0!')
    elif window:
        print('\n\nNON OVERLAPPING WINDOW SIZE {}\n\n'.format(window))
        time.sleep(5)
        n_window_to_avg = int((window - 50) / 10 + 1)
        # To make it non-contiguous, we need to skip 5 windows
        # since wav2vec is size 50, stride 10
        n_window_to_ignore = 5

    # Retrieve pretrained model
    model = get_trained_wav2vec(cache_dir).to(DEVICE)
    model.eval()

    checkpoint_loc = os.path.join(out_dir, 'CHECKPOINT.pkl')
    err_loc = os.path.join(out_dir, 'ERRORS.pkl')

    # Info
    if not args.overwrite and os.path.exists(checkpoint_loc):
        with open(checkpoint_loc, 'rb') as rf:
            all_speakers, done = pickle.load(rf)
        print('{} - retrieved {:,} conversations already processed.'.format(
            datetime.now() - start, len(done)
        ))
    else:
        all_speakers = []
        done = set()
    
    if not args.overwrite and os.path.exists(err_loc):
        with open(err_loc, 'rb') as rf:
            errors = pickle.load(rf)
        print('{} - retrieved {:,} files with errors'.format(
            datetime.now() - start, len(errors)
        ))
    else:
        errors = dict()
    errors = defaultdict(list, errors)

    # Splits
    file_candidates = glob.glob(audio_glob)
    for ii, audio_path in enumerate(file_candidates):
        # If this has been processed before, skip it
        if audio_path in done:
            continue

        file_features = []
        file_speakers = []

        # Find split and corresponding transcript path
        split = audio_path.replace(base_dir, '').split(os.sep)[0]
        file_stub = audio_path.split(os.sep)[-1][:-len('.wav')]
        t_path = audio_path.replace('.wav', '.jsonl')

        # Audio info
        codec, channels, size, duration, sample_rate, bit_rate = \
            get_audio_info(audio_path, 'wav')
        
        if sample_rate != TARGET_SAMPLE_RATE or channels != 1:
            # Convert to a temporary path
            temp_path = '{}-TEMP.wav'.format(audio_path[:-len('.wav')])
            os.rename(audio_path, temp_path)

            # Convert audio file
            convert_audio(temp_path, audio_path)
            os.remove(temp_path)

        # Base information for the file
        si, ei = torchaudio.info(audio_path)
        total_duration = get_duration(filename=audio_path)
        with open(t_path, 'r+') as rf:
            utt = [json.loads(u.strip()) for u in rf.readlines()]
        
        # Load full audio
        audio_full, _ = torchaudio.load(audio_path)
        max_frames = audio_full.shape[1]
        del audio_full
        gc.collect()

        # Speaker information
        for u in tqdm(utt):
            utterance = BeautifulSoup(u['utterance'], 'lxml').get_text()
            speaker_name = u['speaker'].strip()
            relative_speaker_id = u['speaker_id']

            # Store absolute speaker ID
            if speaker_name.lower() not in all_speakers:
                speaker_id = len(all_speakers)
                all_speakers.append(speaker_name.lower())
            else:
                speaker_id = all_speakers.index(speaker_name.lower())

            # Get audio segment corresponding to this utterance
            start_s = u['utterance_start']
            end_s = total_duration if np.isnan(u['utterance_end']) else u['utterance_end']
            duration = end_s - start_s

            # Retrieve the wav2vec features for an utterance
            try:
                # Load audio segment
                offset = int(start_s * si.rate)
                duration_frames = int(duration * si.rate)
                audio, _ = torchaudio.load(
                    audio_path,
                    offset=offset,
                    num_frames=duration_frames,
                )
                audio = audio.to(DEVICE)

                # Retrieve features
                with torch.no_grad():
                    z = model.feature_extractor(audio)
                    c = model.feature_aggregator(z)
                
                # Detach from gradients, squeeze batch dim (always 1), H last
                features = c.detach().squeeze(0).permute(1, 0).cpu().numpy()

                # If we want non-overlapping windows, we average the wav2vec representations
                # across (W - 50) / 10 + 1 windows, then ignore the next 5
                if window:
                    features = np.vstack([
                        features[i:(i+n_window_to_avg), :].mean(axis=0) 
                        for i in range(0, features.shape[0], n_window_to_avg + n_window_to_ignore)
                    ])

                # Store
                file_features.append(features)
                file_speakers.extend([speaker_id] * len(features))
            except Exception as e:
                err_str = '\n\nUnable to process/load audio for {} (max frames {:,}, duration {} s)\nTrying to get offset {} s ({:,} frames) and duration {} s ({:,} frames)\nSpeaker {} ("{}"), utterance:\n\n{}!\n\n:{}'.format(
                        audio_path, max_frames, total_duration,
                        start_s, offset, duration, duration_frames,
                        speaker_id, speaker_name, utterance, e
                    )
                print(err_str)
                errors[audio_path].append(err_str)
                continue

        # Save it. Format for a single conversation w/ turn T and dim H:
        # Features: np array of size T x H
        # Cluster IDs: list of length T
        seq_loc = os.path.join(out_dir, '{}_seq.npy'.format(file_stub))
        file_features = np.concatenate(file_features, axis=0)
        np.save(seq_loc, file_features)
        print('{} - Saved for #{}, sequence w/ shape {} {} ({:,.3f} MB)'.format(
            datetime.now() - start, ii, file_features.shape, seq_loc,
            os.path.getsize(seq_loc) / 1024 / 1024
        ))
        cluster_loc = os.path.join(out_dir, '{}_cluster_id.npy'.format(file_stub))
        np.save(cluster_loc, file_speakers)
        print('{} - Saved {:,} speaker IDs to {} ({:,.3f} MB)'.format(
            datetime.now() - start, len(file_speakers), cluster_loc,
            os.path.getsize(cluster_loc) / 1024 / 1024
        ))
        
        # Save checkpoint
        done.add(audio_path)
        with open(checkpoint_loc, 'wb') as wf:
            pickle.dump([all_speakers, done], wf)
        with open(err_loc, 'wb') as wf:
            pickle.dump(errors, wf)

    # Checkpoint and proceed
    with open(checkpoint_loc, 'wb') as wf:
        pickle.dump([all_speakers, done], wf)
    with open(err_loc, 'wb') as wf:
        pickle.dump(errors, wf)
    print('{} - Retrieved all info!'.format(
        datetime.now() - start
    ))
