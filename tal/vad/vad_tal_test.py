"""
git clone https://github.com/calclavia/wild-speech.git

python -u -m wildspeech.vad.vad_tal_test --in-dir ./data/tal-final/test/ --out-dir ./data/tal-test-vad/
"""
if __name__ == "__main__":
    import os
    import pickle
    import argparse
    from tqdm import tqdm
    from wildspeech.vad.webrtcvad import run_vad, write_wave

    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir')
    parser.add_argument('--out-dir')
    parser.add_argument('--aggressiveness', type=int, default=3)
    args = parser.parse_args()

    # Make output directory
    os.makedirs(args.out_dir, exist_ok=True)

    for wav_file in tqdm(os.listdir(args.in_dir)):
        if not wav_file.endswith('.wav'):
            continue

        out_stub = os.path.join(args.out_dir, wav_file.replace('.wav', ''))
        bounds, audios, n_segments, speech_duration, file_len = run_vad(
            aggressiveness=args.aggressiveness,
            file_loc=os.path.join(args.in_dir, wav_file),
            sample_rate=16000,
            output_bounds_stub=out_stub,
            overwrite=True,
        )

        wav_out_file = os.path.join(args.out_dir, wav_file)
        write_wave(
            wav_out_file,
            b''.join(audios),
            16000
        )
        print('Saved VAD audio ({:,}/{:,} s) to {} ({:.3f} MB)'.format(
            speech_duration, file_len, wav_out_file, os.path.getsize(
                wav_out_file) / 1024 / 1024
        ))
