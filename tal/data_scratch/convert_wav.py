import os
import sys
import multiprocessing
import torchaudio
import subprocess
from subprocess import DEVNULL
from tqdm import tqdm
from datetime import datetime


def convert_wav(in_file):
    convert_audio(
        in_file=in_file,
        out_file=in_file.replace('.mp3', '.wav'),
        bit_rate=16,
        sample_rate=16000,
        n_channels=1,
        overwrite=True
    )


def convert_audio(in_file, out_file,
                  bit_rate=16, sample_rate=16000,
                  n_channels=1, sph2pipe_bin='sph2pipe', overwrite=False):
    """
    Converts an audio file to consistent specifications

    Args:
        in_file (str): Input file location (raw)
        out_file (str): Output file location (converted)
        bit_rate (int): Sample rate * bits / sample
        sample_rate (int): Target # samples / unit time
        n_channels (int): Number of channels to keep. Default 1 (mono).
        sph2pipe_bin (str): Name with which to invoke sph2pipe utility
        overwrite (bool): If True, overwrites existing output file.
    """
    if not overwrite and os.path.exists(out_file):
        return

    _, in_format = in_file.rsplit('.', 1)
    out_stub, _ = out_file.rsplit('.', 1)
    temp_loc = None

    # If it's a sphere-encoded file, need to decompress first with sph2pipe
    if in_format.startswith(('sph', 'wv1', 'wv2', 'sphere')):
        raise Exception('Sphere conversion not supported yet~')
        # temp_loc = '{}_TMP.wav'.format(out_stub)
        # convert_sphere(sph2pipe_bin, in_file, temp_loc)

    # Convert audio with ffmpeg
    subprocess.call(
        [
            'ffmpeg',
            '-y',  # Yes, overwrite
            '-i',
            temp_loc or in_file,  # Input file
            '-ac',
            str(n_channels),  # Number of channels
            '-ab',
            str(bit_rate),  # Bit rate
            '-ar',
            str(sample_rate),  # Sample rate
            '-preset',
            'veryfast',  # Faster encoding, larger file (no effect on WAV)
            out_file,  # Output file
        ],
        shell=False,
        stdout=DEVNULL,
        stderr=DEVNULL)
    sys.stdout.write('.')
    sys.stdout.flush()

    if temp_loc:
        os.remove(temp_loc)


def convert_to_wav(in_file):
    start = datetime.now()
    out_file = in_file.replace('.mp3', '.wav')

    # Load audio
    audio, sr = torchaudio.load(in_file)
    print('{} - Loaded {:,} frames from {} (SR {})'.format(
        datetime.now() - start,
        audio.shape[1], in_file, sr))

    # Stereo to mono
    if audio.shape[0] != 1:
        audio = audio.cuda()
        audio = audio.mean(dim=0, keepdim=True)
        print('Converted {} to mono'.format(in_file))

    # Resample
    if sr != 16000:
        audio = audio.cuda()
        resamp = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=16000,
        )
        audio = resamp(audio).cpu()
        print('{} - Resampled {} to 16000'.format(
            datetime.now() - start, in_file))

    # Save it
    torchaudio.save(out_file, audio, sample_rate=16000)
    print('{} - Saved to {} ({:.3f} MB)'.format(
        datetime.now() - start, out_file, os.path.getsize(out_file) / 1024 / 1024
    ))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir', type=str, required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--pool', type=int, default=1)
    args = parser.parse_args()

    base_dir = args.in_dir

    print('Scanning directory')
    base_files = [
        os.path.join(base_dir, entry.name) for entry in os.scandir(base_dir)
    ]
    mp3_candidates = sorted([
        f for f in base_files if os.path.exists(f) and not os.path.exists(f.replace('.mp3', '.wav'))
    ])

    if args.end:
        mp3_candidates = mp3_candidates[args.start:args.end]
    print('{:,} candidates to convert'.format(len(mp3_candidates)))

    with multiprocessing.Pool(processes=args.pool) as pool:
        pool.map(convert_wav, mp3_candidates)
