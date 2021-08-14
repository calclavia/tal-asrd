import os
import argparse
from joblib import Parallel, delayed
from tqdm import tqdm
import torchaudio


def split_file(path, dest, filename, max_secs, ext):
    audio_file = os.path.join(path, filename)
    wav, sr = torchaudio.load(audio_file)
    chunk_size = max_secs * sr
    # We will ignore the last chunk, to avoid having different sized chunks
    num_chunks = wav.size(1) // chunk_size
    
    for i in range(num_chunks):
        save_path = os.path.join(dest, filename.replace(ext, '{}.wav').format(i))
        torchaudio.save(save_path, wav[:, i * chunk_size:(i + 1) * chunk_size], sr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chunks a directory of audio files into segments')
    parser.add_argument('input_path', type=str, help='Path to directory of audio files')
    parser.add_argument('dest', type=str, help='Path to directory of audio files')
    parser.add_argument('--max-secs', type=float, default=30, help='Maximum amount of seconds to split the audio files')
    parser.add_argument('--ext', type=str, default='wav', help='File extension to look for')
    parser.add_argument('--workers', type=int, default=32, help='Number of parallel workers')
    args = parser.parse_args()
    print(args)

    gen = (entry.name for entry in os.scandir(args.input_path) if entry.is_file() and entry.name.endswith(args.ext))

    indexes = Parallel(n_jobs=args.workers)(delayed(split_file)(args.input_path, args.dest, name, args.max_secs, args.ext) for name in tqdm(gen))
