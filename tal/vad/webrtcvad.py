"""
python3.7 -m pip install webrtcvad==2.0.10
"""
from datetime import datetime
from librosa.core import get_duration
from pydub import AudioSegment
import webrtcvad
import collections
import contextlib
import sys
import os
import wave
import numpy as np
import pickle
from tqdm import tqdm
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.
    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    """Represents a "frame" of audio data."""

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.
    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False
    voiced_frames = []
    start_seg = None
    # for frame in tqdm(frames):
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                start_seg = ring_buffer[0][0].timestamp
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                end_seg = frame.timestamp + frame.duration
                triggered = False
                yield (start_seg, end_seg, b''.join([f.bytes for f in voiced_frames]))
                ring_buffer.clear()
                voiced_frames = []
    # if triggered:
        # sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    # sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield (start_seg, None, b''.join([f.bytes for f in voiced_frames]))


def run_vad(
        aggressiveness: int,
        file_loc: str,
        raw_audio: bytes = None,
        sample_rate: int = None,
        output_bounds_stub: str = None,
        output_wav_stub: str = None,
        overwrite: bool = False):
    # Short circuit for bounds
    bounds_loc = '{}-bounds.pkl'.format(output_bounds_stub) if output_bounds_stub else \
        None
    if not overwrite and bounds_loc and os.path.exists(bounds_loc):
        print('{} Exists already - skipping'.format(bounds_loc))
        return
    start = datetime.now()

    if raw_audio is not None and sample_rate is not None:
        audio = raw_audio
        sample_rate = sample_rate
    elif file_loc.endswith('.wav'):
        audio, sample_rate = read_wave(file_loc)
    elif file_loc.endswith('.mp3'):
        audio_seg = AudioSegment.from_mp3(file_loc)
        audio = audio_seg.raw_data
        sample_rate = audio_seg.frame_rate
    vad = webrtcvad.Vad(aggressiveness)
    file_len = get_duration(filename=file_loc)
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 30, 300, vad, frames)
    bounds = []
    audios = []
    n_segments = 0
    speech_duration = 0.0
    for seg_ix, (start_s, end_s, audio_seg) in enumerate(segments):
        if end_s is None or np.isnan(end_s):
            end_s = file_len
        bounds.append((start_s, end_s))
        speech_duration += (end_s - start_s)
        if output_wav_stub:
            seg_fname = '{}-{:04d}.wav'.format(output_wav_stub, seg_ix)
            write_wave(
                seg_fname,
                audio_seg,
                16000
            )
        n_segments += 1
        audios.append(audio_seg)
    print('{:.3f}/{:.3f} s speech extracted ({:.2f}%) in {}'.format(
        speech_duration, file_len, 100.0 * speech_duration / file_len,
        datetime.now() - start,
    ))
    if bounds_loc:
        with open(bounds_loc, 'wb') as wf:
            pickle.dump(bounds, wf)

    return bounds, audios, n_segments, speech_duration, file_len


def reconstruct_filename(episode: str, segment_n: int):
    return '{}-{:04d}.wav'.format(episode, segment_n)


"""
mc mirror . bernard/bernard-jobs/vad-castbox/ --exclude 'k8/*' --exclude 'out/*' --exclude '.git/*' --exclude '.vscode/*' --overwrite --remove -q

python -u -m wildspeech.vad.webrtcvad --base-dir ./data/castbox --output-dir ./data/castbox-speech --aggressiveness 3

python3 -u -m wildspeech.vad.webrtcvad --base-dir /home/shuyang/data4/cb-test --output-dir /home/shuyang/data4/cb-speech --aggressiveness 3
"""
if __name__ == "__main__":
    import argparse
    import pickle
    import os
    from multiprocessing import cpu_count
    from multiprocessing.pool import Pool

    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--aggressiveness', type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # https://stackoverflow.com/questions/57849854/os-scandir-and-multiprocessing-threadpool-works-but-multi-process-pool-doesn
    POOL = Pool(cpu_count())

    n_vad = 0
    # Work on each split separately
    for split in os.scandir(args.base_dir):
        split_dir = split.path
        if not os.path.isdir(split_dir):
            continue
        split_name = os.path.basename(split_dir)
        os.makedirs(os.path.join(args.output_dir, split_name), exist_ok=True)

        # Process each candidate
        for candidate in os.scandir(split_dir):
            file_path = candidate.path
            base_fname = os.path.basename(file_path)
            if not base_fname.endswith('.wav') and not base_fname.endswith('.mp3'):
                continue
            episode_name = base_fname.rsplit('.', 1)[0]
            stub = os.path.join(args.output_dir, split_name, episode_name)
            POOL.apply_async(run_vad, kwds={
                'aggressiveness': args.aggressiveness,
                'file_loc': file_path,
                'output_bounds_stub': stub,
                'output_wav_stub': None,
            })

    POOL.close()
    POOL.join()
    print('DONE')
