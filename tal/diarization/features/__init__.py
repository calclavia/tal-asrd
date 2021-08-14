'''
import os
import subprocess
from subprocess import DEVNULL
from bs4 import BeautifulSoup
import wave
from librosa.core import get_duration

def strip_html(text):
    return BeautifulSoup(text.strip(), 'lxml').get_text()

def get_audio_info_wav(audio_loc):
    """
    Get audio information for a WAV file
    Args:
        audio_loc (str): Target audio file path
    Returns:
        str: Codec type
        int: Number of channels
        int: Size of file in bytes
        float: Duration of file in seconds
        int: Sample rate
        int: Bit rate
    """
    # We don't need to read the header for this information
    codec = 'wav'
    size = os.path.getsize(audio_loc)
    duration = get_duration(filename=audio_loc)
    # Information from the WAV header
    with wave.open(audio_loc, 'rb') as rf:
        channels = rf.getnchannels()
        sample_rate = rf.getframerate()
        frame_width = rf.getsampwidth()
    # https://stackoverflow.com/questions/33747728/how-can-i-get-the-same-bitrate-of-input-and-output-file-in-pydub
    bit_rate = sample_rate * frame_width * 8 * channels
    return codec, channels, size, duration, sample_rate, bit_rate


ep_name = 'ep-548'
wav_name = '{}.wav'.format(ep_name)
temp_name = '{}-TEMP.wav'.format(ep_name)
transcript_name = '{}.jsonl'.format(ep_name)

"""
from wildspeech.utils.audio import get_audio_info, TARGET_BIT_RATE, TARGET_SAMPLE_RATE, convert_audio
codec, channels, size, duration, sample_rate, bit_rate = \
    get_audio_info(wav_name, audio_fmt='wav')
"""

TARGET_SAMPLE_RATE = 16000
TARGET_BIT_RATE = 16000

def convert_audio(in_file, out_file,
                  bit_rate=TARGET_BIT_RATE, sample_rate=TARGET_SAMPLE_RATE,
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
    if temp_loc:
        os.remove(temp_loc)


import torchaudio
codec, channels, size, duration, sample_rate, bit_rate = get_audio_info_wav(wav_name)
if sample_rate != TARGET_SAMPLE_RATE or bit_rate != TARGET_BIT_RATE:
    print('CONVERTING from BR {:,} & SR {:,}'.format(
        bit_rate, sample_rate
    ))
    os.rename(wav_name, temp_name)
    convert_audio(
        temp_name, wav_name
    )

# Parse utterance-by-utterance
import json
with open(transcript_name, 'r+') as rf:
    for u in rf.readilnes():
        # Load one utterance
        u_dict = json.loads(u)
        # 
'''