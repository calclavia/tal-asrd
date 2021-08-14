"""
Audio conversion utilities, mostly using ffmpeg & sph2pipe
"""
import os
import wave
import subprocess

from subprocess import DEVNULL
from mutagen import File
from librosa.core import get_duration

TARGET_SAMPLE_RATE = 16000
TARGET_BIT_RATE = 16000

def convert_sphere(sph2pipe_bin, in_file, out_file):
    """
    Converts a .sph file into a normal output format

    Args:
        sph2pipe_bin (str): Name with which to invoke sph2pipe utility
        in_file (str): Input file location (raw)
        out_file (str): Output file location (converted)
    """
    out_format = out_file.split('.')[-1]
    subprocess.call(
        [
            sph2pipe_bin,
            '-p',  # force conversion to 16-bit linear pcm
            '-f',  # File format
            out_format,  # Output format (usually wav)
            in_file,  # Input file
            out_file,  # Output file
        ],
        shell=False,
        stdout=DEVNULL,
        stderr=DEVNULL)

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

    # If it's a sphere-encoded file, need to decompress first with sph2pipe
    if in_format.startswith(('sph', 'wv1', 'wv2', 'sphere')):
        temp_loc = '{}_TMP.wav'.format(out_stub)
        convert_sphere(sph2pipe_bin, in_file, temp_loc)

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

def get_audio_info(audio_loc, audio_fmt=None):
    """
    Get audio information for an audio file

    Args:
        audio_loc (str): Target audio file path
        audio_fmt (str): If 'wav', process differently

    Returns:
        str: Codec type
        int: Number of channels
        int: Size of file in bytes
        float: Duration of file in seconds
        int: Sample rate
        int: Bit rate
    """
    # Mutagen doesn't handle WAV format
    if audio_fmt == 'wav':
        return get_audio_info_wav(audio_loc)

    # Mutagen file info w/ type inference
    file_info = File(audio_loc).info

    # Size in bytes
    size = os.path.getsize(audio_loc)

    # Audio codec
    if hasattr(file_info, 'codec'):
        codec = file_info.codec
    elif hasattr(file_info, 'encoder_info'):
        codec = file_info.encoder_info
    else:
        codec = file_info.__class__.__name__

    channels = file_info.channels  # Number of channels - usually 1/mono, 2/stereo
    duration = file_info.length  # Duration in seconds
    sample_rate = file_info.sample_rate  # Sample rate
    bit_rate = file_info.bitrate  # Bit rate

    return codec, channels, size, duration, sample_rate, bit_rate

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
