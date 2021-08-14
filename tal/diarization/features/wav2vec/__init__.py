"""
From:
https://github.com/pytorch/fairseq/tree/master/examples/wav2vec
"""
import os
import wget
import torch
import subprocess

from datetime import datetime
from wildspeech import count_parameters
from fairseq.models.wav2vec import Wav2VecModel

# Pretrained wav2vec on Librispeech, from `fairseq`
WAV2VEC_PRETRAINED_URI = 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt'
WAV2VEC_FILENAME = WAV2VEC_PRETRAINED_URI.split('/')[-1]

def download_model(loc, url, overwrite=False):
    """
    Download pretrained wav2vec
    
    Args:
        loc (str): Save location for model
        url (str): Download URI
        overwrite (bool, optional): If true, re-downloads the model. Only use in interactive mode.
    """
    if overwrite or not os.path.exists(loc):
        # Download with wget
        subprocess.call([
            'wget',     # Wget utility
            '-O',       # Output file flag
            loc,        # Download to here
            url,        # Source URL
        ], shell=False)

    print('Downloaded to {} ({:,.3f} MB on disk)'.format(
        loc,
        os.path.getsize(loc) / 1024 / 1024
    ))

def get_trained_wav2vec(cache_path):
    """
    Retrieve a trained Wav2Vec model from local cache

    Args:
        cache_path (str): Cache directory

    Returns:
        Wav2VecModel: Trained model
    """
    start = datetime.now()
    model_loc = os.path.join(cache_path, WAV2VEC_FILENAME)

    # Download the model, if it hasn't already been downloaded
    os.makedirs(cache_path, exist_ok=True)
    download_model(model_loc, WAV2VEC_PRETRAINED_URI)

    # Load model
    cp = torch.load(model_loc)
    model = Wav2VecModel.build_model(cp['args'], task=None)
    model.load_state_dict(cp['model'])
    print('{} - Loaded wav2vec model from {} with {:,} parameters'.format(
        datetime.now() - start, model_loc, count_parameters(model)
    ))

    return model
