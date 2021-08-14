import os
import sys
import inspect
import torch
import numpy as np
import random
from collections.abc import Iterable
from typing import Callable


def get_device(use_cuda=True):
    cuda_available = torch.cuda.is_available()
    use_cuda = use_cuda and cuda_available

    # Prompt user to use CUDA if available
    if cuda_available and not use_cuda:
        print(
            "WARNING: You have a CUDA device, so you should probably run with --cuda"
        )

    # Set device
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    print('Device: {}'.format(device))
    if use_cuda:
        print('Using CUDA {}'.format(torch.cuda.current_device()))
    return use_cuda, device


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed, gpu=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if gpu:
        torch.cuda.manual_seed_all(seed)
    print('Set seeds to {}'.format(seed))


class SuppressPrint(object):
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def debug_log(x: object,
              msg: str = '',
              log_fxn: Callable = print,
              debug: bool = False):
    """
    12/4/2019 - Shuyang Li

    Debug logging of an object. Usage:

    from functools import partial
    debug_fxn = partial(debug_log, debug=True)
    
    Arguments:
        x (object): Some object to be logged/debugged
    
    Keyword Arguments:
        msg (str): Custom message to be spawned
        log_fxn (callable): Logging function
    """
    if not debug:
        return

    # Get caller function
    caller_fxn = inspect.currentframe().f_back.f_code.co_name
    log_str = '[{}] {}'.format(caller_fxn, type(x))

    # Get shape/length
    if hasattr(x, '__len__'):
        log_str += ' len: {},'.format(len(x))
    if hasattr(x, 'shape'):
        log_str += ' shape: {},'.format(x.shape)

    # Torch tensor
    if isinstance(x, torch.Tensor):
        # Dtype
        log_str += ' dtype {}, '.format(x.dtype)

        # Device
        log_str += ' on device "{}", '.format(x.device)

        # Device
        n_nan = torch.isnan(x).sum()
        n_zero = (x == 0).sum()
        n_pinf = (x == np.inf).sum()
        n_ninf = (x == -np.inf).sum()
        log_str += ' ({:,} NaN, {:,} 0, {:,} +inf, {:,} -inf),'.format(
            n_nan, n_zero, n_pinf, n_ninf)

    # Numpy array
    elif isinstance(x, np.ndarray):
        # Dtype
        log_str += ' dtype {}, '.format(x.dtype)

        n_none = (x == None).sum()
        n_zero = (x == 0).sum()
        n_pinf = (x == np.inf).sum()
        n_ninf = (x == -np.inf).sum()
        n_nan = 0

        # Need to loop over nditer
        for xx in np.nditer(x, ['refs_ok']):
            try:
                if np.isnan(xx):
                    n_nan += 1
            except:
                pass

        log_str += ' ({:,} NaN, {:,} 0, {:,} +inf, {:,} -inf),'.format(
            n_nan, n_zero, n_pinf, n_ninf)

    # Arbitrary iterable
    elif isinstance(x, Iterable):
        n_none = 0
        n_nan = 0
        n_pinf = 0
        n_zero = 0
        n_ninf = 0

        for xx in x:
            # None
            if xx is None:
                n_none += 1

            # Zero/Nan/+inf/-inf
            try:
                if xx == 0:
                    n_zero += 1
                if np.isnan(xx):
                    n_nan += 1
                if np.isposinf(xx):
                    n_pinf += 1
                if np.isneginf(xx):
                    n_ninf += 1
            except:
                continue

        log_str += ' ({:,} None, {:,} NaN, {:,} 0, {:,} +inf, {:,} -inf),'.format(
            n_none, n_nan, n_zero, n_pinf, n_ninf)

    log_str += ': {}'.format(msg)

    # Log it
    log_fxn(log_str)
