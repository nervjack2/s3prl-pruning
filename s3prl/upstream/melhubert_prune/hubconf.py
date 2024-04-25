"""
    Hubconf for Mel HuBERT.
    Author: Tzu-Quan Lin (https://github.com/nervjack2)
"""

import os
from .expert import UpstreamExpert as _UpstreamExpert

def compression_20ms_row_pruning_960hours_local(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    mean_std_npy_path = '/home/nervjack2/Desktop/SSL_FFN_Analysis/mean-std-dir/libri-960-mean-std.npy'
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, mode='row-pruning', fp=20, mean_std_npy_path=mean_std_npy_path, *args, **kwargs)

def compression_10ms_row_pruning_960hours_local(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    mean_std_npy_path = '/home/nervjack2/Desktop/SSL_FFN_Analysis/mean-std-dir/libri-960-mean-std.npy'
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, mode='row-pruning', fp=10, mean_std_npy_path=mean_std_npy_path, *args, **kwargs)