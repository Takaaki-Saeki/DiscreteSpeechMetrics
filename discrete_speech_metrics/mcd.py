# Copyright 2020 Wen-Chin Huang and Tomoki Hayashi
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE/asr1/pyscripts/utils/evaluate_mcd.py
# Modifications made in 2024 by Takaaki Saeki are licensed under the MIT License (https://opensource.org/license/mit/)

from typing import Tuple
import numpy as np
import pysptk
from fastdtw import fastdtw
from scipy import spatial


def sptk_extract(
    x: np.ndarray,
    fs: int,
    n_fft: int = 512,
    n_shift: int = 256,
    mcep_dim: int = 25,
    mcep_alpha: float = 0.41,
    is_padding: bool = False,
) -> np.ndarray:
    """Extract SPTK-based mel-cepstrum.

    Args:
        x (ndarray): 1D waveform array.
        fs (int): Sampling rate
        n_fft (int): FFT length in point (default=512).
        n_shift (int): Shift length in point (default=256).
        mcep_dim (int): Dimension of mel-cepstrum (default=25).
        mcep_alpha (float): All pass filter coefficient (default=0.41).
        is_padding (bool): Whether to pad the end of signal (default=False).

    Returns:
        ndarray: Mel-cepstrum with the size (N, n_fft).

    """
    # perform padding
    if is_padding:
        n_pad = n_fft - (len(x) - n_fft) % n_shift
        x = np.pad(x, (0, n_pad), "reflect")

    # get number of frames
    n_frame = (len(x) - n_fft) // n_shift + 1

    # get window function
    win = pysptk.sptk.hamming(n_fft)

    # check mcep and alpha
    if mcep_dim is None or mcep_alpha is None:
        mcep_dim, mcep_alpha = _get_best_mcep_params(fs)

    # calculate spectrogram
    mcep = [
        pysptk.mcep(
            x[n_shift * i : n_shift * i + n_fft] * win,
            mcep_dim,
            mcep_alpha,
            eps=1e-6,
            etype=1,
        )
        for i in range(n_frame)
    ]

    return np.stack(mcep)


def _get_best_mcep_params(fs: int) -> Tuple[int, float]:
    if fs == 16000:
        return 23, 0.42
    elif fs == 22050:
        return 34, 0.45
    elif fs == 24000:
        return 34, 0.46
    elif fs == 44100:
        return 39, 0.53
    elif fs == 48000:
        return 39, 0.55
    else:
        raise ValueError(f"Not found the setting for {fs}.")


class MCD:

    def __init__(self, sr=16000, n_fft=1024, n_shift=256, mcep_dim=None, mcep_alpha=None):
        """
        Args:
            sr (int): Sampling rate.
            n_fft (int): FFT length in point (default=512).
            n_shift (int): Shift length in point (default=256).
            mcep_dim (int): Dimension of mel-cepstrum (default=25).
            mcep_alpha (float): All pass filter coefficient (default=0.41).
        """
        self.sr = sr
        self.n_fft = n_fft
        self.n_shift = n_shift
        self.mcep_dim = mcep_dim
        self.mcep_alpha = mcep_alpha
    
    def score(self, gt_wav, gen_wav):
        """
        Args:
            gt_wav (np.ndarray): Ground truth waveform (T,).
            gen_wav (np.ndarray): Generated waveform (T,).
        Returns:
            float: MCD value.
        """
        gen_mcep = sptk_extract(
            x=gen_wav,
            fs=self.sr,
            n_fft=self.n_fft,
            n_shift=self.n_shift,
            mcep_dim=self.mcep_dim,
            mcep_alpha=self.mcep_alpha,
        )
        gt_mcep = sptk_extract(
            x=gt_wav,
            fs=self.sr,
            n_fft=self.n_fft,
            n_shift=self.n_shift,
            mcep_dim=self.mcep_dim,
            mcep_alpha=self.mcep_alpha,
        )

        # DTW
        _, path = fastdtw(gen_mcep, gt_mcep, dist=spatial.distance.euclidean)
        twf = np.array(path).T
        gen_mcep_dtw = gen_mcep[twf[0]]
        gt_mcep_dtw = gt_mcep[twf[1]]

        # MCD
        diff2sum = np.sum((gen_mcep_dtw - gt_mcep_dtw) ** 2, 1)
        mcd = np.mean(10.0 / np.log(10.0) * np.sqrt(2 * diff2sum), 0)
        return mcd