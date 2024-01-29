# Copyright 2021 Wen-Chin Huang and Tomoki Hayashi
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE/asr1/pyscripts/utils/evaluate_f0.py
# Modifications made in 2024 by Takaaki Saeki are licensed under the MIT License (https://opensource.org/license/mit/)

from typing import Tuple

from fastdtw import fastdtw
from scipy import spatial
import numpy as np
import pysptk
import pyworld as pw
from fastdtw import fastdtw
from scipy import spatial


def world_extract(
    x: np.ndarray,
    fs: int,
    f0min: int = 40,
    f0max: int = 800,
    n_fft: int = 512,
    n_shift: int = 256,
    mcep_dim: int = 25,
    mcep_alpha: float = 0.41,
) -> np.ndarray:
    """
    Extract World-based acoustic features.

    Args:
        x (ndarray): 1D waveform array.
        fs (int): Minimum f0 value (default=40).
        f0 (int): Maximum f0 value (default=800).
        n_shift (int): Shift length in point (default=256).
        n_fft (int): FFT length in point (default=512).
        n_shift (int): Shift length in point (default=256).
        mcep_dim (int): Dimension of mel-cepstrum (default=25).
        mcep_alpha (float): All pass filter coefficient (default=0.41).

    Returns:
        ndarray: Mel-cepstrum with the size (N, n_fft).
        ndarray: F0 sequence (N,).

    """
    # extract features
    x = x.astype(np.float64)
    f0, time_axis = pw.harvest(
        x,
        fs,
        f0_floor=f0min,
        f0_ceil=f0max,
        frame_period=n_shift / fs * 1000,
    )
    sp = pw.cheaptrick(x, f0, time_axis, fs, fft_size=n_fft)
    if mcep_dim is None or mcep_alpha is None:
        mcep_dim, mcep_alpha = _get_best_mcep_params(fs)
    mcep = pysptk.sp2mc(sp, mcep_dim, mcep_alpha)

    return mcep, f0


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


class LogF0RMSE:

    def __init__(
        self,
        f0min: int = 40,
        f0max: int = 800,
        n_fft: int = 512,
        n_shift: int = 256,
        mcep_dim: int = 25,
        mcep_alpha: float = 0.41):
        """
        Args:
            sr (int): Sampling rate.
            f0min (int): Minimum f0 value (default=40).
            f0max (int): Maximum f0 value (default=800).
            n_fft (int): FFT length in point (default=512).
            n_shift (int): Shift length in point (default=256).
            mcep_dim (int): Dimension of mel-cepstrum (default=25).
            mcep_alpha (float): All pass filter coefficient (default=0.41).
        """
        self.f0min = f0min
        self.f0max = f0max
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
            float: Log F0 RMSE.
        """
        gen_mcep, gen_f0 = world_extract(
            x=gen_wav,
            fs=self.sr,
            f0min=self.f0min,
            f0max=self.f0max,
            n_fft=self.n_fft,
            n_shift=self.n_shift,
            mcep_dim=self.mcep_dim,
            mcep_alpha=self.mcep_alpha,
        )
        gt_mcep, gt_f0 = world_extract(
            x=gt_wav,
            fs=self.sr,
            f0min=self.f0min,
            f0max=self.f0max,
            n_fft=self.n_fft,
            n_shift=self.n_shift,
            mcep_dim=self.mcep_dim,
            mcep_alpha=self.mcep_alpha,
        )

        # DTW
        _, path = fastdtw(gen_mcep, gt_mcep, dist=spatial.distance.euclidean)
        twf = np.array(path).T
        gen_f0_dtw = gen_f0[twf[0]]
        gt_f0_dtw = gt_f0[twf[1]]

        # Get voiced part
        nonzero_idxs = np.where((gen_f0_dtw != 0) & (gt_f0_dtw != 0))[0]
        gen_f0_dtw_voiced = np.log(gen_f0_dtw[nonzero_idxs])
        gt_f0_dtw_voiced = np.log(gt_f0_dtw[nonzero_idxs])

        # log F0 RMSE
        log_f0_rmse = np.sqrt(np.mean((gen_f0_dtw_voiced - gt_f0_dtw_voiced) ** 2))
        return log_f0_rmse
