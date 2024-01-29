# Copyright 2024 Takaaki Saeki
# MIT LICENSE (https://opensource.org/license/mit/)

from typing import Dict, List, Tuple
import numpy as np
import librosa
from pypesq import pesq

class PESQ:

    def __init__(self, sr=16000):
        """
        Args:
            sr (int): Sampling rate.
        """
        self.sr = sr
        self.tar_fs = 16000
    
    def score(self, gt_wav, gen_wav):
        """
        Args:
            gt_wav (np.ndarray): Ground truth waveform (T,).
            gen_wav (np.ndarray): Generated waveform (T,).
        Returns:
            float: PESQ score.
        """
        if self.sr != self.tar_fs:
            gt_wav = librosa.resample(gt_wav.astype(np.float), self.sr, self.tar_fs)
        if self.sr != self.tar_fs:
            gen_wav = librosa.resample(gen_wav.astype(np.float), self.sr, self.tar_fs)

        score = pesq(gt_wav, gen_wav, self.tar_fs)
        return score