# Copyright 2024 Takaaki Saeki
# MIT LICENSE (https://opensource.org/license/mit/)
# Using SpeechMOS (https://github.com/tarepan/SpeechMOS) developed by @tarepan.

import torch

class UTMOS:

    def __init__(self, sr=16000, use_gpu=True):
        """
        Args:
            sr (int): Sampling rate.
            use_gpu (bool): Whether to use GPU.
        """
        self.predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
        if use_gpu and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.predictor.eval()
        self.predictor.to(self.device)
        self.sr = sr
    
    def score(self, gen_wav):
        """
        Args:
            gen_wav (np.ndarray): Generated waveform (T,).
        Returns:
            float: UTMOS score.
        """
        gen_wav = torch.from_numpy(gen_wav).unsqueeze(0).to(self.device).float()
        score = self.predictor(gen_wav, self.sr)
        return score[0].item()
