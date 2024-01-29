# Copyright 2024 Takaaki Saeki
# MIT LICENSE (https://opensource.org/license/mit/)

from transformers import HubertModel
import os
import pathlib
import subprocess
import torch
import torchaudio
import joblib
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu
nltk.download('punkt')


def int_array_to_chinese_unicode(arr):
    """
    Map each integer value in the array to a distinct Unicode Chinese character.
    Unicode region for Chinese characters: 4E00 - 9FFF (20992 characters)

    Args:
        arr (list): Array of integers.
    Returns:
        str: Unicode Chinese sentence.
    """
    # Base Unicode point for Chinese characters.
    base_unicode_point = 0x4E00

    # Convert each integer in the array to a Unicode Chinese character.
    unicode_sentence = ''.join(chr(base_unicode_point + val) for val in arr)

    return unicode_sentence


class ApplyKmeans(object):
    def __init__(self, km_path, device):
        """
        Args:
            km_path (str): Path to the kmeans model.
            device (str): Device to use.
        """
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np**2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        self.C = self.C.to(device)
        self.Cnorm = self.Cnorm.to(device)

    def __call__(self, x):
        """
        Args:
            x (torch.Tensor or np.ndarray): Input tensor (T, D).
        Returns:
            np.ndarray: Cluster index (T,).
        """
        if isinstance(x, torch.Tensor):
            dist = (
                x.pow(2).sum(1, keepdim=True) - 2 * torch.matmul(x, self.C) + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x**2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)


class SpeechBLEU:
    def __init__(self, sr=16000, model_type="hubert-base", vocab=200, layer=None, n_ngram=4, remove_repetition=False, use_gpu=True):
        """
        Args:
            sr (int): Sampling rate.
            model_type (str): Model type. Select from "hubert-base".
            vocab (int): Vocabulary size. Select from 50, 100, 200.
            layer (int): Layer number to extract features. If None, the last layer is used.
            n_ngram (int): N-gram size.
            remove_repetition (bool): Whether to remove token repetitions.
            use_gpu (bool): Whether to use GPU.
        """
        if use_gpu and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        if model_type == "hubert-base":
            self.model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
            self.model.eval()
            self.model.to(self.device)
        else:
            raise ValueError(f"Not found the setting for {model_type}.")
        file_path = pathlib.Path(__file__).parent.absolute()
        km_path = file_path / f"km/km{vocab}.bin"
        os.makedirs(file_path / "km", exist_ok=True)
        if not vocab in [50, 100, 200]:
            raise ValueError(f"km vocabularies other than 50, 100, 200 are not supported.")
        if not km_path.exists():
            url = f"http://sarulab.sakura.ne.jp/saeki/discrete_speech_metrics/km/km{vocab}.bin"
            subprocess.run(["wget", url, "-O", km_path])
            print(f"Downloaded file from {url} to {km_path}")
        else:
            print(f"Using a cache at {km_path}")
        self.sr = sr
        self.layer = layer
        self.apply_kmeans = ApplyKmeans(km_path, device=self.device)
        self.n_ngram = n_ngram
        self.remove_repetition = remove_repetition
        self.weights = [1. / self.n_ngram] * self.n_ngram
        self.resampler = torchaudio.transforms.Resample(orig_freq=self.sr, new_freq=16000).to(self.device)

    def decode_label(self, audio):
        """
        Args:
            audio (torch.Tensor): Audio waveform tensor (1, T).
        Returns:
            list: List of token IDs.
        """
        audio = audio.to(self.device)
        if self.sr != 16000:
            audio_16khz = self.resampler(audio)
        if self.layer == None:
            outputs = self.model(audio)
            feats = outputs.last_hidden_state
        else:
            feats_hiddens = self.model(audio_16khz, output_hidden_states=True).hidden_states
            feats = feats_hiddens[self.layer]
        km_label = self.apply_kmeans(feats[0, ...]).tolist()
        return km_label
    
    def remove_token_repetitions(self, token_sequence):
        """
        Args:
            token_sequence (list): List of tokens.
        Returns:
            list: List of tokens with repetitions removed.
        """
        unique_tokens = []
        for token in token_sequence:
            if not unique_tokens or unique_tokens[-1] != token:
                unique_tokens.append(token)
        return unique_tokens

    def calculate_bleu(self, reference, candidate):
        """
        Args:
            reference (str): Reference text.
            candidate (str): Candidate text.
        Returns:
            float: BLEU score.
        """
        score = sentence_bleu([reference], candidate, weights=self.weights)
        return score
    
    def score(self, gt_wav, gen_wav):
        """
        Args:
            gt_wav (np.ndarray): Ground truth waveform (T,).
            gen_wav (np.ndarray): Generated waveform (T,).
        Returns:
            float: BLEU score.
        """
        gt_wav = torch.from_numpy(gt_wav).unsqueeze(0).to(self.device).float()
        gen_wav = torch.from_numpy(gen_wav).unsqueeze(0).to(self.device).float()
        gt_label = self.decode_label(gt_wav)
        gen_label = self.decode_label(gen_wav)
        if self.remove_repetition:
            gt_label = self.remove_token_repetitions(gt_label)
            gen_label = self.remove_token_repetitions(gen_label)
        gt_text = int_array_to_chinese_unicode(gt_label)
        gen_text = int_array_to_chinese_unicode(gen_label)
        bleu_score = self.calculate_bleu(gt_text, gen_text)
        return bleu_score