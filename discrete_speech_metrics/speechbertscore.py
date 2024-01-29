# Copyright 2024 Takaaki Saeki
# MIT LICENSE (https://opensource.org/license/mit/)

import torchaudio
import torch
from transformers import HubertModel, Wav2Vec2Model, WavLMModel


def bert_score(v_generated, v_reference):
    """
    Args:
        v_generated (torch.Tensor): Generated feature tensor (T, D).
        v_reference (torch.Tensor): Reference feature tensor (T, D).
    Returns:
        float: Precision.
        float: Recall.
        float: F1 score.
    """
    # Calculate cosine similarity
    sim_matrix = torch.matmul(v_generated, v_reference.T) / (torch.norm(v_generated, dim=1, keepdim=True) * torch.norm(v_reference, dim=1).unsqueeze(0))

    # Calculate precision and recall
    precision = torch.max(sim_matrix, dim=1)[0].mean().item()
    recall = torch.max(sim_matrix, dim=0)[0].mean().item()

    # Calculate F1 score
    f1_score = 2 * precision * recall / (precision + recall)

    return precision, recall, f1_score


class SpeechBERTScore:

    def __init__(self, sr=16000, model_type="hubert-base", layer=None, use_gpu=True):
        """
        Args:
            sr (int): Sampling rate.
            model_type (str): Model type. Select from "hubert-base", "hubert-large", "wav2vec2-base", "wav2vec2-large", "wavlm-base", "wavlm-base-plus", "wavlm-large".
            layer (int): Layer number to extract features. If None, the last layer is used.
            use_gpu (bool): Whether to use GPU.
        """
        if model_type == "hubert-base":
            self.model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        elif model_type == "hubert-large":
            self.model = HubertModel.from_pretrained("facebook/hubert-large-ll60k")
        elif model_type == "wav2vec2-base":
            self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        elif model_type == "wav2vec2-large":
            self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large")
        elif model_type == "wavlm-base":
            self.model = WavLMModel.from_pretrained("microsoft/wavlm-base")
        elif model_type == "wavlm-base-plus":
            self.model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
        elif model_type == "wavlm-large":
            self.model = WavLMModel.from_pretrained("microsoft/wavlm-large")
        else:
            raise ValueError(f"Not found the setting for {model_type}.")
        self.model.eval()
        if use_gpu and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.model.to(self.device)
        self.layer = layer
        self.sr = sr
        self.resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000).to(self.device)

    def process_feats(self, audio):
        """
        Args:
            audio (torch.Tensor): Audio waveform tensor (1, T).
        """
        if self.layer == None:
            feats = self.model(audio).last_hidden_state
        else:
            feats_hiddens = self.model(audio, output_hidden_states=True).hidden_states
            feats = feats_hiddens[self.layer]
        return feats
    
    def score(self, gt_wav, gen_wav):
        """
        Args:
            gt_wav (np.ndarray): Ground truth waveform (T,).
            gen_wav (np.ndarray): Generated waveform (T,).
        Returns:
            float: Precision.
            float: Recall.
            float: F1 score.
        """
        gt_wav = torch.from_numpy(gt_wav).unsqueeze(0).to(self.device).float()
        gen_wav = torch.from_numpy(gen_wav).unsqueeze(0).to(self.device).float()

        if self.sr != 16000:
            gt_wav = self.resampler(gt_wav)
            gen_wav = self.resampler(gen_wav)
        
        v_ref = self.process_feats(gt_wav)
        v_gen = self.process_feats(gen_wav)
        precision, recall, f1_score = bert_score(v_gen.squeeze(0), v_ref.squeeze(0))

        return precision, recall, f1_score