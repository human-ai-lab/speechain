"""
Author: Bagus Tris Atmaja (with Claude Code)
Affiliation: NAIST
Date: 2026.09

SNR recognition model of the Lombard machine speech chain
(Novitasari et al., IEEE/ACM TASLP 2022, Fig. 2c). Given a noisy speech
waveform it predicts the SNR class of the environment and produces the SNR
embedding Z_SNR fed back to the TTS. It is made of stacks of convolution +
residual blocks followed by linear layers.
"""

from typing import Dict, List

import torch
from torch.amp import autocast

from speechain.module.abs import Module
from speechain.utilbox.import_util import import_class
from speechain.utilbox.train_util import make_mask_from_len


class ResBlock1d(torch.nn.Module):
    """Two Conv1d-BatchNorm layers with a residual connection."""

    def __init__(self, channels: int, kernel_size: int, dropout: float):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = torch.nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.conv2 = torch.nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = torch.nn.BatchNorm1d(channels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.dropout(self.bn2(self.conv2(y)))
        return torch.relu(x + y)


class SNRPredictor(Module):
    """Predict the SNR class of noisy speech and produce the SNR embedding.

    Input waveforms are turned into log-mel spectrograms by a built-in frontend, normalized per
    utterance (SNR is a relative measure, so the absolute level is discarded), and encoded by
    `len(conv_dims)` stacks of [Conv1d + ReLU + ResBlock]. Frame-level embeddings are produced by a
    linear layer and averaged over the valid frames to obtain the utterance-level embedding. The
    frame-level outputs allow short-term feedback in dynamic noise environments.
    """

    def module_init(
        self,
        snr_classes: List[str],
        frontend: Dict = None,
        conv_dims: List[int] = None,
        conv_kernel: int = 3,
        res_blocks: int = 1,
        emb_dim: int = 256,
        dropout: float = 0.1,
    ):
        """
        Args:
            snr_classes: List[str]
                The names of the SNR classes, e.g. ['clean', '0', '-10'].
            frontend: Dict
                The configuration of the waveform frontend (type + conf). If not given, the input of
                forward() must be acoustic features (batch, feat_maxlen, feat_dim).
            conv_dims: List[int]
                The channel number of each convolution stack.
            conv_kernel: int
                Kernel size of the convolution layers.
            res_blocks: int
                Number of residual blocks in each stack.
            emb_dim: int
                Dimension of the SNR embedding Z_SNR.
            dropout: float
        """
        self.snr_classes = [str(c) for c in snr_classes]
        self.class2idx = {c: i for i, c in enumerate(self.snr_classes)}
        conv_dims = [128, 128, 128, 128] if conv_dims is None else conv_dims

        if frontend is not None:
            frontend_class = import_class("speechain.module." + frontend["type"])
            self.frontend = frontend_class(**frontend.get("conf", dict()))
            feat_dim = self.frontend.output_size
        else:
            assert self.input_size is not None, (
                "Please give either a frontend or input_size (the dimension of the input acoustic "
                "features) to SNRPredictor."
            )
            feat_dim = self.input_size

        layers, in_dim = [], feat_dim
        for out_dim in conv_dims:
            layers.append(
                torch.nn.Conv1d(in_dim, out_dim, conv_kernel, padding=conv_kernel // 2)
            )
            layers.append(torch.nn.BatchNorm1d(out_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))
            for _ in range(res_blocks):
                layers.append(ResBlock1d(out_dim, conv_kernel, dropout))
            in_dim = out_dim
        self.conv_stack = torch.nn.Sequential(*layers)
        self.emb_proj = torch.nn.Linear(in_dim, emb_dim)
        self.classifier = torch.nn.Linear(emb_dim, len(self.snr_classes))
        self.emb_dim = emb_dim
        self.output_size = emb_dim

    def extract_feat(self, wav: torch.Tensor, wav_len: torch.Tensor):
        """Turn waveforms into per-utterance-normalized log-mel features."""
        if hasattr(self, "frontend") and wav.size(-1) == 1:
            with autocast("cuda", enabled=False):
                # the frontend modifies the length tensor in place, so a copy is given
                feat, feat_len = self.frontend(wav.float(), wav_len.clone())[:2]
        else:
            feat, feat_len = wav, wav_len
        mask = (
            make_mask_from_len(feat_len, return_3d=False).to(feat.device).unsqueeze(-1)
        )
        count = mask.sum(dim=1).clamp(min=1)
        mean = (feat * mask).sum(dim=1, keepdim=True) / count.unsqueeze(1)
        var = ((feat - mean) ** 2 * mask).sum(dim=1, keepdim=True) / count.unsqueeze(1)
        feat = (feat - mean) / torch.sqrt(var + 1e-5) * mask
        return feat, feat_len

    def forward(
        self, wav: torch.Tensor, wav_len: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            wav: (batch, wav_maxlen, 1) waveforms or (batch, feat_maxlen, feat_dim) features.
            wav_len: (batch,)

        Returns:
            Dict with
                logits: (batch, class_num) utterance-level SNR class logits
                emb: (batch, emb_dim) utterance-level SNR embedding Z_SNR
                frame_logits: (batch, feat_maxlen, class_num)
                frame_emb: (batch, feat_maxlen, emb_dim)
                feat_len: (batch,) number of valid frames
        """
        feat, feat_len = self.extract_feat(wav, wav_len)
        hidden = self.conv_stack(feat.transpose(1, 2)).transpose(1, 2)
        frame_emb = self.emb_proj(hidden)
        mask = (
            make_mask_from_len(feat_len, return_3d=False).to(feat.device).unsqueeze(-1)
        )
        emb = (frame_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return dict(
            logits=self.classifier(emb),
            emb=emb,
            frame_logits=self.classifier(frame_emb),
            frame_emb=frame_emb,
            feat_len=feat_len,
        )

    def class_ids(self, class_names: List[str]) -> torch.Tensor:
        return torch.LongTensor([self.class2idx[str(c)] for c in class_names])

    def extra_repr(self) -> str:
        return f"snr_classes={self.snr_classes}, emb_dim={self.emb_dim}"
