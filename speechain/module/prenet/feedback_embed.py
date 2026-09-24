"""
Author: Bagus Tris Atmaja (with Claude Code)
Affiliation: NAIST
Date: 2026.09

Auditory feedback embedding of the Lombard machine speech chain
(Novitasari et al., IEEE/ACM TASLP 2022). The SNR embedding Z_SNR and the
ASR-loss embedding Z_ASR are combined with the TTS encoder output:

    h^e = h^e_trm + Z_SPK + Z_SNR + Z_ASR        (Eq. 8)

Both embeddings can be given at the utterance level (one vector per
utterance) or at the token level (one vector per phoneme, i.e. short-term
feedback for dynamic noise environments).
"""

import torch

from speechain.module.abs import Module


class FeedbackEmbedPrenet(Module):
    """Embed the auditory feedback (SNR + ASR loss) and add it to the encoder output."""

    def module_init(
        self,
        d_model: int,
        snr_emb_dim: int = None,
        asr_loss_hidden: int = 64,
        snr_coeff: float = 1.0,
        asr_coeff: float = 1.0,
    ):
        """
        Args:
            d_model: int
                Dimension of the TTS encoder output.
            snr_emb_dim: int
                Dimension of the SNR embedding produced by the SNR predictor. If it differs from
                d_model, a linear projection is used.
            asr_loss_hidden: int
                Hidden size of the ASR-loss embedding MLP.
            snr_coeff: float
                Default coefficient of Z_SNR (Fig. 3a of the paper).
            asr_coeff: float
                Default coefficient of Z_ASR.
        """
        self.d_model = d_model
        self.snr_coeff, self.asr_coeff = snr_coeff, asr_coeff
        snr_emb_dim = d_model if snr_emb_dim is None else snr_emb_dim
        self.snr_proj = (
            torch.nn.Linear(snr_emb_dim, d_model)
            if snr_emb_dim != d_model
            else torch.nn.Identity()
        )
        self.asr_loss_embed = torch.nn.Sequential(
            torch.nn.Linear(1, asr_loss_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(asr_loss_hidden, d_model),
        )
        self.output_size = d_model

    @staticmethod
    def _broadcast(emb: torch.Tensor, enc_text: torch.Tensor) -> torch.Tensor:
        """Broadcast an utterance-level (batch, dim) or token-level (batch, len, dim) embedding."""
        if emb.dim() == 2:
            return emb.unsqueeze(1).expand(-1, enc_text.size(1), -1)
        assert emb.dim() == 3 and emb.size(1) == enc_text.size(1), (
            f"Token-level feedback must have the same length as the encoder output, "
            f"but got {emb.shape} vs {enc_text.shape}."
        )
        return emb

    def embed_asr_loss(self, asr_loss: torch.Tensor) -> torch.Tensor:
        """(batch,) or (batch, len) ASR losses -> (batch, d_model) or (batch, len, d_model)."""
        return self.asr_loss_embed(asr_loss.float().unsqueeze(-1))

    def embed_snr(self, snr_emb: torch.Tensor) -> torch.Tensor:
        return self.snr_proj(snr_emb.float())

    def forward(
        self,
        enc_text: torch.Tensor,
        snr_emb: torch.Tensor = None,
        asr_loss: torch.Tensor = None,
        snr_coeff: float = None,
        asr_coeff: float = None,
    ) -> torch.Tensor:
        """
        Args:
            enc_text: (batch, text_maxlen, d_model)
                TTS encoder output.
            snr_emb: (batch, snr_emb_dim) or (batch, text_maxlen, snr_emb_dim)
                SNR embedding from the SNR predictor. None means no SNR feedback.
            asr_loss: (batch,) or (batch, text_maxlen)
                ASR loss of the noisy speech. None means no ASR feedback.
            snr_coeff, asr_coeff: float
                Override the default coefficients.

        Returns:
            (batch, text_maxlen, d_model) encoder output combined with the feedback.
        """
        snr_coeff = self.snr_coeff if snr_coeff is None else snr_coeff
        asr_coeff = self.asr_coeff if asr_coeff is None else asr_coeff
        if snr_emb is not None and snr_coeff != 0:
            enc_text = enc_text + snr_coeff * self._broadcast(
                self.embed_snr(snr_emb), enc_text
            )
        if asr_loss is not None and asr_coeff != 0:
            enc_text = enc_text + asr_coeff * self._broadcast(
                self.embed_asr_loss(asr_loss), enc_text
            )
        return enc_text

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, snr_coeff={self.snr_coeff}, asr_coeff={self.asr_coeff}"
