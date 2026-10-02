"""
Author: Bagus Tris Atmaja (with Claude Code)
Affiliation: NAIST
Date: 2026.09

Additive noise simulation used by the Lombard machine speech chain
(Novitasari et al., IEEE/ACM TASLP 2022). Speech is mixed with white or
babble noise at a given speech-to-noise ratio (SNR). The SNR can be static
(one value per utterance) or dynamic (a piece-wise constant profile changing
inside the utterance), which corresponds to the static and dynamic noise
environments studied in the paper.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

from speechain.module.abs import Module
from speechain.utilbox.audio_util import get_cached_resampler
from speechain.utilbox.data_loading_util import parse_path_args, read_data_by_path
from speechain.utilbox.train_util import make_mask_from_len

# A SNR specification for one utterance is one of
#   1. None: no noise is added (clean condition)
#   2. float: static SNR in dB for the whole utterance
#   3. List[(start_ratio, snr_db)]: dynamic SNR profile. Each element gives the SNR that
#      starts at `start_ratio` (relative position in [0, 1) of the utterance).
SNRSpec = Union[None, float, int, Sequence[Tuple[float, float]]]


class NoiseMixer(Module):
    """Mix additive noise into waveforms at a target SNR.

    Supported noise types:
        1. `white`: Gaussian white noise generated on-the-fly.
        2. any other name: a noise waveform registered in `noise_files` (e.g. `babble`),
           from which a random segment is cropped for each utterance.
    """

    def module_init(
        self,
        sample_rate: int = 16000,
        noise_files: Dict[str, str] = None,
        snr_jitter: float = 0.0,
        eps: float = 1e-8,
    ):
        """
        Args:
            sample_rate: int
                The sampling rate of the input waveforms. Registered noise files are resampled to it.
            noise_files: Dict[str, str]
                Mapping from noise type names to the paths of the noise waveforms. The waveforms are
                lazily loaded at the first use.
            snr_jitter: float
                Uniform random jitter (in dB) added to the static target SNR during training.
            eps: float
                Numerical stability constant for power calculation.
        """
        self.sample_rate = sample_rate
        self.noise_files = (
            {}
            if noise_files is None
            else {k: parse_path_args(v) for k, v in noise_files.items()}
        )
        self.snr_jitter = snr_jitter
        self.eps = eps
        self.noise_cache: Dict[str, torch.Tensor] = {}
        self.resampler_cache = {}
        self.output_size = 1

    @property
    def noise_types(self) -> List[str]:
        return ["white"] + list(self.noise_files.keys())

    def _load_noise(self, noise_type: str, device: torch.device) -> torch.Tensor:
        """Load (and cache) a noise waveform as a 1d tensor on the target device."""
        if noise_type not in self.noise_cache:
            assert noise_type in self.noise_files, (
                f"Unknown noise type {noise_type}! "
                f"Please register it in noise_files or use 'white'. Known: {self.noise_types}"
            )
            noise, sr = read_data_by_path(
                self.noise_files[noise_type],
                return_sample_rate=True,
                return_tensor=True,
            )
            noise = noise.squeeze(-1).float()
            if sr is not None and sr != self.sample_rate:
                noise = get_cached_resampler(
                    self.resampler_cache, sr, self.sample_rate
                )(noise)
            self.noise_cache[noise_type] = noise
        noise = self.noise_cache[noise_type]
        if noise.device != device:
            noise = noise.to(device)
            self.noise_cache[noise_type] = noise
        return noise

    def get_noise(
        self,
        noise_type: str,
        length: int,
        device: torch.device,
        generator: torch.Generator = None,
    ) -> torch.Tensor:
        """Return a noise segment of `length` samples."""
        if noise_type == "white":
            return torch.randn(length, device=device, generator=generator)
        noise = self._load_noise(noise_type, device)
        # tile the noise if it is shorter than the requested length
        if noise.size(0) < length:
            noise = noise.repeat(math.ceil(length / noise.size(0)))
        start = int(
            torch.randint(0, noise.size(0) - length + 1, (1,), generator=generator)
        )
        return noise[start : start + length]

    @staticmethod
    def snr_profile_to_frames(
        snr_spec: SNRSpec, length: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        """Expand a SNR specification into a per-sample SNR curve (dB).

        Returns None for the clean condition.
        """
        if snr_spec is None:
            return None
        if isinstance(snr_spec, (int, float)):
            return torch.full((length,), float(snr_spec), device=device)
        # dynamic profile: piece-wise constant
        profile = sorted(
            [(float(s), float(v)) for s, v in snr_spec], key=lambda x: x[0]
        )
        assert profile[0][0] == 0.0, (
            "The first segment of a dynamic SNR profile must start at 0.0! A negative "
            "start_ratio would leave the leading part of the curve uninitialized, since "
            "curve[start:end] with a negative start writes into the tail instead."
        )
        curve = torch.empty(length, device=device)
        for i, (start_ratio, snr_db) in enumerate(profile):
            start = int(round(start_ratio * length))
            end = (
                length
                if i == len(profile) - 1
                else int(round(profile[i + 1][0] * length))
            )
            curve[start:end] = snr_db
        return curve

    @staticmethod
    def measure_snr(
        clean: torch.Tensor,
        noisy: torch.Tensor,
        wav_len: torch.Tensor = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Measure the utterance-level SNR (dB) between clean speech and its noisy version.

        Args:
            clean: (batch, wav_maxlen)
            noisy: (batch, wav_maxlen)
            wav_len: (batch,)

        Returns:
            (batch,) SNR values in dB.
        """
        clean, noisy = clean.squeeze(-1), noisy.squeeze(-1)
        mask = (
            make_mask_from_len(wav_len, return_3d=False).to(clean.device)
            if wav_len is not None
            else torch.ones_like(clean, dtype=torch.bool)
        )
        noise = (noisy - clean) * mask
        p_s = (clean * mask).pow(2).sum(dim=-1)
        p_n = noise.pow(2).sum(dim=-1)
        return 10 * torch.log10((p_s + eps) / (p_n + eps))

    def forward(
        self,
        wav: torch.Tensor,
        wav_len: torch.Tensor,
        snr: Union[SNRSpec, List[SNRSpec]],
        noise_type: Union[str, List[str]] = "white",
        generator: torch.Generator = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mix noise into a batch of waveforms.

        Args:
            wav: (batch, wav_maxlen) or (batch, wav_maxlen, 1)
                Clean waveforms (zero-padded).
            wav_len: (batch,)
                The lengths of the waveforms.
            snr: SNRSpec or List[SNRSpec]
                The target SNR of each utterance (see SNRSpec). A single specification is shared by
                the whole batch.
            noise_type: str or List[str]
                The noise type of each utterance.
            generator: torch.Generator
                Optional random generator for reproducibility.

        Returns:
            noisy: (batch, wav_maxlen) noisy waveforms with the same padding as the input.
            wav_len: (batch,) unchanged lengths.
            snr_applied: (batch,) the effective utterance-level SNR in dB
                (inf for the clean condition).
        """
        squeeze_back = wav.dim() == 3
        wav = wav.squeeze(-1) if squeeze_back else wav
        batch_size, max_len = wav.size()
        device = wav.device

        # a single shared dynamic profile (a list of (start_ratio, snr_db) pairs) is
        # structurally distinguished from a per-utterance List[SNRSpec] by its first
        # element being a numeric pair, not by comparing len(snr) to batch_size --
        # the latter breaks whenever the profile's segment count equals batch_size
        def _is_single_profile(spec) -> bool:
            return (
                isinstance(spec, (list, tuple))
                and len(spec) > 0
                and isinstance(spec[0], (list, tuple))
                and len(spec[0]) == 2
                and isinstance(spec[0][0], (int, float))
            )

        if not isinstance(snr, list) or _is_single_profile(snr):
            snr = [snr for _ in range(batch_size)]
        if isinstance(noise_type, str):
            noise_type = [noise_type for _ in range(batch_size)]
        assert len(snr) == batch_size and len(noise_type) == batch_size

        noisy = wav.clone()
        snr_applied = torch.full((batch_size,), float("inf"), device=device)
        for i in range(batch_size):
            length = int(wav_len[i])
            curve = self.snr_profile_to_frames(snr[i], length, device)
            if curve is None:
                continue
            if self.training and self.snr_jitter > 0:
                curve = (
                    curve
                    + (torch.rand(1, device=device, generator=generator) * 2 - 1)
                    * self.snr_jitter
                )

            speech = wav[i, :length]
            noise = self.get_noise(noise_type[i], length, device, generator)
            p_s = speech.pow(2).mean() + self.eps
            p_n = noise.pow(2).mean() + self.eps
            # per-sample scaling so that the local SNR follows the (possibly dynamic) curve
            scale = torch.sqrt(p_s / (p_n * torch.pow(10.0, curve / 10)))
            noisy[i, :length] = speech + scale * noise
            snr_applied[i] = 10 * torch.log10(
                p_s / ((scale * noise).pow(2).mean() + self.eps)
            )

        if squeeze_back:
            noisy = noisy.unsqueeze(-1)
        return noisy, wav_len, snr_applied

    def extra_repr(self) -> str:
        return f"sample_rate={self.sample_rate}, noise_types={self.noise_types}, snr_jitter={self.snr_jitter}"
