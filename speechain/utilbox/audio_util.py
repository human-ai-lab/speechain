"""
Shared helpers for on-the-fly waveform resampling.
"""

from typing import Dict, Optional, Tuple

import torch
import torchaudio


def get_cached_resampler(
    cache: Dict[Tuple[int, int], torchaudio.transforms.Resample],
    orig_freq: int,
    new_freq: int,
    device: Optional[torch.device] = None,
) -> torchaudio.transforms.Resample:
    """Return the ``torchaudio.transforms.Resample`` for (orig_freq, new_freq), creating and
    storing it in ``cache`` on first use so it isn't rebuilt for every utterance.

    Args:
        cache (Dict[Tuple[int, int], torchaudio.transforms.Resample]):
            Caller-owned dict (e.g. an instance attribute) that persists across calls.
        orig_freq (int):
            The source sampling rate.
        new_freq (int):
            The target sampling rate.
        device (torch.device, optional):
            Device to move a newly created resampler to. Ignored for a cache hit, so all
            callers sharing a cache must request a consistent device.

    Returns:
        torchaudio.transforms.Resample: The cached (or newly created) resampler.
    """
    key = (orig_freq, new_freq)
    if key not in cache:
        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_freq, new_freq=new_freq
        )
        if device is not None:
            resampler = resampler.to(device)
        cache[key] = resampler
    return cache[key]
