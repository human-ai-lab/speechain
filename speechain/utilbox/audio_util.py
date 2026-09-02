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
            Device the returned resampler must be on. Also applied on a cache hit, so a
            cache shared across devices (e.g. a model moved from CPU to GPU between calls)
            always gets a resampler on the currently requested device.

    Returns:
        torchaudio.transforms.Resample: The cached (or newly created) resampler.
    """
    key = (orig_freq, new_freq)
    if key not in cache:
        cache[key] = torchaudio.transforms.Resample(
            orig_freq=orig_freq, new_freq=new_freq
        )
    if device is not None:
        cache[key] = cache[key].to(device)
    return cache[key]
