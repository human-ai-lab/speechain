"""
Author: Heli Qi
Affiliation: NAIST
Date: 2022.12
"""

import math

import numpy as np
import pyworld
import torch
import torchaudio
from scipy import signal
from scipy.fftpack import dct
from scipy.interpolate import interp1d

from speechain.utilbox.tensor_util import to_native


def preemphasize_wav(wav: np.ndarray, coeff: float) -> np.ndarray:
    return signal.lfilter([1, -coeff], [1], wav)


def feat_derivation(feat: np.ndarray, delta_order: int, delta_N: int) -> np.ndarray:
    comb_feat = [feat]
    if delta_order >= 1:
        delta_feat = signal.savgol_filter(
            feat, window_length=2 * delta_N + 1, polyorder=1, deriv=1, mode="interp"
        )
        comb_feat.append(delta_feat)

        if delta_order >= 2:
            delta2_feat = signal.savgol_filter(
                delta_feat,
                window_length=2 * delta_N + 1,
                polyorder=2,
                deriv=2,
                mode="interp",
            )
            comb_feat.append(delta2_feat)

    return np.hstack(comb_feat)


def compute_stft(
    wav: np.ndarray,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: str = "hann",
    center: bool = True,
) -> np.ndarray:
    """A drop-in replacement for librosa.stft() implemented with numpy and scipy, so
    that speechain doesn't have to depend on librosa. It assumes zero-padded
    centering (center=True) and produces bit-identical results to librosa.stft() for
    the same arguments.

    Args:
        wav: (n_sample,)
            The single-channel waveform to be processed.
        n_fft: int
            The number of Fourier points.
        hop_length: int
            The number of time steps between two adjacent frames.
        win_length: int
            The window length used to extract each frame.
        window: str
            The window function used to extract each frame. Has to be a name
            recognized by scipy.signal.get_window().
        center: bool
            Whether to pad the waveform so that the t-th frame is centered at
            wav[t * hop_length].

    Returns:
        The complex-valued STFT results in the shape (1 + n_fft // 2, n_frame).
    """
    fft_window = signal.get_window(window, win_length, fftbins=True)
    if win_length < n_fft:
        pad = n_fft - win_length
        l_pad = pad // 2
        fft_window = np.pad(fft_window, (l_pad, pad - l_pad))

    if center:
        wav = np.pad(wav, n_fft // 2, mode="constant")

    n_frames = 1 + (len(wav) - n_fft) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        wav,
        shape=(n_fft, n_frames),
        strides=(wav.strides[0], wav.strides[0] * hop_length),
    )
    return np.fft.rfft(fft_window[:, None] * frames, axis=0)


def mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int,
    fmin: float = 0.0,
    fmax: float = None,
    htk: bool = False,
    norm: str or None = "slaney",
) -> np.ndarray:
    """A drop-in replacement for librosa.filters.mel() based on
    torchaudio.functional.melscale_fbanks(), which speechain already depends on for
    its main mel-spectrogram frontend
    (speechain/module/frontend/linear2mel.py).

    Args:
        sr: int
            The sampling rate of the incoming signal.
        n_fft: int
            The number of Fourier points used to obtain the linear spectrogram.
        n_mels: int
            The number of filters in the mel-fbank.
        fmin: float
            The minimal frequency for the mel-fbank.
        fmax: float
            The maximal frequency for the mel-fbank. sr / 2 is used if not given.
        htk: bool
            Whether to use the HTK formula instead of the Slaney one for the mel scale.
        norm: str or None
            Whether to perform Slaney-style area normalization on the mel-fbank filters.

    Returns:
        The mel-fbank matrix in the shape (n_mels, 1 + n_fft // 2).
    """
    fmax = sr / 2 if fmax is None else fmax
    mel_basis = torchaudio.functional.melscale_fbanks(
        n_freqs=n_fft // 2 + 1,
        f_min=fmin,
        f_max=fmax,
        n_mels=n_mels,
        sample_rate=sr,
        norm=norm,
        mel_scale="htk" if htk else "slaney",
    )
    return mel_basis.numpy().T


def convert_wav_to_stft(
    wav: np.ndarray,
    hop_length: int or float,
    win_length: int or float,
    sr: int = 16000,
    n_fft: int = None,
    preemphasis: float = None,
    pre_stft_norm: str = None,
    window: str = "hann",
    center: bool = True,
    mag_spec: bool = False,
    clamp: float = 1e-10,
    logging: bool = False,
    log_base: float or None = None,
) -> np.ndarray:
    """For the details about the arguments and returns, please refer to
    ${SPEECHAIN_ROOT}/speechain/module/frontend/speech2linear.py."""
    # if hop_length and win_length are given in the unit of seconds, turn them into the corresponding time steps
    hop_length = int(hop_length * sr) if isinstance(hop_length, float) else hop_length
    win_length = int(win_length * sr) if isinstance(win_length, float) else win_length

    # if n_fft is not given, it will be initialized to the window length
    if n_fft is None:
        n_fft = win_length

    # --- 1. Pre-emphasis --- #
    if preemphasis is not None:
        wav = preemphasize_wav(wav, preemphasis)

    # --- 2. Waveform Pre-Normalization --- #
    # normalization for audio signals before STFT
    if pre_stft_norm is not None:
        if pre_stft_norm == "mean_std":
            wav = (wav - wav.mean(axis=0)) / wav.std(axis=0)
        elif pre_stft_norm == "min_max":
            wav_min, wav_max = np.amin(wav, axis=1, keepdims=True), np.amax(
                wav, axis=1, keepdims=True
            )
            wav = (wav - wav_min) / (wav_max - wav_min) * 2 - 1
        else:
            raise ValueError

    # --- 3. STFT Processing --- #
    stft_results = compute_stft(
        wav.squeeze(-1) if len(wav.shape) == 2 else wav,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
    )
    linear_spec = np.abs(stft_results) ** 2 + np.angle(stft_results) ** 2

    # --- 4. STFT Post-Processing --- #
    # convert the energy spectrogram to the magnitude spectrogram if specified
    if mag_spec:
        linear_spec = np.sqrt(linear_spec)

    # take the logarithm operation
    if logging:
        # pre-log clamping for numerical stability
        linear_spec = np.maximum(linear_spec, clamp)
        linear_spec = np.log(linear_spec)
        if log_base is not None:
            linear_spec /= math.log(log_base)

    return linear_spec.transpose(1, 0)


def convert_wav_to_logmel(
    wav: np.ndarray,
    n_mels: int,
    hop_length: int or float,
    win_length: int or float,
    sr: int = 16000,
    n_fft: int = None,
    preemphasis: float = None,
    pre_stft_norm: str = None,
    window: str = "hann",
    center: bool = True,
    mag_spec: bool = False,
    fmin: float = 0.0,
    fmax: float = None,
    clamp: float = 1e-10,
    logging: bool = True,
    log_base: float or None = 10.0,
    htk: bool = False,
    norm: str or None = "slaney",
    delta_order: int = 0,
    delta_N: int = 2,
) -> np.ndarray:
    """For the details about the arguments and returns, please refer to
    ${SPEECHAIN_ROOT}/speechain/module/frontend/speech2mel.py."""
    # if hop_length and win_length are given in the unit of seconds, turn them into the corresponding time steps
    hop_length = int(hop_length * sr) if isinstance(hop_length, float) else hop_length
    win_length = int(win_length * sr) if isinstance(win_length, float) else win_length

    # if n_fft is not given, it will be initialized to the window length
    if n_fft is None:
        n_fft = win_length

    # --- 1. Waveform -> Linear Spectrogram --- #
    linear_spec = convert_wav_to_stft(
        wav=wav,
        hop_length=hop_length,
        win_length=win_length,
        sr=sr,
        n_fft=n_fft,
        preemphasis=preemphasis,
        pre_stft_norm=pre_stft_norm,
        window=window,
        center=center,
        mag_spec=mag_spec,
        clamp=clamp,
    )

    # --- 2. Linear Spectrogram -> Mel Spectrogram --- #
    mel_basis = mel_filterbank(
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        htk=htk,
        norm=norm,
    )
    mel_spec = mel_basis @ linear_spec.transpose(1, 0)
    # take the logarithm operation
    if logging:
        # pre-log clamping for numerical stability
        mel_spec = np.maximum(mel_spec, clamp)

        # go through the logarithm operation for the mel-fbanks
        mel_spec = np.log(mel_spec)
        if log_base is not None:
            mel_spec /= math.log(log_base)

    # --- 3. Log-Mel Spectrogram -> Log-Mel Spectrogram + Deltas --- #
    return feat_derivation(mel_spec, delta_order, delta_N).transpose(1, 0)


def convert_wav_to_mfcc(
    wav: np.ndarray,
    hop_length: int or float,
    win_length: int or float,
    num_ceps: int = None,
    n_mfcc: int = 20,
    sr: int = 16000,
    n_fft: int = None,
    n_mels: int = 80,
    preemphasis: float = None,
    pre_stft_norm: str = None,
    window: str = "hann",
    center: bool = True,
    fmin: float = 0.0,
    fmax: float = None,
    clamp: float = 1e-10,
    logging: bool = True,
    log_base: float or None = 10.0,
    htk: bool = False,
    norm: str or None = "slaney",
    delta_order: int = 0,
    delta_N: int = 2,
) -> np.ndarray:
    """For the details about the arguments and returns, please refer to
    ${SPEECHAIN_ROOT}/speechain/utilbox/feat_util.convert_wav_to_logmel() and
    scipy.fftpack.dct."""
    # if hop_length and win_length are given in the unit of seconds, turn them into the corresponding time steps
    hop_length = int(hop_length * sr) if isinstance(hop_length, float) else hop_length
    win_length = int(win_length * sr) if isinstance(win_length, float) else win_length

    # if n_fft is not given, it will be initialized to the window length
    if n_fft is None:
        n_fft = win_length

    # update n_mfcc if num_ceps is too large
    if num_ceps is not None and num_ceps > n_mfcc - 1:
        n_mfcc = num_ceps + 1

    # --- 1. Waveform -> Log-Mel Power Spectrogram --- #
    # log-mel power spectrogram is returned for MFCC calculation
    mel_spec = convert_wav_to_logmel(
        wav=wav,
        n_mels=n_mels,
        hop_length=hop_length,
        win_length=win_length,
        sr=sr,
        n_fft=n_fft,
        preemphasis=preemphasis,
        pre_stft_norm=pre_stft_norm,
        window=window,
        center=center,
        mag_spec=False,
        fmin=fmin,
        fmax=fmax,
        clamp=clamp,
        logging=logging,
        log_base=log_base,
        htk=htk,
        norm=norm,
    )

    # --- 2. Log-Mel Power Spectrogram -> MFCC --- #
    # remove the first mel cepstral coefficient
    # mfcc = dct(mel_spec.transpose(1, 0), axis=-2, type=2, norm="ortho")[1:n_mfcc, :]
    mfcc = dct(mel_spec.transpose(1, 0), axis=-2, type=2, norm="ortho")[:n_mfcc, :]
    if num_ceps is not None:
        mfcc = mfcc[:num_ceps, :]

    # --- 3. MFCC -> MFCC + Deltas --- #
    return feat_derivation(mfcc, delta_order, delta_N).transpose(1, 0)


def convert_wav_to_pitch(
    wav: np.ndarray or torch.Tensor,
    hop_length: int or float = 256,
    sr: int = 22050,
    f0min: int = 80,
    f0max: int = 400,
    continuous_f0: bool = True,
    return_tensor: bool = False,
) -> np.ndarray or torch.Tensor:
    """The function that converts a waveform to a pitch contour by dio & stonemask of
    pyworld.

    Args:
        wav: (n_sample, 1) or (n_sample,)
            The waveform to be processed.
        hop_length: int = 256
                The value of the argument 'hop_length' given to pyworld.dio()
        sr: int = 22050
            The value of the argument 'fs' given to pyworld.dio()
        f0min: int = 80
            The value of the argument 'f0min' given to pyworld.dio()
        f0max: int = 400
            The value of the argument 'f0max' given to pyworld.dio()
        continuous_f0: bool = True
            Whether to make the calculated pitch values continuous over time.
        return_tensor: bool
            Whether to return the pitch in torch.Tensor. If False, np.ndarray will be returned.

    Returns:
        f0: (n_frame,)
    """
    if isinstance(hop_length, float):
        hop_length = int(hop_length * sr)

    # datatype checking
    if isinstance(wav, torch.Tensor):
        wav = to_native(wav, tgt="numpy").astype(np.float64)
    elif isinstance(wav, np.ndarray):
        wav = wav.astype(np.float64)
    else:
        raise TypeError(
            f"wav should be either a torch.Tensor or a np.ndarray, but got type(wav)={type(wav)}!"
        )

    # dimension checking
    if wav.shape[-1] == 1:
        wav = wav.squeeze(-1)
    if len(wav.shape) > 2:
        raise RuntimeError(
            "convert_wav_to_pitch doesn't support batch_level pitch extraction!"
        )

    f0, timeaxis = pyworld.dio(
        wav, sr, f0_floor=f0min, f0_ceil=f0max, frame_period=1000 * hop_length / sr
    )
    f0 = pyworld.stonemask(wav, f0, timeaxis, sr)

    # borrowed from https://github.com/espnet/espnet/blob/master/espnet2/tts/feats_extract/dio.py#L125
    if continuous_f0:
        # padding start and end of f0 sequence
        start_f0, end_f0 = f0[f0 != 0][0], f0[f0 != 0][-1]
        start_idx, end_idx = (
            np.where(f0 == start_f0)[0][0],
            np.where(f0 == end_f0)[0][-1],
        )
        f0[:start_idx], f0[end_idx:] = start_f0, end_f0

        # get non-zero frame index
        nonzero_idxs = np.where(f0 != 0)[0]

        # perform linear interpolation
        interp_fn = interp1d(
            nonzero_idxs,
            f0[nonzero_idxs],
            bounds_error=False,
            fill_value=(start_f0, end_f0),
        )
        f0 = interp_fn(np.arange(0, f0.shape[0]))

    if return_tensor:
        f0 = torch.tensor(f0, dtype=torch.float32)
    else:
        f0 = f0.astype(np.float32)
    return f0
