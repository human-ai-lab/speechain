import os
from typing import Union

import torch

from speechain.module.vocoder import HiFiGAN
from speechain.utilbox.data_loading_util import parse_path_args


class SpeechBrainWrapper(object):
    """A wrapper class for the vocoder forward function of the speechbrain package.

    This wrapper is not implemented as a Module because we don't want it to be in the computational graph of a TTS model.

    Before wrapping:
        feat -> vocoder -> wav
    After wrapping:
        feat, feat_len -> SpeechBrainWrapper(vocoder) -> wav, wav_len
    """

    def __init__(self, vocoder: HiFiGAN):
        self.vocoder = vocoder

    def __call__(self, feat: torch.Tensor, feat_len: torch.Tensor):
        # feat is (batch, time, channels), need to transpose to (batch, channels, time) for HiFiGAN forward
        # Check for NaN/Inf in features - if present, this indicates a problem with the model
        if not torch.isfinite(feat).all():
            import warnings
            warnings.warn(
                "Non-finite values (NaN/Inf) detected in mel-spectrogram features! "
                "This indicates numerical instability in the TTS model. "
                "The model needs to be retrained with proper gradient clipping and loss scaling."
            )
            # Convert NaN to zero as a last resort to avoid crashing
            feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        
        wav = self.vocoder.decode_batch(feat.transpose(-2, -1))
        # wav output is (batch, time) after decode_batch
        # add channel dimension back: (batch, time) -> (batch, time, 1)
        wav = wav.unsqueeze(-1)
        # the lengths of the shorter utterances in the batch are estimated by their feature lengths
        wav_len = (feat_len * (wav.size(1) / feat.size(1))).long()
        # make sure that the redundant parts are set to silence
        for i in range(len(wav_len)):
            wav[i][wav_len[i] :] = 0
        return wav[:, : wav_len.max()], wav_len


def get_speechbrain_hifigan(
    device: Union[int, str, torch.device],
    sample_rate: int = 22050,
    use_multi_speaker: bool = True,
) -> SpeechBrainWrapper:
    assert sample_rate in [16000, 22050]

    # initialize the HiFiGAN model
    if isinstance(device, int):
        device = f"cuda:{device}" if device >= 0 else "cpu"
    elif isinstance(device, str):
        if device != "cpu":
            assert device.startswith("cuda:")

    download_dir = parse_path_args("recipes/tts/speechbrain_vocoder")

    if not use_multi_speaker:
        assert sample_rate == 22050
        # Use our local HiFiGAN implementation with auto-download
        hifi_gan = HiFiGAN.from_pretrained(
            model_name="speechbrain/tts-hifigan-ljspeech",
            cache_dir=os.path.join(download_dir, "hifigan-ljspeech"),
            device=device,
        )
    else:
        sr_mark = "16kHz" if sample_rate == 16000 else "22050Hz"
        hifi_gan = HiFiGAN.from_pretrained(
            model_name=f"speechbrain/tts-hifigan-libritts-{sr_mark}",
            cache_dir=os.path.join(download_dir, f"hifigan-libritts-{sr_mark}"),
            device=device,
        )

    return SpeechBrainWrapper(hifi_gan)
