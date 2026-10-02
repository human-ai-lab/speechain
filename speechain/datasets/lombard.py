"""
Author: Bagus Tris Atmaja (with Claude Code)
Affiliation: NAIST
Date: 2026.09

Dataset for training the Lombard machine speech chain TTS
(Novitasari et al., IEEE/ACM TASLP 2022). Besides the regular TTS data
(target waveform, phoneme text, duration), each instance carries

    1. `feat_ref`: the normal (non-Lombard) waveform of the same sentence used to
       produce the auditory feedback (it is mixed with noise and listened to by
       the SNR predictor and the ASR).
    2. `text_asr`: the raw transcript used by the ASR to calculate the ASR loss.
    3. `snr_cond`: the name of the noise condition of the target waveform,
       e.g. 'clean', 'snr0', 'snr-10'.
"""

from typing import Any, Dict, List

import torch

from speechain.datasets.speech_text import SpeechTextDataset
from speechain.utilbox.audio_util import get_cached_resampler
from speechain.utilbox.data_loading_util import read_data_by_path

EXTRA_KEYS = ["feat_ref", "text_asr", "snr_cond"]


class LombardSpeechTextDataset(SpeechTextDataset):
    """SpeechTextDataset extended with the reference waveform, the ASR transcript and
    the noise condition of each utterance."""

    @staticmethod
    def _trim_ratios(main_data: Dict) -> (float, float):
        """Ratios of the leading and trailing silence, mirroring the silence trimming done by
        SpeechTextDataset.extract_main_data_fn() so that `feat_ref` is trimmed consistently.
        """
        text, duration = main_data.get("text", None), main_data.get("duration", None)
        if not (isinstance(text, str) and isinstance(duration, str)):
            return 0.0, 0.0
        if not (text.startswith("[") and duration.startswith("[")):
            return 0.0, 0.0
        tokens = [t.strip().strip("'") for t in text[1:-1].split(", ")]
        durations = [float(d.strip().strip("'")) for d in duration[1:-1].split(", ")]
        if (
            len(tokens) != len(durations)
            or tokens[0] != "<space>"
            and tokens[-1] != "<space>"
        ):
            return 0.0, 0.0
        total = sum(durations)
        front = tail = 0.0
        i, j = 0, len(tokens) - 1
        while i <= j and tokens[i] == "<space>":
            front += durations[i]
            i += 1
        while j >= i and tokens[j] == "<space>":
            tail += durations[j]
            j -= 1
        if i > j or total == 0:
            return 0.0, 0.0
        return front / total, tail / total

    def extract_main_data_fn(self, main_data: Dict) -> Dict[str, Any] or None:
        extra = {
            key: main_data.pop(key) for key in EXTRA_KEYS if key in main_data.keys()
        }
        front_ratio, tail_ratio = self._trim_ratios(main_data)

        main_data = super().extract_main_data_fn(main_data)
        if main_data is None:
            return None

        if "feat_ref" in extra.keys():
            feat_ref, sample_rate = read_data_by_path(
                extra["feat_ref"], return_sample_rate=True, return_tensor=True
            )
            if feat_ref.size(0) == 0:
                return None
            if sample_rate is not None and sample_rate > self.sample_rate:
                if not hasattr(self, "wav_resampler_dict"):
                    self.wav_resampler_dict = {}
                resampler = get_cached_resampler(
                    self.wav_resampler_dict, sample_rate, self.sample_rate
                )
                feat_ref = resampler(feat_ref.squeeze(-1)).unsqueeze(-1)
            elif sample_rate is not None and sample_rate < self.sample_rate:
                raise RuntimeError(
                    f"The reference waveform has a lower sampling rate than {self.sample_rate}!"
                )
            # trim the leading & trailing silence in the same proportion as the target waveform
            start, end = int(front_ratio * len(feat_ref)), int(
                tail_ratio * len(feat_ref)
            )
            feat_ref = feat_ref[start:]
            if end > 0:
                feat_ref = feat_ref[:-end]
            main_data["feat_ref"] = feat_ref

        for key in ["text_asr", "snr_cond"]:
            if key in extra.keys():
                assert isinstance(
                    extra[key], str
                ), f"'{key}' must be a string, got {extra[key]}"
                main_data[key] = extra[key]
        return main_data

    def collate_main_data_fn(
        self, batch_dict: Dict[str, List]
    ) -> Dict[str, torch.Tensor or List]:
        extra = {
            key: batch_dict.pop(key) for key in EXTRA_KEYS if key in batch_dict.keys()
        }
        batch_dict = super().collate_main_data_fn(batch_dict)

        if "feat_ref" in extra.keys():
            feat_ref_len = torch.LongTensor([ele.shape[0] for ele in extra["feat_ref"]])
            feat_ref = torch.zeros(
                (
                    len(extra["feat_ref"]),
                    feat_ref_len.max().item(),
                    extra["feat_ref"][0].shape[-1],
                ),
                dtype=torch.float32,
            )
            for i, ele in enumerate(extra["feat_ref"]):
                feat_ref[i][: feat_ref_len[i]] = torch.as_tensor(ele)
            batch_dict["feat_ref"], batch_dict["feat_ref_len"] = feat_ref, feat_ref_len

        for key in ["text_asr", "snr_cond"]:
            if key in extra.keys():
                batch_dict[key] = extra[key]
        return batch_dict

    def __repr__(self):
        return super().__repr__() + f", extra_keys={EXTRA_KEYS}"
