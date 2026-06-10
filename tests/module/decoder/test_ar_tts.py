import pytest

try:
    from unittest.mock import MagicMock

    import torchaudio  # noqa: F401

    if isinstance(torchaudio, MagicMock):
        raise ImportError("torchaudio is mocked")
    from speechain.module.decoder.ar_tts import ARTTSDecoder

    HAS_TORCHAUDIO = True
except (ImportError, OSError):
    HAS_TORCHAUDIO = False
    ARTTSDecoder = None  # type: ignore[assignment,misc]

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not HAS_TORCHAUDIO, reason="torchaudio not available in this environment"
)


def _make_ar_tts_decoder(n_mels=80, d_model=64):
    frontend = {
        "type": "frontend.speech2mel.Speech2MelSpec",
        "conf": {
            "n_mels": n_mels,
            "hop_length": 160,
            "win_length": 400,
            "sr": 16000,
        },
    }
    decoder = {
        "type": "transformer.decoder.TransformerDecoder",
        "conf": {
            "d_model": d_model,
            "num_heads": 4,
            "num_layers": 1,
            "fdfwd_dim": d_model * 2,
        },
    }
    postnet = {
        "type": "postnet.conv1d.Conv1dPostnet",
        "conf": {"conv_dims": [128, 128, 0], "conv_kernel": 5},
    }
    return ARTTSDecoder(frontend=frontend, decoder=decoder, postnet=postnet)


class TestARTTSDecoder:
    def test_instantiation(self):
        dec = _make_ar_tts_decoder()
        assert hasattr(dec, "frontend")
        assert hasattr(dec, "decoder")
        assert hasattr(dec, "postnet")

    def test_output_size(self):
        dec = _make_ar_tts_decoder(n_mels=80)
        assert dec.output_size == 80

    def test_forward_training(self):
        dec = _make_ar_tts_decoder(n_mels=80, d_model=64)
        dec.train()
        # build a short mel feature tensor (already processed, not waveform)
        batch, feat_len_val, feat_dim = 2, 20, 80
        feat = torch.randn(batch, feat_len_val, feat_dim)
        feat_len = torch.tensor([feat_len_val, feat_len_val // 2])
        enc_text = torch.randn(batch, 15, 80)  # 80 = n_mels = actual d_model after input_size override
        enc_text_mask = torch.ones(batch, 1, 15, dtype=torch.bool)
        (
            pred_stop,
            pred_feat_before,
            pred_feat_after,
            tgt_feat,
            tgt_feat_len,
            self_attmat,
            encdec_attmat,
            hidden,
        ) = dec(
            enc_text=enc_text,
            enc_text_mask=enc_text_mask,
            feat=feat,
            feat_len=feat_len,
        )
        assert pred_feat_before.shape[-1] == feat_dim
        assert pred_feat_after.shape[-1] == feat_dim
