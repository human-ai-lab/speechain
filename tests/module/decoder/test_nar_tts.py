import pytest

try:
    from unittest.mock import MagicMock

    import torchaudio  # noqa: F401

    if isinstance(torchaudio, MagicMock):
        raise ImportError("torchaudio is mocked")
    from speechain.module.decoder.nar_tts import FastSpeech2Decoder

    HAS_TORCHAUDIO = True
except (ImportError, OSError):
    HAS_TORCHAUDIO = False
    FastSpeech2Decoder = None  # type: ignore[assignment,misc]

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not HAS_TORCHAUDIO, reason="torchaudio not available in this environment"
)


def _make_nar_tts_decoder(n_mels=80, d_model=64):
    feat_frontend = {
        "type": "frontend.speech2mel.Speech2MelSpec",
        "conf": {
            "n_mels": n_mels,
            "hop_length": 160,
            "win_length": 400,
            "sr": 16000,
        },
    }
    decoder = {
        "type": "transformer.encoder.TransformerEncoder",
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
    duration_predictor = {
        "type": "prenet.var_pred.Conv1dVarPredictor",
        "conf": {"conv_dims": [d_model], "conv_kernel": 3},
    }
    pitch_predictor = {
        "type": "prenet.var_pred.Conv1dVarPredictor",
        "conf": {"conv_dims": [d_model], "conv_kernel": 3},
    }
    energy_predictor = {
        "type": "prenet.var_pred.Conv1dVarPredictor",
        "conf": {"conv_dims": [d_model], "conv_kernel": 3},
    }
    return FastSpeech2Decoder(
        input_size=d_model,
        feat_frontend=feat_frontend,
        decoder=decoder,
        postnet=postnet,
        pitch_predictor=pitch_predictor,
        energy_predictor=energy_predictor,
        duration_predictor=duration_predictor,
        feat_normalize=False,
        pitch_normalize=False,
        energy_normalize=False,
    )


class TestFastSpeech2Decoder:
    def test_instantiation(self):
        dec = _make_nar_tts_decoder()
        assert hasattr(dec, "decoder")
        assert hasattr(dec, "postnet")
        assert hasattr(dec, "duration_predictor")
        assert hasattr(dec, "pitch_predictor")
        assert hasattr(dec, "energy_predictor")

    def test_has_feat_frontend(self):
        dec = _make_nar_tts_decoder()
        assert hasattr(dec, "feat_frontend")

    def test_has_feat_pred(self):
        dec = _make_nar_tts_decoder(n_mels=80)
        assert hasattr(dec, "feat_pred")
        # feat_pred output dimension should equal n_mels
        assert dec.feat_pred.out_features == 80

    def test_inference_forward(self):
        dec = _make_nar_tts_decoder(n_mels=80, d_model=64)
        dec.eval()
        batch = 2
        text_len = 10
        enc_text = torch.randn(batch, text_len, 64)
        enc_text_mask = torch.ones(batch, 1, text_len, dtype=torch.bool)
        with torch.no_grad():
            result = dec(enc_text=enc_text, enc_text_mask=enc_text_mask)
        # returns (pred_feat_before, pred_feat_after, pred_duration, pred_pitch, pred_energy, ...)
        assert result is not None
