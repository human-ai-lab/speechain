import pytest

try:
    from speechain.module.encoder.asr import ASREncoder

    HAS_DEPS = True
except (ImportError, OSError):
    HAS_DEPS = False
    ASREncoder = None  # type: ignore[assignment,misc]

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not HAS_DEPS, reason="required dependencies not available in this environment"
)


def _make_asr_encoder(d_model=64, feat_dim=80, num_layers=2):
    encoder = {
        "type": "transformer.encoder.TransformerEncoder",
        "conf": {
            "d_model": d_model,
            "num_heads": 4,
            "num_layers": num_layers,
            "fdfwd_dim": d_model * 2,
        },
    }
    prenet = {
        "type": "prenet.conv2d.Conv2dPrenet",
        "conf": {"feat_dim": feat_dim, "conv_dims": [32, 32], "lnr_dims": d_model},
    }
    return ASREncoder(prenet=prenet, encoder=encoder)


class TestASREncoder:
    def test_forward_shape(self):
        enc = _make_asr_encoder()
        feat = torch.randn(2, 50, 80)
        feat_len = torch.tensor([50, 40])
        feat_out, feat_mask, attmat, hidden = enc(feat, feat_len)
        assert feat_out.shape[-1] == 64
        assert feat_out.shape[0] == 2

    def test_output_size(self):
        enc = _make_asr_encoder(d_model=128)
        assert enc.output_size == 128

    def test_no_prenet(self):
        encoder = {
            "type": "transformer.encoder.TransformerEncoder",
            "conf": {
                "d_model": 64,
                "num_heads": 4,
                "num_layers": 1,
                "fdfwd_dim": 128,
            },
        }
        enc = ASREncoder(input_size=64, encoder=encoder)
        feat = torch.randn(2, 20, 64)
        feat_len = torch.tensor([20, 15])
        feat_out, _, attmat, hidden = enc(feat, feat_len)
        assert feat_out.shape == (2, 20, 64)

    def test_attmat_length(self):
        enc = _make_asr_encoder(num_layers=3)
        feat = torch.randn(2, 40, 80)
        feat_len = torch.tensor([40, 30])
        _, _, attmat, hidden = enc(feat, feat_len)
        assert len(attmat) == 3
        assert len(hidden) == 3
