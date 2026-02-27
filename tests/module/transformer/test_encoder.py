import pytest

torch = pytest.importorskip("torch")

from speechain.module.transformer.encoder import (
    TransformerEncoder,
    TransformerEncoderLayer,
)


class TestTransformerEncoderLayer:
    def test_forward_shape(self):
        layer = TransformerEncoderLayer(d_model=64, num_heads=4, fdfwd_dim=128)
        src = torch.randn(2, 10, 64)
        mask = torch.ones(2, 1, 10, dtype=torch.bool)
        out, attmat = layer(src, mask)
        assert out.shape == (2, 10, 64)
        assert attmat.shape == (2, 4, 10, 10)

    def test_layernorm_last(self):
        layer = TransformerEncoderLayer(
            d_model=64, num_heads=4, fdfwd_dim=128, layernorm_first=False
        )
        src = torch.randn(2, 8, 64)
        mask = torch.ones(2, 1, 8, dtype=torch.bool)
        out, _ = layer(src, mask)
        assert out.shape == (2, 8, 64)


class TestTransformerEncoder:
    def test_forward_shape(self):
        enc = TransformerEncoder(d_model=64, num_heads=4, num_layers=2, fdfwd_dim=128)
        src = torch.randn(2, 15, 64)
        mask = torch.ones(2, 1, 15, dtype=torch.bool)
        out, out_mask, attmat, hidden = enc(src, mask)
        assert out.shape == (2, 15, 64)
        assert len(attmat) == 2
        assert len(hidden) == 2

    def test_output_size(self):
        enc = TransformerEncoder(d_model=128, num_heads=4, num_layers=2, fdfwd_dim=256)
        assert enc.output_size == 128

    def test_input_size_kwarg(self):
        enc = TransformerEncoder(
            input_size=64, num_heads=4, num_layers=1, fdfwd_dim=128
        )
        src = torch.randn(2, 10, 64)
        mask = torch.ones(2, 1, 10, dtype=torch.bool)
        out, _, _, _ = enc(src, mask)
        assert out.shape == (2, 10, 64)

    def test_unidirectional(self):
        enc = TransformerEncoder(
            d_model=64, num_heads=4, num_layers=1, fdfwd_dim=128, uni_direction=True
        )
        src = torch.randn(2, 8, 64)
        mask = torch.ones(2, 1, 8, dtype=torch.bool)
        out, _, _, _ = enc(src, mask)
        assert out.shape == (2, 8, 64)

    def test_subsequent_mask_shape(self):
        mask = TransformerEncoder.subsequent_mask(2, 10)
        assert mask.shape == (2, 10, 10)
        # lower-triangular: mask[0, i, j] is True iff i >= j
        assert mask[0, 0, 0].item() is True
        assert mask[0, 0, 1].item() is False
