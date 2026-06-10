import pytest

torch = pytest.importorskip("torch")

from speechain.module.transformer.feed_forward import PositionwiseFeedForward


class TestPositionwiseFeedForward:
    def test_linear_type_output_shape(self):
        d_model, fdfwd_dim = 64, 256
        module = PositionwiseFeedForward(
            d_model=d_model, fdfwd_dim=fdfwd_dim, fdfwd_type="linear"
        )
        batch, seq = 2, 20
        x = torch.randn(batch, seq, d_model)
        out = module(x)
        assert out.shape == x.shape

    def test_conv_type_output_shape(self):
        d_model, fdfwd_dim = 64, 256
        module = PositionwiseFeedForward(
            d_model=d_model, fdfwd_dim=fdfwd_dim, fdfwd_type="conv"
        )
        batch, seq = 2, 20
        x = torch.randn(batch, seq, d_model)
        out = module(x)
        assert out.shape == x.shape
