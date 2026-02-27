import pytest

torch = pytest.importorskip("torch")

from speechain.module.frontend.delta_feat import DeltaFeature


class TestDeltaFeature:
    def test_first_order_shape(self):
        delta = DeltaFeature(delta_order=1)
        x = torch.randn(2, 30, 80)
        x_len = torch.tensor([30, 20])
        out, out_len = delta(x, x_len)
        # original + 1 delta = 2x feature dim
        assert out.shape == (2, 30, 160)
        assert (out_len == x_len).all()

    def test_second_order_shape(self):
        delta = DeltaFeature(delta_order=2)
        x = torch.randn(2, 30, 80)
        x_len = torch.tensor([30, 25])
        out, out_len = delta(x, x_len)
        # original + delta + delta-delta = 3x feature dim
        assert out.shape == (2, 30, 240)

    def test_delta_n(self):
        delta = DeltaFeature(delta_order=1, delta_N=1)
        x = torch.randn(2, 20, 40)
        x_len = torch.tensor([20, 15])
        out, _ = delta(x, x_len)
        assert out.shape == (2, 20, 80)

    def test_invalid_delta_order(self):
        with pytest.raises(AssertionError):
            DeltaFeature(delta_order=3)

    def test_repr(self):
        delta = DeltaFeature(delta_order=1, delta_N=2)
        r = repr(delta)
        assert "DeltaFeature" in r
