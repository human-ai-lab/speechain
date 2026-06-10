import pytest

torch = pytest.importorskip("torch")

from speechain.criterion.att_guid import AttentionGuidance


class TestAttentionGuidance:
    def setup_method(self):
        self.criterion = AttentionGuidance(sigma=0.2)

    def test_get_weight_matrix_shape(self):
        X, Y = 10, 15
        w = self.criterion.get_weight_matrix(X, Y)
        assert w.shape == (X, Y)

    def test_get_weight_matrix_values_in_range(self):
        w = self.criterion.get_weight_matrix(8, 8)
        assert w.min().item() >= 0.0
        assert w.max().item() <= 1.0

    def test_call_returns_scalar(self):
        batch, heads, xlen, ylen = 2, 4, 10, 12
        att = torch.rand(batch, heads, xlen, ylen)
        x_len = torch.tensor([10, 8])
        y_len = torch.tensor([12, 10])
        result = self.criterion(att, x_len, y_len)
        assert result.dim() == 0

    def test_call_y_len_none(self):
        batch, heads, xlen, ylen = 2, 2, 8, 8
        att = torch.rand(batch, heads, xlen, ylen)
        x_len = torch.tensor([8, 6])
        result = self.criterion(att, x_len)
        assert result.dim() == 0
