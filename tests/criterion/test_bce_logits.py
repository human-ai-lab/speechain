import pytest

torch = pytest.importorskip("torch")

from speechain.criterion.bce_logits import BCELogits


class TestBCELogits:
    def test_returns_scalar(self):
        criterion = BCELogits()
        batch, seq = 3, 8
        pred = torch.randn(batch, seq)
        tgt = torch.randint(0, 2, (batch, seq)).float()
        tgt_len = torch.tensor([8, 6, 5])
        result = criterion(pred, tgt, tgt_len)
        assert result.dim() == 0
        assert result.item() >= 0.0

    def test_is_normalized_true(self):
        criterion = BCELogits(is_normalized=True)
        batch, seq = 2, 6
        pred = torch.randn(batch, seq)
        tgt = torch.randint(0, 2, (batch, seq)).float()
        tgt_len = torch.tensor([6, 4])
        result = criterion(pred, tgt, tgt_len)
        assert result.dim() == 0

    def test_is_normalized_false(self):
        criterion = BCELogits(is_normalized=False)
        batch, seq = 2, 6
        pred = torch.randn(batch, seq)
        tgt = torch.randint(0, 2, (batch, seq)).float()
        tgt_len = torch.tensor([6, 4])
        result = criterion(pred, tgt, tgt_len)
        assert result.dim() == 0
