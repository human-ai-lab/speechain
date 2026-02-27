import pytest

torch = pytest.importorskip("torch")

from speechain.criterion.least_error import LeastError


class TestLeastError:
    def _make_inputs(self, batch=2, seq=5, dim=10):
        pred = torch.randn(batch, seq, dim)
        tgt = torch.randn(batch, seq, dim)
        tgt_len = torch.tensor([seq] * batch)
        return pred, tgt, tgt_len

    def test_l1_loss(self):
        criterion = LeastError(loss_type="L1")
        pred, tgt, tgt_len = self._make_inputs()
        loss = criterion(pred, tgt, tgt_len)
        assert loss.dim() == 0
        assert loss.item() >= 0.0

    def test_l2_loss(self):
        criterion = LeastError(loss_type="L2")
        pred, tgt, tgt_len = self._make_inputs()
        loss = criterion(pred, tgt, tgt_len)
        assert loss.dim() == 0
        assert loss.item() >= 0.0

    def test_l1_plus_l2_loss(self):
        criterion = LeastError(loss_type="L1+L2")
        pred, tgt, tgt_len = self._make_inputs()
        loss = criterion(pred, tgt, tgt_len)
        assert loss.dim() == 0
        assert loss.item() >= 0.0

    def test_invalid_loss_type_raises(self):
        with pytest.raises(AssertionError):
            LeastError(loss_type="L3")
