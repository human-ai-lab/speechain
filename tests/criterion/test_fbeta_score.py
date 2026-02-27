import pytest

torch = pytest.importorskip("torch")

from speechain.criterion.fbeta_score import FBetaScore


class TestFBetaScore:
    def test_perfect_predictions(self):
        criterion = FBetaScore(beta=1)
        batch, seq = 2, 6
        tgt = torch.randint(0, 2, (batch, seq))
        tgt_len = torch.full((batch,), seq)
        score = criterion(tgt.clone(), tgt, tgt_len)
        assert abs(score.item() - 1.0) < 1e-4

    def test_mixed_predictions(self):
        criterion = FBetaScore(beta=1)
        batch, seq = 2, 6
        pred = torch.zeros(batch, seq, dtype=torch.long)
        tgt = torch.ones(batch, seq, dtype=torch.long)
        tgt_len = torch.full((batch,), seq)
        score = criterion(pred, tgt, tgt_len)
        assert 0.0 <= score.item() <= 1.0

    def test_beta_one(self):
        criterion = FBetaScore(beta=1)
        batch, seq = 3, 8
        pred = torch.randint(0, 2, (batch, seq))
        tgt = torch.randint(0, 2, (batch, seq))
        tgt_len = torch.full((batch,), seq)
        score = criterion(pred, tgt, tgt_len)
        assert score.dim() == 0
