import pytest

torch = pytest.importorskip("torch")

from speechain.criterion.accuracy import Accuracy


class TestAccuracy:
    def setup_method(self):
        self.criterion = Accuracy()

    def test_perfect_predictions(self):
        # logits and text equal length (no sos/eos)
        batch, seq, vocab = 2, 5, 10
        text = torch.randint(0, vocab, (batch, seq))
        text_len = torch.full((batch,), seq)
        # build logits that argmax to text
        logits = torch.zeros(batch, seq, vocab)
        for b in range(batch):
            for t in range(seq):
                logits[b, t, text[b, t]] = 1.0
        acc = self.criterion(logits, text, text_len)
        assert abs(acc.item() - 1.0) < 1e-5

    def test_partial_correct(self):
        batch, seq, vocab = 2, 4, 10
        text = torch.zeros(batch, seq, dtype=torch.long)
        text_len = torch.full((batch,), seq)
        # logits predict token 0 for all positions (half correct)
        logits = torch.zeros(batch, seq, vocab)
        logits[:, :, 0] = 1.0
        acc = self.criterion(logits, text, text_len)
        assert 0.0 <= acc.item() <= 1.0

    def test_equal_length_no_sos(self):
        batch, seq, vocab = 3, 6, 8
        text = torch.randint(0, vocab, (batch, seq))
        text_len = torch.full((batch,), seq)
        logits = torch.randn(batch, seq, vocab)
        acc = self.criterion(logits, text, text_len)
        assert acc.dim() == 0

    def test_sos_prefix_stripped(self):
        # text has sos prefix: shape (batch, seq+1), logits shape (batch, seq, vocab)
        batch, seq, vocab = 2, 5, 10
        sos_id = 0
        text_body = torch.randint(1, vocab, (batch, seq))
        sos_col = torch.full((batch, 1), sos_id)
        text = torch.cat([sos_col, text_body], dim=1)  # (batch, seq+1)
        text_len = torch.full((batch,), seq + 1)
        # logits that perfectly predict text_body
        logits = torch.zeros(batch, seq, vocab)
        for b in range(batch):
            for t in range(seq):
                logits[b, t, text_body[b, t]] = 1.0
        acc = self.criterion(logits, text, text_len)
        assert abs(acc.item() - 1.0) < 1e-5
