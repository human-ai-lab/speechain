import pytest

torch = pytest.importorskip("torch")

from speechain.criterion.cross_entropy import CrossEntropy


class TestCrossEntropy:
    def test_basic_forward(self):
        criterion = CrossEntropy()
        batch, seq, vocab = 2, 5, 20
        logits = torch.randn(batch, seq, vocab)
        text = torch.randint(0, vocab, (batch, seq))
        text_len = torch.tensor([5, 4])
        loss = criterion(logits, text, text_len)
        assert loss.dim() == 0
        assert loss.item() > 0.0

    def test_label_smoothing(self):
        criterion = CrossEntropy(label_smoothing=0.1)
        batch, seq, vocab = 2, 5, 20
        logits = torch.randn(batch, seq, vocab)
        text = torch.randint(0, vocab, (batch, seq))
        text_len = torch.tensor([5, 4])
        loss = criterion(logits, text, text_len)
        assert loss.dim() == 0

    def test_length_normalized(self):
        criterion = CrossEntropy(length_normalized=True)
        batch, seq, vocab = 3, 6, 15
        logits = torch.randn(batch, seq, vocab)
        text = torch.randint(0, vocab, (batch, seq))
        text_len = torch.tensor([6, 5, 4])
        loss = criterion(logits, text, text_len)
        assert loss.dim() == 0
        assert loss.item() > 0.0
