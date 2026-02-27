import pytest

torch = pytest.importorskip("torch")

from speechain.criterion.perplexity import Perplexity


class TestPerplexity:
    def test_returns_positive_scalar(self):
        criterion = Perplexity()
        batch, seq, vocab = 2, 6, 20
        # logits for positions 0..seq-1, text has sos at index 0
        logits = torch.randn(batch, seq - 1, vocab)
        text = torch.randint(0, vocab, (batch, seq))
        text_len = torch.full((batch,), seq)
        ppl = criterion(logits, text, text_len)
        assert ppl.dim() == 0
        assert ppl.item() > 0.0
