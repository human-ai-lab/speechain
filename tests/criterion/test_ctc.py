import pytest

torch = pytest.importorskip("torch")

from speechain.criterion.ctc import CTCLoss


@pytest.fixture
def ctc_loss():
    return CTCLoss(blank=0)


class TestCTCLoss:
    def test_instantiation(self, ctc_loss):
        assert ctc_loss is not None
        assert ctc_loss.blank == 0
        assert ctc_loss.zero_infinity is True

    def test_instantiation_custom_blank(self):
        loss = CTCLoss(blank=1, zero_infinity=False)
        assert loss.blank == 1
        assert loss.zero_infinity is False

    def test_forward(self, ctc_loss):
        batch, enc_len, vocab = 2, 10, 8
        ctc_logits = torch.randn(batch, enc_len, vocab)
        enc_feat_len = torch.tensor([enc_len, enc_len - 2])
        # text includes <sos/eos> at start and end (removed inside CTCLoss)
        text = torch.tensor([[1, 3, 4, 5, 1], [1, 3, 4, 1, 0]])
        text_len = torch.tensor([5, 4])
        loss_val = ctc_loss(ctc_logits, enc_feat_len, text, text_len)
        assert isinstance(loss_val, torch.Tensor)
        assert loss_val.ndim == 0  # scalar

    def test_recover_2d_input(self, ctc_loss):
        # shape: (batch, time)
        ctc_text = torch.tensor([[0, 1, 1, 2, 0], [3, 3, 0, 4, 0]])
        ctc_text_len = torch.tensor([5, 5])
        result = ctc_loss.recover(ctc_text, ctc_text_len)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_recover_3d_input(self, ctc_loss):
        # shape: (batch, time, vocab) - uses argmax
        vocab = 5
        ctc_text = torch.zeros(2, 4, vocab)
        ctc_text[:, 0, 1] = 10  # strong prediction for token 1 at step 0
        ctc_text[:, 1, 1] = 10
        ctc_text[:, 2, 2] = 10
        ctc_text[:, 3, 0] = 10  # blank
        ctc_text_len = torch.tensor([4, 4])
        result = ctc_loss.recover(ctc_text, ctc_text_len)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_recover_removes_blanks_and_repeats(self, ctc_loss):
        # blank=0, sequence: [0, 1, 1, 2, 0] should collapse to [1, 2]
        ctc_text = torch.tensor([[0, 1, 1, 2, 0]])
        ctc_text_len = torch.tensor([5])
        result = ctc_loss.recover(ctc_text, ctc_text_len)
        assert result[0] == [1, 2]
