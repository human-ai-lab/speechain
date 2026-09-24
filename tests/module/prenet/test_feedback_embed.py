import pytest

torch = pytest.importorskip("torch")

from speechain.module.prenet.feedback_embed import FeedbackEmbedPrenet


class TestFeedbackEmbedPrenet:
    def test_utterance_level(self):
        module = FeedbackEmbedPrenet(d_model=8, snr_emb_dim=4)
        enc = torch.zeros(2, 5, 8)
        out = module(enc, snr_emb=torch.randn(2, 4), asr_loss=torch.tensor([1.0, 3.0]))
        assert out.shape == (2, 5, 8)
        # the same feedback is added to every token of an utterance
        assert torch.allclose(out[0, 0], out[0, 4])

    def test_token_level_and_coefficients(self):
        module = FeedbackEmbedPrenet(
            d_model=8, snr_emb_dim=8, snr_coeff=0.5, asr_coeff=0.0
        )
        enc = torch.zeros(2, 3, 8)
        snr_emb = torch.randn(2, 3, 8)
        out = module(enc, snr_emb=snr_emb, asr_loss=torch.tensor([1.0, 1.0]))
        assert torch.allclose(out, 0.5 * snr_emb)
        # coefficients can be overridden per call
        out = module(enc, snr_emb=snr_emb, snr_coeff=0.0)
        assert torch.allclose(out, enc)

    def test_no_feedback_is_identity(self):
        module = FeedbackEmbedPrenet(d_model=8)
        enc = torch.randn(1, 4, 8)
        assert torch.equal(module(enc), enc)

    def test_token_length_mismatch(self):
        module = FeedbackEmbedPrenet(d_model=8)
        with pytest.raises(AssertionError):
            module(torch.zeros(1, 4, 8), snr_emb=torch.zeros(1, 3, 8))
