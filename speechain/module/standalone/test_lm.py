import pytest

torch = pytest.importorskip("torch")

from speechain.module.standalone.lm import LanguageModel


def _make_lm(d_model=64, num_layers=2):
    emb = {"type": "embed", "conf": {"embedding_dim": d_model}}
    encoder = {
        "type": "transformer",
        "conf": {
            "d_model": d_model,
            "num_heads": 4,
            "num_layers": num_layers,
            "fdfwd_dim": d_model * 2,
        },
    }
    return LanguageModel(vocab_size=100, emb=emb, encoder=encoder)


class TestLanguageModel:
    def test_forward_shape(self):
        lm = _make_lm()
        text = torch.randint(1, 100, (2, 15))
        text_len = torch.tensor([15, 10])
        logits, enc_mask, enc_attmat = lm(text, text_len)
        assert logits.shape == (2, 15, 100)

    def test_logits_vocab_size(self):
        lm = _make_lm()
        text = torch.randint(1, 100, (1, 8))
        text_len = torch.tensor([8])
        logits, _, _ = lm(text, text_len)
        assert logits.shape[-1] == 100

    def test_attmat_returned(self):
        lm = _make_lm(num_layers=2)
        text = torch.randint(1, 100, (2, 10))
        text_len = torch.tensor([10, 8])
        _, _, enc_attmat = lm(text, text_len)
        assert enc_attmat is not None
        assert len(enc_attmat) == 2

    def test_single_token(self):
        lm = _make_lm()
        text = torch.randint(1, 100, (2, 1))
        text_len = torch.tensor([1, 1])
        logits, _, _ = lm(text, text_len)
        assert logits.shape == (2, 1, 100)
