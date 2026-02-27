import pytest

try:
    from speechain.module.decoder.ar_asr import ARASRDecoder

    HAS_DEPS = True
except (ImportError, OSError):
    HAS_DEPS = False
    ARASRDecoder = None  # type: ignore[assignment,misc]

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not HAS_DEPS, reason="required dependencies not available in this environment"
)


def _make_decoder(d_model=64, num_layers=2, vocab_size=100):
    embedding = {
        "type": "prenet.embed.EmbedPrenet",
        "conf": {"embedding_dim": d_model},
    }
    decoder = {
        "type": "transformer.decoder.TransformerDecoder",
        "conf": {
            "d_model": d_model,
            "num_heads": 4,
            "num_layers": num_layers,
            "fdfwd_dim": d_model * 2,
        },
    }
    return ARASRDecoder(embedding=embedding, decoder=decoder, vocab_size=vocab_size)


class TestARASRDecoder:
    def test_forward_shape(self):
        dec = _make_decoder()
        enc_feat = torch.randn(2, 20, 64)
        enc_feat_mask = torch.ones(2, 1, 20, dtype=torch.bool)
        text = torch.randint(1, 100, (2, 10))
        text_len = torch.tensor([10, 8])
        logits, self_attmat, encdec_attmat, hidden = dec(
            enc_feat, enc_feat_mask, text, text_len
        )
        assert logits.shape == (2, 10, 100)

    def test_attmat_lengths(self):
        dec = _make_decoder(num_layers=3)
        enc_feat = torch.randn(2, 15, 64)
        enc_feat_mask = torch.ones(2, 1, 15, dtype=torch.bool)
        text = torch.randint(1, 100, (2, 8))
        text_len = torch.tensor([8, 6])
        _, self_attmat, encdec_attmat, hidden = dec(
            enc_feat, enc_feat_mask, text, text_len
        )
        assert len(self_attmat) == 3
        assert len(encdec_attmat) == 3
        assert len(hidden) == 3

    def test_vocab_size_in_logits(self):
        dec = _make_decoder(vocab_size=200)
        enc_feat = torch.randn(1, 10, 64)
        enc_feat_mask = torch.ones(1, 1, 10, dtype=torch.bool)
        text = torch.randint(1, 200, (1, 5))
        text_len = torch.tensor([5])
        logits, _, _, _ = dec(enc_feat, enc_feat_mask, text, text_len)
        assert logits.shape[-1] == 200
