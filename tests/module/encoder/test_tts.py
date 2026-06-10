import pytest

try:
    from speechain.module.encoder.tts import TTSEncoder

    HAS_DEPS = True
except (ImportError, OSError):
    HAS_DEPS = False
    TTSEncoder = None  # type: ignore[assignment,misc]

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not HAS_DEPS, reason="required dependencies not available in this environment"
)


def _make_tts_encoder(d_model=64, num_layers=2, vocab_size=100):
    embedding = {
        "type": "prenet.embed.EmbedPrenet",
        "conf": {"embedding_dim": d_model},
    }
    encoder = {
        "type": "transformer.encoder.TransformerEncoder",
        "conf": {
            "d_model": d_model,
            "num_heads": 4,
            "num_layers": num_layers,
            "fdfwd_dim": d_model * 2,
        },
    }
    return TTSEncoder(vocab_size=vocab_size, embedding=embedding, encoder=encoder)


class TestTTSEncoder:
    def test_forward_shape(self):
        enc = _make_tts_encoder()
        text = torch.randint(1, 100, (2, 15))
        text_len = torch.tensor([15, 10])
        text_out, text_mask, attmat, hidden = enc(text, text_len)
        assert text_out.shape == (2, 15, 64)
        assert text_out.shape[0] == 2

    def test_output_size(self):
        enc = _make_tts_encoder(d_model=128)
        assert enc.output_size == 128

    def test_attmat_length(self):
        enc = _make_tts_encoder(num_layers=3)
        text = torch.randint(1, 100, (2, 10))
        text_len = torch.tensor([10, 8])
        _, _, attmat, hidden = enc(text, text_len)
        assert len(attmat) == 3
        assert len(hidden) == 3

    def test_with_prenet(self):
        embedding = {
            "type": "prenet.embed.EmbedPrenet",
            "conf": {"embedding_dim": 64},
        }
        encoder = {
            "type": "transformer.encoder.TransformerEncoder",
            "conf": {"d_model": 64, "num_heads": 4, "num_layers": 1, "fdfwd_dim": 128},
        }
        prenet = {
            "type": "prenet.conv1d.Conv1dPrenet",
            "conf": {"conv_dims": [64, 64], "conv_kernel": 3},
        }
        enc = TTSEncoder(
            vocab_size=50, embedding=embedding, encoder=encoder, prenet=prenet
        )
        text = torch.randint(1, 50, (2, 12))
        text_len = torch.tensor([12, 8])
        text_out, _, _, _ = enc(text, text_len)
        assert text_out.shape == (2, 12, 64)
