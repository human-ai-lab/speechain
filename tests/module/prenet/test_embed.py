import pytest

torch = pytest.importorskip("torch")

from speechain.module.prenet.embed import EmbedPrenet


class TestEmbedPrenet:
    def test_forward_shape(self):
        embed = EmbedPrenet(embedding_dim=256, vocab_size=100)
        tokens = torch.randint(0, 100, (2, 20))
        out = embed(tokens)
        assert out.shape == (2, 20, 256)

    def test_output_size(self):
        embed = EmbedPrenet(embedding_dim=128, vocab_size=50)
        assert embed.output_size == 128

    def test_scale_true(self):
        import math

        embed_scaled = EmbedPrenet(embedding_dim=64, vocab_size=50, scale=True)
        embed_unscaled = EmbedPrenet(embedding_dim=64, vocab_size=50, scale=False)
        # copy weights so outputs are comparable
        embed_scaled.embed.weight.data = embed_unscaled.embed.weight.data.clone()
        tokens = torch.randint(0, 50, (2, 10))
        out_scaled = embed_scaled(tokens)
        out_unscaled = embed_unscaled(tokens)
        assert torch.allclose(out_scaled, out_unscaled * math.sqrt(64))

    def test_padding_idx(self):
        embed = EmbedPrenet(embedding_dim=32, vocab_size=20, padding_idx=0)
        tokens = torch.zeros(2, 5, dtype=torch.long)
        out = embed(tokens)
        # padding index embeddings should be zero
        assert torch.allclose(out, torch.zeros_like(out))

    def test_single_token(self):
        embed = EmbedPrenet(embedding_dim=16, vocab_size=10)
        tokens = torch.randint(1, 10, (1, 1))
        out = embed(tokens)
        assert out.shape == (1, 1, 16)
