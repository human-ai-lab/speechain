import pytest

torch = pytest.importorskip("torch")

from speechain.module.transformer.attention import MultiHeadedAttention


class TestMultiHeadedAttention:
    def test_forward_shape(self):
        attn = MultiHeadedAttention(num_heads=4, d_model=64)
        x = torch.randn(2, 10, 64)
        out, score = attn(x, x, x)
        assert out.shape == (2, 10, 64)
        assert score.shape == (2, 4, 10, 10)

    def test_cross_attention_shape(self):
        attn = MultiHeadedAttention(num_heads=4, d_model=64)
        src = torch.randn(2, 20, 64)
        tgt = torch.randn(2, 10, 64)
        out, score = attn(src, src, tgt)
        assert out.shape == (2, 10, 64)
        assert score.shape == (2, 4, 10, 20)

    def test_with_mask(self):
        attn = MultiHeadedAttention(num_heads=4, d_model=64)
        x = torch.randn(2, 8, 64)
        mask = torch.ones(2, 1, 8, dtype=torch.bool)
        out, _ = attn(x, x, x, mask=mask)
        assert out.shape == (2, 8, 64)

    def test_d_model_not_divisible_raises(self):
        with pytest.raises(AssertionError):
            MultiHeadedAttention(num_heads=3, d_model=64)

    def test_scale_by_head(self):
        attn = MultiHeadedAttention(num_heads=4, d_model=64, scale_dp_by_head=True)
        x = torch.randn(2, 5, 64)
        out, _ = attn(x, x, x)
        assert out.shape == (2, 5, 64)

    def test_different_batch_sizes(self):
        attn = MultiHeadedAttention(num_heads=8, d_model=128)
        for batch in [1, 4, 8]:
            x = torch.randn(batch, 12, 128)
            out, _ = attn(x, x, x)
            assert out.shape == (batch, 12, 128)
