import pytest

torch = pytest.importorskip("torch")

from speechain.module.conformer.attention import RelPosMultiHeadedAttention
from speechain.module.conformer.pos_enc import RelPositionalEncoding


class TestRelPosMultiHeadedAttention:
    def _make_posenc(self, seq_len, d_model):
        posenc_mod = RelPositionalEncoding(d_model=d_model)
        x = torch.randn(2, seq_len, d_model)
        _, posenc = posenc_mod(x)
        return posenc

    def test_forward_shape(self):
        attn = RelPosMultiHeadedAttention(num_heads=4, d_model=64)
        seq_len = 10
        x = torch.randn(2, seq_len, 64)
        posenc = self._make_posenc(seq_len, 64)
        out, score = attn(x, x, x, posenc=posenc)
        assert out.shape == (2, seq_len, 64)
        assert score.shape == (2, 4, seq_len, seq_len)

    def test_with_mask(self):
        attn = RelPosMultiHeadedAttention(num_heads=4, d_model=64)
        seq_len = 8
        x = torch.randn(2, seq_len, 64)
        posenc = self._make_posenc(seq_len, 64)
        mask = torch.ones(2, 1, seq_len, dtype=torch.bool)
        out, _ = attn(x, x, x, mask=mask, posenc=posenc)
        assert out.shape == (2, seq_len, 64)

    def test_posenc_required(self):
        attn = RelPosMultiHeadedAttention(num_heads=4, d_model=64)
        x = torch.randn(2, 5, 64)
        with pytest.raises(AssertionError):
            attn(x, x, x)
