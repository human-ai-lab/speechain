import pytest

torch = pytest.importorskip("torch")

from speechain.module.conformer.pos_enc import RelPositionalEncoding


class TestRelPositionalEncoding:
    def test_forward_shape_mix(self):
        posenc = RelPositionalEncoding(posenc_type="mix", d_model=64)
        x = torch.randn(2, 20, 64)
        emb, pe = posenc(x)
        assert emb.shape == (2, 20, 64)
        # posenc shape: (1, 2*seq_len-1, d_model)
        assert pe.shape == (1, 39, 64)

    def test_forward_shape_sep(self):
        posenc = RelPositionalEncoding(posenc_type="sep", d_model=64)
        x = torch.randn(2, 15, 64)
        emb, pe = posenc(x)
        assert emb.shape == (2, 15, 64)
        assert pe.shape == (1, 29, 64)

    def test_invalid_type(self):
        with pytest.raises(AssertionError):
            RelPositionalEncoding(posenc_type="invalid", d_model=64)

    def test_odd_d_model_raises(self):
        with pytest.raises(AssertionError):
            RelPositionalEncoding(d_model=63)

    def test_emb_scale(self):
        import math

        posenc_scaled = RelPositionalEncoding(d_model=64, emb_scale=True)
        posenc_plain = RelPositionalEncoding(d_model=64, emb_scale=False)
        x = torch.ones(1, 5, 64)
        emb_scaled, _ = posenc_scaled(x.clone())
        emb_plain, _ = posenc_plain(x.clone())
        # scaled embeddings should differ from unscaled by sqrt(d_model)
        assert torch.allclose(emb_scaled, emb_plain * math.sqrt(64))

    def test_long_sequence_updates_buffer(self):
        posenc = RelPositionalEncoding(d_model=64, max_len=10)
        x = torch.randn(1, 50, 64)
        emb, pe = posenc(x)
        assert pe.shape == (1, 99, 64)
