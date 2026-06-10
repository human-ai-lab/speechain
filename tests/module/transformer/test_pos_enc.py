import pytest

torch = pytest.importorskip("torch")

from speechain.module.transformer.pos_enc import PositionalEncoding


class TestPositionalEncoding:
    def test_mix_type_output_shape(self):
        d_model = 64
        module = PositionalEncoding(d_model=d_model, posenc_type="mix")
        batch, seq = 2, 30
        x = torch.randn(batch, seq, d_model)
        out = module(x)
        assert out.shape == x.shape

    def test_sep_type_output_shape(self):
        d_model = 64
        module = PositionalEncoding(d_model=d_model, posenc_type="sep")
        batch, seq = 2, 30
        x = torch.randn(batch, seq, d_model)
        out = module(x)
        assert out.shape == x.shape

    def test_posenc_scale_true(self):
        d_model = 32
        module = PositionalEncoding(d_model=d_model, posenc_scale=True, init_alpha=1.0)
        batch, seq = 3, 20
        x = torch.randn(batch, seq, d_model)
        out = module(x)
        assert out.shape == x.shape

    def test_emb_scale_true(self):
        d_model = 64
        module = PositionalEncoding(d_model=d_model, emb_scale=True)
        batch, seq = 2, 15
        x = torch.randn(batch, seq, d_model)
        out = module(x)
        assert out.shape == x.shape
