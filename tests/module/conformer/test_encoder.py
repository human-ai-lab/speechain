import pytest

torch = pytest.importorskip("torch")

from speechain.module.conformer.encoder import ConformerEncoder, ConformerEncoderLayer


class TestConformerEncoderLayer:
    def _make_posenc(self, seq_len, d_model):
        from speechain.module.conformer.pos_enc import RelPositionalEncoding

        pe = RelPositionalEncoding(d_model=d_model)
        x = torch.randn(2, seq_len, d_model)
        _, posenc = pe(x)
        return posenc

    def test_forward_shape(self):
        layer = ConformerEncoderLayer(d_model=64, num_heads=4, fdfwd_dim=128)
        seq_len = 10
        src = torch.randn(2, seq_len, 64)
        mask = torch.ones(2, 1, seq_len, dtype=torch.bool)
        posenc = self._make_posenc(seq_len, 64)
        out, attmat = layer(src, mask, posenc)
        assert out.shape == (2, seq_len, 64)


class TestConformerEncoder:
    def test_forward_shape(self):
        enc = ConformerEncoder(d_model=64, num_heads=4, num_layers=2, fdfwd_dim=128)
        src = torch.randn(2, 15, 64)
        mask = torch.ones(2, 1, 15, dtype=torch.bool)
        out, out_mask, attmat, hidden = enc(src, mask)
        assert out.shape == (2, 15, 64)
        assert len(attmat) == 2
        assert len(hidden) == 2

    def test_output_size(self):
        enc = ConformerEncoder(d_model=128, num_heads=4, num_layers=2, fdfwd_dim=256)
        assert enc.output_size == 128

    def test_input_size_kwarg(self):
        enc = ConformerEncoder(input_size=64, num_heads=4, num_layers=1, fdfwd_dim=128)
        src = torch.randn(2, 10, 64)
        mask = torch.ones(2, 1, 10, dtype=torch.bool)
        out, _, _, _ = enc(src, mask)
        assert out.shape == (2, 10, 64)

    def test_unidirectional(self):
        enc = ConformerEncoder(
            d_model=64, num_heads=4, num_layers=1, fdfwd_dim=128, uni_direction=True
        )
        src = torch.randn(2, 8, 64)
        mask = torch.ones(2, 1, 8, dtype=torch.bool)
        out, _, _, _ = enc(src, mask)
        assert out.shape == (2, 8, 64)
