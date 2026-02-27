import pytest

torch = pytest.importorskip("torch")

from speechain.module.transformer.decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
)


class TestTransformerDecoderLayer:
    def test_forward_shape(self):
        layer = TransformerDecoderLayer(d_model=64, num_heads=4, fdfwd_dim=128)
        tgt = torch.randn(2, 8, 64)
        src = torch.randn(2, 10, 64)
        tgt_mask = torch.ones(2, 8, 8, dtype=torch.bool)
        src_mask = torch.ones(2, 1, 10, dtype=torch.bool)
        out, self_att, encdec_att = layer(tgt, src, tgt_mask, src_mask)
        assert out.shape == (2, 8, 64)
        assert self_att.shape == (2, 4, 8, 8)
        assert encdec_att.shape == (2, 4, 8, 10)


class TestTransformerDecoder:
    def test_forward_shape(self):
        dec = TransformerDecoder(d_model=64, num_heads=4, num_layers=2, fdfwd_dim=128)
        tgt = torch.randn(2, 8, 64)
        src = torch.randn(2, 10, 64)
        tgt_mask = torch.ones(2, 1, 8, dtype=torch.bool)
        src_mask = torch.ones(2, 1, 10, dtype=torch.bool)
        out, self_attmat, encdec_attmat, hidden = dec(tgt, src, tgt_mask, src_mask)
        assert out.shape == (2, 8, 64)
        assert len(self_attmat) == 2
        assert len(encdec_attmat) == 2
        assert len(hidden) == 2

    def test_output_size(self):
        dec = TransformerDecoder(d_model=128, num_heads=4, num_layers=2, fdfwd_dim=256)
        assert dec.output_size == 128

    def test_input_size_kwarg(self):
        dec = TransformerDecoder(
            input_size=64, num_heads=4, num_layers=1, fdfwd_dim=128
        )
        tgt = torch.randn(2, 5, 64)
        src = torch.randn(2, 10, 64)
        tgt_mask = torch.ones(2, 1, 5, dtype=torch.bool)
        src_mask = torch.ones(2, 1, 10, dtype=torch.bool)
        out, _, _, _ = dec(tgt, src, tgt_mask, src_mask)
        assert out.shape == (2, 5, 64)

    def test_tgt_mask_required(self):
        dec = TransformerDecoder(d_model=64, num_heads=4, num_layers=1, fdfwd_dim=128)
        tgt = torch.randn(2, 5, 64)
        src = torch.randn(2, 10, 64)
        src_mask = torch.ones(2, 1, 10, dtype=torch.bool)
        with pytest.raises(AssertionError):
            dec(tgt, src, tgt_mask=None, src_mask=src_mask)

    def test_subsequent_mask(self):
        mask = TransformerDecoder.subsequent_mask(2, 6)
        assert mask.shape == (2, 6, 6)
        assert mask[0, 0, 0].item() is True
        assert mask[0, 0, 1].item() is False
