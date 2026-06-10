import pytest

torch = pytest.importorskip("torch")

from speechain.module.prenet.linear import LinearPrenet


class TestLinearPrenet:
    def test_output_shape(self):
        feat_dim, out_dim = 80, 256
        module = LinearPrenet(feat_dim=feat_dim, lnr_dims=[out_dim, out_dim])
        batch, seq = 2, 20
        feat = torch.randn(batch, seq, feat_dim)
        feat_len = torch.tensor([20, 15])
        out_feat, out_len = module(feat, feat_len)
        assert out_feat.shape == (batch, seq, out_dim)
        assert (out_len == feat_len).all()

    def test_with_dropout(self):
        feat_dim, out_dim = 40, 128
        module = LinearPrenet(feat_dim=feat_dim, lnr_dims=[out_dim], lnr_dropout=0.1)
        module.train()
        batch, seq = 3, 10
        feat = torch.randn(batch, seq, feat_dim)
        feat_len = torch.tensor([10, 8, 7])
        out_feat, out_len = module(feat, feat_len)
        assert out_feat.shape == (batch, seq, out_dim)

    def test_without_dropout(self):
        feat_dim, out_dim = 40, 128
        module = LinearPrenet(feat_dim=feat_dim, lnr_dims=[out_dim], lnr_dropout=None)
        batch, seq = 2, 15
        feat = torch.randn(batch, seq, feat_dim)
        feat_len = torch.tensor([15, 12])
        out_feat, out_len = module(feat, feat_len)
        assert out_feat.shape == (batch, seq, out_dim)
