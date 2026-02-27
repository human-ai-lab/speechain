import pytest

torch = pytest.importorskip("torch")

from speechain.module.augment.specaug import SpecAugment


class TestSpecAugment:
    def test_forward_preserves_shape_and_feat_len(self):
        module = SpecAugment(
            input_size=80,
            time_warp=True,
            freq_mask=True,
            time_mask=True,
        )
        module.eval()
        batch, seq, feat_dim = 2, 50, 80
        feat = torch.randn(batch, seq, feat_dim)
        feat_len = torch.tensor([50, 40])
        out_feat, out_len = module(feat, feat_len)
        assert out_feat.shape == feat.shape
        assert (out_len == feat_len).all()

    def test_time_warp_only(self):
        module = SpecAugment(
            input_size=40,
            time_warp=True,
            freq_mask=False,
            time_mask=False,
        )
        feat = torch.randn(2, 60, 40)
        feat_len = torch.tensor([60, 50])
        out_feat, out_len = module(feat, feat_len)
        assert out_feat.shape == feat.shape

    def test_freq_mask_only(self):
        module = SpecAugment(
            input_size=80,
            time_warp=False,
            freq_mask=True,
            time_mask=False,
        )
        feat = torch.randn(2, 30, 80)
        feat_len = torch.tensor([30, 25])
        out_feat, out_len = module(feat, feat_len)
        assert out_feat.shape == feat.shape

    def test_time_mask_only(self):
        module = SpecAugment(
            input_size=80,
            time_warp=False,
            freq_mask=False,
            time_mask=True,
        )
        feat = torch.randn(2, 40, 80)
        feat_len = torch.tensor([40, 35])
        out_feat, out_len = module(feat, feat_len)
        assert out_feat.shape == feat.shape
        assert (out_len == feat_len).all()
