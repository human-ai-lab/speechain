import pytest

torch = pytest.importorskip("torch")

from speechain.module.norm.feat_norm import FeatureNormalization


class TestFeatureNormalization:
    def test_utterance_level_train_mode(self):
        module = FeatureNormalization(input_size=40, norm_type="utterance")
        module.train()
        batch, seq, dim = 2, 20, 40
        feat = torch.randn(batch, seq, dim)
        feat_len = torch.tensor([20, 15])
        out_feat, out_len = module(feat, feat_len)
        assert out_feat.shape == feat.shape

    def test_utterance_level_eval_mode(self):
        module = FeatureNormalization(input_size=40, norm_type="utterance")
        module.eval()
        batch, seq, dim = 2, 20, 40
        feat = torch.randn(batch, seq, dim)
        feat_len = torch.tensor([20, 15])
        out_feat, out_len = module(feat, feat_len)
        assert out_feat.shape == feat.shape

    def test_batch_level_train_mode(self):
        module = FeatureNormalization(input_size=40, norm_type="batch")
        module.train()
        batch, seq, dim = 3, 10, 40
        feat = torch.randn(batch, seq, dim)
        feat_len = torch.tensor([10, 8, 7])
        out_feat, out_len = module(feat, feat_len)
        assert out_feat.shape == feat.shape

    def test_batch_level_eval_mode(self):
        # In eval mode, batch-level normalization falls back to utterance-level
        module = FeatureNormalization(input_size=40, norm_type="batch")
        module.eval()
        batch, seq, dim = 2, 10, 40
        feat = torch.randn(batch, seq, dim)
        feat_len = torch.tensor([10, 8])
        out_feat, out_len = module(feat, feat_len)
        assert out_feat.shape == feat.shape
