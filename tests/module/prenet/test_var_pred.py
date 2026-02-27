import pytest

torch = pytest.importorskip("torch")

from speechain.module.prenet.var_pred import Conv1dVarPredictor


class TestConv1dVarPredictor:
    def test_forward_shape(self):
        predictor = Conv1dVarPredictor(feat_dim=256, conv_dims=[256, 256])
        x = torch.randn(2, 30, 256)
        x_len = torch.tensor([30, 20])
        pred, out_len = predictor(x, x_len)
        assert pred.shape == (2, 30)
        assert (out_len == x_len).all()

    def test_output_size(self):
        predictor = Conv1dVarPredictor(feat_dim=128, conv_dims=[128])
        assert predictor.output_size == 128

    def test_input_size_kwarg(self):
        predictor = Conv1dVarPredictor(input_size=64, conv_dims=[64])
        x = torch.randn(2, 10, 64)
        x_len = torch.tensor([10, 8])
        pred, _ = predictor(x, x_len)
        assert pred.shape == (2, 10)

    def test_no_conv_emb(self):
        predictor = Conv1dVarPredictor(feat_dim=64, conv_dims=[64], use_conv_emb=False)
        assert not hasattr(predictor, "conv_emb")

    def test_emb_pred_scalar(self):
        predictor = Conv1dVarPredictor(feat_dim=64, conv_dims=[64], use_conv_emb=True)
        pred_scalar = torch.randn(2, 20)
        emb = predictor.emb_pred_scalar(pred_scalar)
        assert emb.shape == (2, 20, 64)

    def test_use_gate(self):
        predictor = Conv1dVarPredictor(
            feat_dim=64, conv_dims=[64], use_gate=True, use_conv_emb=False
        )
        x = torch.randn(2, 10, 64)
        x_len = torch.tensor([10, 8])
        result = predictor(x, x_len)
        assert len(result) == 3
        pred, out_len, gate = result
        assert pred.shape == (2, 10)
        assert gate.shape == (2, 10)
