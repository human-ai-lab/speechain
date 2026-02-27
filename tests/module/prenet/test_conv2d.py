import pytest

torch = pytest.importorskip("torch")

from speechain.module.prenet.conv2d import Conv2dPrenet


class TestConv2dPrenet:
    def test_forward_shape(self):
        prenet = Conv2dPrenet(feat_dim=80, conv_dims=[64, 64], lnr_dims=256)
        x = torch.randn(2, 50, 80)
        x_len = torch.tensor([50, 40])
        out, out_len = prenet(x, x_len)
        assert out.shape[0] == 2
        assert out.shape[-1] == 256

    def test_output_size(self):
        prenet = Conv2dPrenet(feat_dim=80, conv_dims=[64], lnr_dims=128)
        assert prenet.output_size == 128

    def test_input_size_kwarg(self):
        prenet = Conv2dPrenet(input_size=40, conv_dims=[32], lnr_dims=64)
        x = torch.randn(2, 30, 40)
        x_len = torch.tensor([30, 20])
        out, out_len = prenet(x, x_len)
        assert out.shape[-1] == 64

    def test_with_batchnorm(self):
        prenet = Conv2dPrenet(
            feat_dim=40, conv_dims=[16], conv_batchnorm=True, lnr_dims=32
        )
        x = torch.randn(2, 20, 40)
        x_len = torch.tensor([20, 15])
        out, _ = prenet(x, x_len)
        assert out.shape[-1] == 32

    def test_length_reduced(self):
        prenet = Conv2dPrenet(feat_dim=80, conv_dims=[32, 32], lnr_dims=64)
        x = torch.randn(2, 50, 80)
        x_len = torch.tensor([50, 40])
        _, out_len = prenet(x, x_len)
        # lengths should be reduced by convolutions
        assert (out_len <= x_len).all()

    def test_no_linear(self):
        prenet = Conv2dPrenet(feat_dim=80, conv_dims=[32], lnr_dims=None)
        assert not hasattr(prenet, "linear")
