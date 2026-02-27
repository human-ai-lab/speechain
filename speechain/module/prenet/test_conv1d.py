import pytest

torch = pytest.importorskip("torch")

from speechain.module.prenet.conv1d import Conv1dEv, Conv1dPrenet


class TestConv1dEv:
    def test_same_padding_odd_kernel(self):
        conv = Conv1dEv(16, 32, 5)
        x = torch.randn(2, 16, 50)
        out = conv(x)
        assert out.shape == (2, 32, 50)

    def test_same_padding_even_kernel(self):
        conv = Conv1dEv(16, 32, 4)
        x = torch.randn(2, 16, 50)
        out = conv(x)
        assert out.shape == (2, 32, 50)

    def test_valid_padding(self):
        conv = Conv1dEv(16, 32, 5, padding_mode="valid")
        x = torch.randn(2, 16, 50)
        out = conv(x)
        assert out.shape == (2, 32, 46)

    def test_full_padding(self):
        conv = Conv1dEv(16, 32, 5, padding_mode="full")
        x = torch.randn(2, 16, 50)
        out = conv(x)
        assert out.shape[0] == 2 and out.shape[1] == 32

    def test_causal_padding(self):
        conv = Conv1dEv(16, 32, 5, padding_mode="causal")
        x = torch.randn(2, 16, 50)
        out = conv(x)
        assert out.shape == (2, 32, 50)

    def test_invalid_padding_mode(self):
        with pytest.raises(ValueError):
            Conv1dEv(16, 32, 5, padding_mode="invalid")

    def test_weight_norm(self):
        conv = Conv1dEv(16, 32, 3, use_weight_norm=True)
        x = torch.randn(2, 16, 20)
        out = conv(x)
        assert out.shape == (2, 32, 20)


class TestConv1dPrenet:
    def test_forward_shape(self):
        prenet = Conv1dPrenet(feat_dim=80, conv_dims=[256, 256], conv_kernel=5)
        x = torch.randn(2, 30, 80)
        x_len = torch.tensor([30, 20])
        out, out_len = prenet(x, x_len)
        assert out.shape == (2, 30, 256)
        assert (out_len == x_len).all()

    def test_output_size(self):
        prenet = Conv1dPrenet(feat_dim=80, conv_dims=[128], conv_kernel=3)
        assert prenet.output_size == 128

    def test_with_linear_part(self):
        prenet = Conv1dPrenet(feat_dim=80, conv_dims=[256], conv_kernel=3, lnr_dims=128)
        x = torch.randn(2, 20, 80)
        x_len = torch.tensor([20, 15])
        out, out_len = prenet(x, x_len)
        assert out.shape == (2, 20, 128)
        assert prenet.output_size == 128

    def test_input_size_kwarg(self):
        prenet = Conv1dPrenet(input_size=40, conv_dims=[64], conv_kernel=3)
        x = torch.randn(2, 10, 40)
        x_len = torch.tensor([10, 8])
        out, _ = prenet(x, x_len)
        assert out.shape == (2, 10, 64)

    def test_no_batchnorm(self):
        prenet = Conv1dPrenet(
            feat_dim=40, conv_dims=[64], conv_kernel=3, conv_batchnorm=False
        )
        x = torch.randn(2, 10, 40)
        x_len = torch.tensor([10, 8])
        out, _ = prenet(x, x_len)
        assert out.shape == (2, 10, 64)
