import pytest

torch = pytest.importorskip("torch")

from speechain.module.postnet.conv1d import Conv1dPostnet


class TestConv1dPostnet:
    def test_forward_shape(self):
        postnet = Conv1dPostnet(feat_dim=80, conv_dims=[256, 256, 0])
        x = torch.randn(2, 30, 80)
        x_len = torch.tensor([30, 20])
        out = postnet(x, x_len)
        assert out.shape == (2, 30, 80)

    def test_output_size(self):
        postnet = Conv1dPostnet(feat_dim=80, conv_dims=[256, 256, 0])
        assert postnet.output_size == 80

    def test_single_layer(self):
        postnet = Conv1dPostnet(feat_dim=40, conv_dims=[128])
        x = torch.randn(2, 20, 40)
        x_len = torch.tensor([20, 15])
        out = postnet(x, x_len)
        assert out.shape == (2, 20, 128)

    def test_no_activation(self):
        postnet = Conv1dPostnet(feat_dim=40, conv_dims=[64], conv_activation=None)
        x = torch.randn(2, 10, 40)
        x_len = torch.tensor([10, 8])
        out = postnet(x, x_len)
        assert out.shape == (2, 10, 64)

    def test_input_size_kwarg(self):
        postnet = Conv1dPostnet(input_size=80, conv_dims=[256, 0])
        x = torch.randn(2, 15, 80)
        x_len = torch.tensor([15, 10])
        out = postnet(x, x_len)
        assert out.shape == (2, 15, 80)

    def test_tanh_activation(self):
        postnet = Conv1dPostnet(feat_dim=40, conv_dims=[64, 0], conv_activation="Tanh")
        x = torch.randn(2, 10, 40)
        x_len = torch.tensor([10, 8])
        out = postnet(x, x_len)
        assert out.shape == (2, 10, 40)

    def test_minus_one_dim(self):
        postnet = Conv1dPostnet(feat_dim=64, conv_dims=[128, -1, 0])
        assert postnet.conv_dims == [128, 128, 64]
