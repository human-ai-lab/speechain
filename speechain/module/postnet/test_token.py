import pytest

torch = pytest.importorskip("torch")

from speechain.module.postnet.token import TokenPostnet


class TestTokenPostnet:
    def test_forward_shape(self):
        postnet = TokenPostnet(vocab_size=100, input_dim=256)
        x = torch.randn(2, 30, 256)
        out = postnet(x)
        assert out.shape == (2, 30, 100)

    def test_output_size(self):
        postnet = TokenPostnet(vocab_size=50, input_dim=128)
        assert postnet.output_size == 50

    def test_input_size_kwarg(self):
        postnet = TokenPostnet(input_size=128, vocab_size=50)
        x = torch.randn(2, 10, 128)
        out = postnet(x)
        assert out.shape == (2, 10, 50)

    def test_single_timestep(self):
        postnet = TokenPostnet(vocab_size=30, input_dim=64)
        x = torch.randn(4, 1, 64)
        out = postnet(x)
        assert out.shape == (4, 1, 30)

    def test_logits_not_softmax(self):
        postnet = TokenPostnet(vocab_size=10, input_dim=32)
        x = torch.randn(1, 5, 32)
        out = postnet(x)
        # raw logits, not bounded to [0,1]
        assert not (out.min() >= 0 and out.max() <= 1)
