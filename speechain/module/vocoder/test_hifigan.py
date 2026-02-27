import numpy as np
import pytest

torch = pytest.importorskip("torch")

from speechain.module.vocoder import HIFIGAN
from speechain.module.vocoder.hifigan import HiFiGAN, ResBlock1, ResBlock2


class TestHiFiGAN:
    def setup_method(self):
        self.model = HiFiGAN()
        self.device = torch.device("cpu")

    def test_resblock1_forward(self):
        resblock = ResBlock1(channels=64, kernel_size=3, dilation=(1, 3, 5))
        x = torch.randn(2, 64, 100)
        out = resblock(x)
        assert out.shape == x.shape

    def test_resblock2_forward(self):
        resblock = ResBlock2(channels=64, kernel_size=3, dilation=(1, 3))
        x = torch.randn(2, 64, 100)
        out = resblock(x)
        assert out.shape == x.shape

    def test_hifigan_forward(self):
        batch_size = 2
        seq_len = 80
        mel_channels = 80
        x = torch.randn(batch_size, mel_channels, seq_len)
        out = self.model(x)
        expected_len = seq_len * int(np.prod([8, 8, 2, 2]))
        assert out.shape[0] == batch_size
        assert out.shape[-1] == expected_len

    def test_decode_batch(self):
        batch_size = 2
        seq_len = 80
        mel_channels = 80
        feats = torch.randn(batch_size, seq_len, mel_channels)
        out = self.model.decode_batch(feats)
        expected_len = seq_len * int(np.prod([8, 8, 2, 2]))
        assert out.shape[0] == batch_size
        assert out.shape[-1] == expected_len

    def test_from_hparams_no_weights(self):
        # Just verify that HiFiGAN can be instantiated with default config
        model = HiFiGAN()
        assert isinstance(model, HiFiGAN)
        assert next(model.parameters()).device == torch.device("cpu")

    def test_hifigan_alias(self):
        assert HIFIGAN is HiFiGAN
