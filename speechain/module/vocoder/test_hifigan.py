import unittest
import torch
import numpy as np
from module.vocoder.hifigan import HiFiGAN, ResBlock, MRF


class TestHiFiGAN(unittest.TestCase):
    def setUp(self):
        self.model = HiFiGAN()
        self.device = torch.device("cpu")

    def test_resblock_forward(self):
        resblock = ResBlock(channels=64)
        x = torch.randn(2, 64, 100)
        out = resblock(x)
        self.assertEqual(out.shape, x.shape)

    def test_mrf_forward(self):
        mrf = MRF(channels=64)
        x = torch.randn(2, 64, 100)
        out = mrf(x)
        self.assertEqual(out.shape, x.shape)

    def test_hifigan_forward(self):
        batch_size = 2
        seq_len = 80
        mel_channels = 80
        x = torch.randn(batch_size, mel_channels, seq_len)
        out = self.model(x)
        expected_len = seq_len * np.prod([8, 8, 2, 2])
        self.assertEqual(out.shape, (batch_size, expected_len))

    def test_decode_batch(self):
        batch_size = 2
        seq_len = 80
        mel_channels = 80
        feats = torch.randn(batch_size, seq_len, mel_channels)
        out = self.model.decode_batch(feats)
        expected_len = seq_len * np.prod([8, 8, 2, 2])
        self.assertEqual(out.shape, (batch_size, expected_len))

    def test_from_hparams_no_weights(self):
        model = HiFiGAN.from_hparams(source=None, run_opts={"device": self.device})
        self.assertIsInstance(model, HiFiGAN)
        self.assertEqual(next(model.parameters()).device, self.device)

    def test_model_output_range(self):
        batch_size = 2
        seq_len = 80
        mel_channels = 80
        x = torch.randn(batch_size, mel_channels, seq_len)
        out = self.model(x)
        self.assertTrue(torch.all(out >= -1))
        self.assertTrue(torch.all(out <= 1))


if __name__ == "__main__":
    unittest.main()
