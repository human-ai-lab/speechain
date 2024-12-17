import unittest
import torch

# import torch.nn as nn
from module.encoder.speaker import SEModule, Res2Block, SpeakerEncoder


class TestSpeakerEncoder(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.ecapa_model = SpeakerEncoder(model_type="ecapa")
        self.xvector_model = SpeakerEncoder(model_type="xvector")

    def test_semodule_forward(self):
        se = SEModule(channels=512)
        x = torch.randn(2, 512, 100)
        out = se(x)
        self.assertEqual(out.shape, x.shape)

    def test_res2block_forward(self):
        res = Res2Block(channels=512)
        x = torch.randn(2, 512, 100)
        out = res(x)
        self.assertEqual(out.shape, x.shape)

    def test_ecapa_forward(self):
        batch_size = 2
        seq_len = 100
        mel_channels = 80
        x = torch.randn(batch_size, mel_channels, seq_len)
        out = self.ecapa_model.model(x)
        self.assertEqual(out.shape, (batch_size, 192))

    def test_xvector_forward(self):
        batch_size = 2
        seq_len = 100
        mel_channels = 80
        x = torch.randn(batch_size, mel_channels, seq_len)
        out = self.xvector_model.model(x)
        self.assertEqual(out.shape, (batch_size, 192))

    def test_encode_batch(self):
        batch_size = 2
        seq_len = 100
        mel_channels = 80
        x = torch.randn(batch_size, seq_len, mel_channels)
        out = self.ecapa_model.encode_batch(x)
        self.assertEqual(out.shape, (batch_size, 192))
        self.assertTrue(
            torch.allclose(torch.norm(out, p=2, dim=1), torch.ones(batch_size))
        )

    def test_invalid_model_type(self):
        with self.assertRaises(ValueError):
            SpeakerEncoder(model_type="invalid")

    def test_from_hparams_ecapa(self):
        model = SpeakerEncoder.from_hparams(
            source="ecapa", run_opts={"device": self.device}
        )
        self.assertEqual(model.model_type, "ecapa")
        self.assertEqual(next(model.parameters()).device, self.device)

    def test_from_hparams_xvector(self):
        model = SpeakerEncoder.from_hparams(
            source="xvector", run_opts={"device": self.device}
        )
        self.assertEqual(model.model_type, "xvector")
        self.assertEqual(next(model.parameters()).device, self.device)


if __name__ == "__main__":
    unittest.main()
