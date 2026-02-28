import pytest

torch = pytest.importorskip("torch")

from speechain.module.encoder.speaker import EncoderClassifier, Res2Block, SEModule


class TestEncoderClassifier:
    def setup_method(self):
        self.device = torch.device("cpu")
        self.ecapa_model = EncoderClassifier(model_type="ecapa")
        self.xvector_model = EncoderClassifier(model_type="xvector")

    def test_semodule_forward(self):
        se = SEModule(channels=512)
        x = torch.randn(2, 512, 100)
        out = se(x)
        assert out.shape == x.shape

    def test_res2block_forward(self):
        res = Res2Block(channels=512)
        x = torch.randn(2, 512, 100)
        out = res(x)
        assert out.shape == x.shape

    def test_ecapa_forward(self):
        batch_size = 2
        seq_len = 100
        mel_channels = 80
        x = torch.randn(batch_size, mel_channels, seq_len)
        out = self.ecapa_model.model(x)
        assert out.shape == (batch_size, 192)

    def test_xvector_forward(self):
        batch_size = 2
        seq_len = 100
        mel_channels = 80
        x = torch.randn(batch_size, mel_channels, seq_len)
        out = self.xvector_model.model(x)
        assert out.shape == (batch_size, 192)

    def test_encode_batch(self):
        batch_size = 2
        seq_len = 16000  # 1 second of raw waveform at 16kHz (encode_batch expects raw waveforms)
        x = torch.randn(batch_size, seq_len)
        out = self.ecapa_model.encode_batch(x)
        assert out.shape == (batch_size, 192)
        assert torch.allclose(torch.norm(out, p=2, dim=1), torch.ones(batch_size))

    def test_invalid_model_type(self):
        with pytest.raises(ValueError):
            EncoderClassifier(model_type="invalid")

    def test_from_hparams_ecapa(self):
        model = EncoderClassifier.from_hparams(
            source="ecapa", run_opts={"device": self.device}
        )
        assert model.model_type == "ecapa"
        assert next(model.parameters()).device == self.device

    def test_from_hparams_xvector(self):
        model = EncoderClassifier.from_hparams(
            source="xvector", run_opts={"device": self.device}
        )
        assert model.model_type == "xvector"
        assert next(model.parameters()).device == self.device
