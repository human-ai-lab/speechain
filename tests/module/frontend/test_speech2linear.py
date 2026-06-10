import pytest

try:
    from unittest.mock import MagicMock

    import torchaudio

    if isinstance(torchaudio, MagicMock):
        raise ImportError("torchaudio is mocked")
    from speechain.module.frontend.speech2linear import Speech2LinearSpec

    HAS_TORCHAUDIO = True
except (ImportError, OSError):
    HAS_TORCHAUDIO = False
    Speech2LinearSpec = None  # type: ignore[assignment,misc]

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not HAS_TORCHAUDIO, reason="torchaudio not available in this environment"
)


class TestSpeech2LinearSpec:
    def _make_module(self, **kwargs):
        defaults = dict(hop_length=160, win_length=400, sr=16000)
        defaults.update(kwargs)
        return Speech2LinearSpec(**defaults)

    def _make_speech(self, batch=2, length=16000):
        speech = torch.randn(batch, length)
        speech_len = torch.tensor([length, length // 2])
        return speech, speech_len

    def test_forward_shape(self):
        module = self._make_module()
        speech, speech_len = self._make_speech()
        feat, feat_len = module(speech, speech_len)
        assert feat.ndim == 3
        assert feat.shape[-1] == 201  # n_fft//2 + 1 = 400//2 + 1
        assert feat.shape[0] == 2

    def test_output_size(self):
        module = self._make_module()
        assert module.output_size == 201

    def test_mag_spec(self):
        module = self._make_module(mag_spec=True)
        speech, speech_len = self._make_speech()
        feat, feat_len = module(speech, speech_len)
        assert feat.shape[-1] == 201

    def test_logging(self):
        module = self._make_module(logging=True, log_base=10.0)
        speech, speech_len = self._make_speech()
        feat, feat_len = module(speech, speech_len)
        assert feat.shape[-1] == 201

    def test_return_energy(self):
        module = self._make_module(return_energy=True)
        speech, speech_len = self._make_speech()
        result = module(speech, speech_len)
        assert len(result) == 4
        feat, feat_len, energy, energy_len = result
        assert energy.ndim == 2
        assert energy.shape[0] == 2

    def test_custom_n_fft(self):
        module = self._make_module(n_fft=512)
        speech, speech_len = self._make_speech()
        feat, _ = module(speech, speech_len)
        assert feat.shape[-1] == 257  # 512//2 + 1

    def test_repr(self):
        module = self._make_module()
        r = repr(module)
        assert "Speech2LinearSpec" in r
