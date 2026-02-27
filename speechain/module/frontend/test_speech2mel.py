import pytest

try:
    import torchaudio
    from unittest.mock import MagicMock

    if isinstance(torchaudio, MagicMock):
        raise ImportError("torchaudio is mocked")
    from speechain.module.frontend.speech2mel import Speech2MelSpec

    HAS_TORCHAUDIO = True
except (ImportError, OSError):
    HAS_TORCHAUDIO = False
    Speech2MelSpec = None  # type: ignore[assignment,misc]

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not HAS_TORCHAUDIO, reason="torchaudio not available in this environment"
)


class TestSpeech2MelSpec:
    def _make_module(self, **kwargs):
        defaults = dict(n_mels=80, hop_length=160, win_length=400, sr=16000)
        defaults.update(kwargs)
        return Speech2MelSpec(**defaults)

    def _make_speech(self, batch=2, length=16000):
        speech = torch.randn(batch, length)
        speech_len = torch.tensor([length, length // 2])
        return speech, speech_len

    def test_forward_shape(self):
        module = self._make_module()
        speech, speech_len = self._make_speech()
        feat, feat_len = module(speech, speech_len)
        assert feat.ndim == 3
        assert feat.shape[-1] == 80
        assert feat.shape[0] == 2

    def test_output_size(self):
        module = self._make_module(n_mels=40)
        assert module.output_size == 40

    def test_delta_order_1(self):
        module = self._make_module(delta_order=1)
        assert module.output_size == 160  # 80 * 2
        speech, speech_len = self._make_speech()
        feat, _ = module(speech, speech_len)
        assert feat.shape[-1] == 160

    def test_delta_order_2(self):
        module = self._make_module(delta_order=2)
        assert module.output_size == 240  # 80 * 3
        speech, speech_len = self._make_speech()
        feat, _ = module(speech, speech_len)
        assert feat.shape[-1] == 240

    def test_return_energy(self):
        module = self._make_module(return_energy=True)
        speech, speech_len = self._make_speech()
        result = module(speech, speech_len)
        assert len(result) == 4
        feat, feat_len, energy, energy_len = result
        assert feat.shape[-1] == 80

    def test_repr(self):
        module = self._make_module()
        r = repr(module)
        assert "Speech2MelSpec" in r
