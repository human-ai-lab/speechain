import pytest

try:
    import torchaudio
    from unittest.mock import MagicMock

    if isinstance(torchaudio, MagicMock):
        raise ImportError("torchaudio is mocked")
    from speechain.module.frontend.linear2mel import LinearSpec2MelSpec

    HAS_TORCHAUDIO = True
except (ImportError, OSError):
    HAS_TORCHAUDIO = False
    LinearSpec2MelSpec = None  # type: ignore[assignment,misc]

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not HAS_TORCHAUDIO, reason="torchaudio not available in this environment"
)


class TestLinearSpec2MelSpec:
    def _make_module(self, **kwargs):
        defaults = dict(n_fft=400, n_mels=80, sr=16000)
        defaults.update(kwargs)
        return LinearSpec2MelSpec(**defaults)

    def test_forward_shape(self):
        module = self._make_module()
        stft_dim = 400 // 2 + 1
        x = torch.randn(2, 50, stft_dim).abs()
        x_len = torch.tensor([50, 40])
        out, out_len = module(x, x_len)
        assert out.shape == (2, 50, 80)
        assert (out_len == x_len).all()

    def test_no_logging(self):
        module = self._make_module(logging=False)
        stft_dim = 400 // 2 + 1
        x = torch.randn(2, 20, stft_dim).abs()
        x_len = torch.tensor([20, 15])
        out, _ = module(x, x_len)
        assert out.shape == (2, 20, 80)
        # without log, values can be >= 1
        assert out.max() > 0

    def test_htk_scale(self):
        module = self._make_module(mel_scale="htk", mel_norm=False)
        stft_dim = 400 // 2 + 1
        x = torch.randn(2, 10, stft_dim).abs()
        x_len = torch.tensor([10, 8])
        out, _ = module(x, x_len)
        assert out.shape == (2, 10, 80)

    def test_repr(self):
        module = self._make_module()
        r = repr(module)
        assert "LinearSpec2MelSpec" in r
