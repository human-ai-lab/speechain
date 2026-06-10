import pytest

torch = pytest.importorskip("torch")

try:
    import torchaudio
except (ImportError, OSError):
    pytest.skip("torchaudio not available", allow_module_level=True)


class TestARTTSImport:
    def test_import(self):
        from speechain.model.ar_tts import ARTTS

        assert ARTTS is not None

    def test_is_class(self):
        import inspect

        from speechain.model.ar_tts import ARTTS

        assert inspect.isclass(ARTTS)
