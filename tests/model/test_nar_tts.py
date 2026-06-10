import pytest

torch = pytest.importorskip("torch")

try:
    import torchaudio
except (ImportError, OSError):
    pytest.skip("torchaudio not available", allow_module_level=True)


class TestNARTTSImport:
    def test_import(self):
        from speechain.model.nar_tts import FastSpeech2

        assert FastSpeech2 is not None

    def test_is_class(self):
        import inspect

        from speechain.model.nar_tts import FastSpeech2

        assert inspect.isclass(FastSpeech2)
