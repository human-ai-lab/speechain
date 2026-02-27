import pytest

torch = pytest.importorskip("torch")

try:
    import torchaudio
except (ImportError, OSError):
    pytest.skip("torchaudio not available", allow_module_level=True)


class TestARASRImport:
    def test_import(self):
        from speechain.model.ar_asr import ARASR

        assert ARASR is not None

    def test_is_class(self):
        import inspect

        from speechain.model.ar_asr import ARASR

        assert inspect.isclass(ARASR)
