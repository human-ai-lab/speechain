import pytest

torch = pytest.importorskip("torch")

try:
    import torchaudio
except (ImportError, OSError):
    pytest.skip("torchaudio not available", allow_module_level=True)


class TestSpkUtilImport:
    def test_import(self):
        from speechain.utilbox.spk_util import extract_spk_feat

        assert extract_spk_feat is not None

    def test_is_callable(self):
        from speechain.utilbox.spk_util import extract_spk_feat

        assert callable(extract_spk_feat)
