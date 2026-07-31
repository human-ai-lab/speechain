import pytest

try:
    import torchaudio  # noqa: F401  (imported only to test its availability)
except (ImportError, OSError):
    pytest.skip("torchaudio not available", allow_module_level=True)


class TestSpeechTextDatasetImport:
    def test_import(self):
        from speechain.datasets.speech_text import SpeechTextDataset

        assert SpeechTextDataset is not None

    def test_is_class(self):
        import inspect

        from speechain.datasets.speech_text import SpeechTextDataset

        assert inspect.isclass(SpeechTextDataset)
