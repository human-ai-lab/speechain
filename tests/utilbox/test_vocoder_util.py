import pytest

torch = pytest.importorskip("torch")


class TestVocoderWrapperImport:
    def test_import(self):
        from speechain.utilbox.vocoder_util import VocoderWrapper

        assert VocoderWrapper is not None

    def test_is_class(self):
        import inspect

        from speechain.utilbox.vocoder_util import VocoderWrapper

        assert inspect.isclass(VocoderWrapper)
