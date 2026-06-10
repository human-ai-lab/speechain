import pytest

torch = pytest.importorskip("torch")


class TestSpeechBrainWrapperImport:
    def test_import(self):
        from speechain.utilbox.sb_util import SpeechBrainWrapper

        assert SpeechBrainWrapper is not None

    def test_is_class(self):
        import inspect

        from speechain.utilbox.sb_util import SpeechBrainWrapper

        assert inspect.isclass(SpeechBrainWrapper)
