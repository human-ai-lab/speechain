import pytest

torch = pytest.importorskip("torch")


class TestTTSDecodingImport:
    def test_module_importable(self):
        import speechain.infer_func.tts_decoding as tts

        assert hasattr(tts, "auto_regression")

    def test_auto_regression_callable(self):
        from speechain.infer_func.tts_decoding import auto_regression

        assert callable(auto_regression)
