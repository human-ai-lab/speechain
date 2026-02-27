import pytest

torch = pytest.importorskip("torch")


class TestLMImport:
    def test_import(self):
        from speechain.model.lm import LM

        assert LM is not None

    def test_is_class(self):
        import inspect

        from speechain.model.lm import LM

        assert inspect.isclass(LM)
