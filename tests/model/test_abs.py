import pytest

torch = pytest.importorskip("torch")


class TestModelAbsImport:
    def test_import(self):
        from speechain.model.abs import Model

        assert Model is not None

    def test_is_abstract(self):
        import inspect

        from speechain.model.abs import Model

        assert inspect.isabstract(Model)

    def test_is_torch_module(self):
        from speechain.model.abs import Model

        assert issubclass(Model, torch.nn.Module)
