import pytest

torch = pytest.importorskip("torch")


class TestModelParaRenamerImport:
    def test_import(self):
        import speechain.pyscripts.model_para_renamer as m

        assert m is not None
