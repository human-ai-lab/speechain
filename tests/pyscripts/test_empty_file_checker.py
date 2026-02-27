import pytest

torch = pytest.importorskip("torch")


class TestEmptyFileCheckerImport:
    def test_module_importable(self):
        import speechain.pyscripts.empty_file_checker as efc

        assert efc is not None

    def test_has_main_function(self):
        from speechain.pyscripts.empty_file_checker import main

        assert callable(main)

    def test_has_get_empty_file_function(self):
        from speechain.pyscripts.empty_file_checker import get_empty_file

        assert callable(get_empty_file)
