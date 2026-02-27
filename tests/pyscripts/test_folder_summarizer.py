import pytest

torch = pytest.importorskip("torch")


class TestFolderSummarizerImport:
    def test_module_importable(self):
        import speechain.pyscripts.folder_summarizer as fs

        assert fs is not None

    def test_has_main_function(self):
        from speechain.pyscripts.folder_summarizer import main

        assert callable(main)
