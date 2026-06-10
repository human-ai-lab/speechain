import pytest

torch = pytest.importorskip("torch")


class TestDatasetAbsImport:
    def test_import(self):
        from speechain.dataset.abs import Dataset

        assert Dataset is not None

    def test_is_subclass_of_torch_dataset(self):
        from speechain.dataset.abs import Dataset

        assert issubclass(Dataset, torch.utils.data.Dataset)
