import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")


class TestSaveDataByFormat:
    def test_save_npy(self, tmp_path):
        from speechain.utilbox.data_saving_util import save_data_by_format

        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = save_data_by_format(
            file_format="npy",
            save_path=str(tmp_path),
            file_name_list=["test_array"],
            file_content_list=[data],
        )
        assert "test_array" in result
        saved_path = result["test_array"]
        assert os.path.exists(saved_path)
        loaded = np.load(saved_path)
        np.testing.assert_array_almost_equal(loaded, data)

    def test_save_npy_torch_tensor(self, tmp_path):
        from speechain.utilbox.data_saving_util import save_data_by_format

        data = torch.tensor([1.0, 2.0, 3.0])
        result = save_data_by_format(
            file_format="npy",
            save_path=str(tmp_path),
            file_name_list=["tensor_array"],
            file_content_list=[data],
        )
        assert "tensor_array" in result
        assert os.path.exists(result["tensor_array"])

    def test_save_returns_dict(self, tmp_path):
        from speechain.utilbox.data_saving_util import save_data_by_format

        data = np.zeros(5)
        result = save_data_by_format(
            file_format="npy",
            save_path=str(tmp_path),
            file_name_list=["item"],
            file_content_list=[data],
        )
        assert isinstance(result, dict)
