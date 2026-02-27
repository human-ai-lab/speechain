"""Tests for speechain/utilbox/tensor_util.py"""

import pytest

torch = pytest.importorskip("torch")
numpy = pytest.importorskip("numpy")

from speechain.utilbox.tensor_util import (
    clone,
    detach,
    from_batch,
    to_cpu,
    to_native,
)


class TestToNative:
    def test_tensor_to_list(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        result = to_native(t, "list")
        assert isinstance(result, list)
        assert result == [1.0, 2.0, 3.0]

    def test_tensor_to_numpy(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        result = to_native(t, "numpy")
        assert isinstance(result, numpy.ndarray)

    def test_scalar_tensor_to_numpy(self):
        t = torch.tensor(5.0)
        result = to_native(t, "numpy")
        assert isinstance(result, numpy.ndarray)

    def test_non_tensor_passthrough(self):
        result = to_native(42, "list")
        assert result == 42

    def test_detach_called(self):
        t = torch.tensor([1.0, 2.0], requires_grad=True)
        result = to_native(t, "list")
        assert isinstance(result, list)


class TestDetach:
    def test_tensor(self):
        t = torch.tensor([1.0], requires_grad=True)
        result = detach(t)
        assert not result.requires_grad

    def test_list_of_tensors(self):
        tensors = [
            torch.tensor([1.0], requires_grad=True),
            torch.tensor([2.0], requires_grad=True),
        ]
        result = detach(tensors)
        assert isinstance(result, tuple)
        assert all(not r.requires_grad for r in result)

    def test_dict_of_tensors(self):
        d = {"a": torch.tensor([1.0], requires_grad=True)}
        result = detach(d)
        assert not result["a"].requires_grad

    def test_non_tensor_passthrough(self):
        result = detach(42)
        assert result == 42


class TestClone:
    def test_tensor(self):
        t = torch.tensor([1.0, 2.0], requires_grad=True)
        result = clone(t)
        assert not result.requires_grad

    def test_dict(self):
        d = {"x": torch.tensor([3.0], requires_grad=True)}
        result = clone(d)
        assert not result["x"].requires_grad

    def test_tuple(self):
        t = (torch.tensor([1.0], requires_grad=True),)
        result = clone(t)
        assert isinstance(result, tuple)


class TestToCpu:
    def test_tensor_to_list(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        result = to_cpu(t, tgt="list")
        assert isinstance(result, list)
        assert result == [1.0, 2.0, 3.0]

    def test_tensor_to_numpy(self):
        t = torch.tensor([1.0, 2.0])
        result = to_cpu(t, tgt="numpy")
        assert isinstance(result, numpy.ndarray)

    def test_dict_of_tensors(self):
        d = {"a": torch.tensor([1.0, 2.0])}
        result = to_cpu(d, tgt="list")
        assert isinstance(result["a"], list)

    def test_tuple_of_tensors(self):
        t = (torch.tensor([1.0]), torch.tensor([2.0]))
        result = to_cpu(t, tgt="list")
        assert isinstance(result, tuple)
        assert isinstance(result[0], list)

    def test_batch_idx(self):
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = to_cpu(t, tgt="list", batch_idx=0)
        assert result == [1.0, 2.0]

    def test_non_tensor_passthrough(self):
        result = to_cpu(42, tgt="list")
        assert result == 42


class TestFromBatch:
    def test_tensor_with_idx(self):
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = from_batch(t, batch_idx=1)
        assert result.tolist() == [3.0, 4.0]

    def test_tensor_without_idx(self):
        t = torch.tensor([1.0, 2.0])
        result = from_batch(t)
        assert result.tolist() == [1.0, 2.0]

    def test_numpy_array_with_idx(self):
        arr = numpy.array([[1, 2], [3, 4]])
        result = from_batch(arr, batch_idx=0)
        assert list(result) == [1, 2]

    def test_dict_of_tensors(self):
        d = {"a": torch.tensor([[1.0], [2.0]])}
        result = from_batch(d, batch_idx=0)
        assert result["a"].tolist() == [1.0]

    def test_list_of_tensors(self):
        lst = [torch.tensor([[1.0], [2.0]]), torch.tensor([[3.0], [4.0]])]
        result = from_batch(lst, batch_idx=1)
        assert isinstance(result, tuple)
        assert result[0].tolist() == [2.0]

    def test_scalar_tensor(self):
        t = torch.tensor(5.0)
        result = from_batch(t, batch_idx=0)
        # scalar tensor has 0 dims, so batch_idx is ignored
        assert result.item() == 5.0
