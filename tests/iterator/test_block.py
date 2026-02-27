import inspect

import pytest

torch = pytest.importorskip("torch")


class TestBlockIteratorImport:
    def test_import(self):
        from speechain.iterator.block import BlockIterator

        assert BlockIterator is not None

    def test_is_class(self):
        from speechain.iterator.block import BlockIterator

        assert inspect.isclass(BlockIterator)
