import pytest


class TestBlockIteratorImport:
    def test_import(self):
        from speechain.iterator.block import BlockIterator

        assert BlockIterator is not None

    def test_is_class(self):
        import inspect

        from speechain.iterator.block import BlockIterator

        assert inspect.isclass(BlockIterator)
