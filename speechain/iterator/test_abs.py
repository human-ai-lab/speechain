import pytest


class TestIteratorAbsImport:
    def test_import(self):
        from speechain.iterator.abs import Iterator

        assert Iterator is not None

    def test_is_class(self):
        import inspect

        from speechain.iterator.abs import Iterator

        assert inspect.isclass(Iterator)
