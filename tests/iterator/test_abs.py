import inspect

import pytest

torch = pytest.importorskip("torch")


class TestIteratorAbsImport:
    def test_import(self):
        from speechain.iterator.abs import Iterator

        assert Iterator is not None

    def test_is_class(self):
        from speechain.iterator.abs import Iterator

        assert inspect.isclass(Iterator)
