import os

import pytest


class TestImportClass:
    def test_import_stdlib_class(self):
        from speechain.utilbox.import_util import import_class

        result = import_class("os.path.join")
        assert result is os.path.join

    def test_import_returns_callable(self):
        from speechain.utilbox.import_util import import_class

        result = import_class("os.path.exists")
        assert callable(result)

    def test_import_is_cached(self):
        from speechain.utilbox.import_util import import_class

        r1 = import_class("os.path.join")
        r2 = import_class("os.path.join")
        assert r1 is r2


class TestParsePathArgs:
    def test_absolute_path_unchanged(self):
        from speechain.utilbox.import_util import parse_path_args

        result = parse_path_args("/tmp/some/path")
        assert result == "/tmp/some/path"

    def test_relative_dot_path(self, tmp_path, monkeypatch):
        from speechain.utilbox.import_util import parse_path_args

        monkeypatch.chdir(tmp_path)
        result = parse_path_args("./subdir")
        assert os.path.isabs(result)
        assert result == os.path.abspath("./subdir")

    def test_toolkit_relative_path_requires_env(self, monkeypatch):
        from speechain.utilbox.import_util import parse_path_args

        monkeypatch.setenv("SPEECHAIN_ROOT", "/tmp/fake_root")
        result = parse_path_args("some/path")
        assert result == "/tmp/fake_root/some/path"


class TestGetIdlePort:
    def test_returns_string(self):
        from speechain.utilbox.import_util import get_idle_port

        port = get_idle_port()
        assert isinstance(port, str)

    def test_returns_valid_port_number(self):
        from speechain.utilbox.import_util import get_idle_port

        port = get_idle_port()
        port_num = int(port)
        assert 15000 <= port_num <= 30000


class TestGetIdleGpu:
    def test_skip_if_no_gpu(self):
        torch = pytest.importorskip("torch")

        if not torch.cuda.is_available():
            pytest.skip("No GPU available")
        from speechain.utilbox.import_util import get_idle_gpu

        gpus = get_idle_gpu(ngpu=1)
        assert isinstance(gpus, list)
