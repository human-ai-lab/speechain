"""Tests for speechain/utilbox/yaml_util.py"""

import pytest

ruamel = pytest.importorskip("ruamel.yaml")

from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.scalarfloat import ScalarFloat
from ruamel.yaml.scalarstring import PlainScalarString

from speechain.utilbox.yaml_util import load_yaml, reform_config_dict


class TestReformConfigDict:
    def test_plain_dict_passthrough(self):
        d = {"key": "value", "num": 42}
        result = reform_config_dict(d)
        assert result == {"key": "value", "num": 42}

    def test_nested_dict(self):
        d = {"outer": {"inner": "val"}}
        result = reform_config_dict(d)
        assert result == {"outer": {"inner": "val"}}

    def test_list_passthrough(self):
        lst = [1, 2, 3]
        result = reform_config_dict(lst)
        assert result == [1, 2, 3]

    def test_mixed_nested(self):
        d = {"a": [1, 2, {"b": "c"}]}
        result = reform_config_dict(d)
        assert result == {"a": [1, 2, {"b": "c"}]}

    def test_keys_converted_to_str(self):
        # Integer keys should be converted to strings
        cm = CommentedMap({1: "one"})
        result = reform_config_dict(cm)
        assert "1" in result

    def test_scalar_float_converted(self):
        sf = ScalarFloat(3.14)
        result = reform_config_dict(sf)
        assert isinstance(result, float)
        assert abs(result - 3.14) < 1e-9

    def test_plain_scalar_string_converted(self):
        pss = PlainScalarString("hello")
        result = reform_config_dict(pss)
        assert isinstance(result, str)
        assert result == "hello"


class TestLoadYaml:
    def test_simple_yaml(self, tmp_path):
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("key: value\nnum: 42\n")
        result = load_yaml(str(yaml_file))
        assert result["key"] == "value"
        assert result["num"] == 42

    def test_nested_yaml(self, tmp_path):
        yaml_file = tmp_path / "nested.yaml"
        yaml_file.write_text("outer:\n  inner: hello\n")
        result = load_yaml(str(yaml_file))
        assert result["outer"]["inner"] == "hello"

    def test_list_yaml(self, tmp_path):
        yaml_file = tmp_path / "list.yaml"
        yaml_file.write_text("items:\n  - a\n  - b\n  - c\n")
        result = load_yaml(str(yaml_file))
        assert result["items"] == ["a", "b", "c"]

    def test_nonexistent_file_raises(self):
        with pytest.raises(AssertionError):
            load_yaml("/nonexistent/path/file.yaml")

    def test_file_object(self, tmp_path):
        yaml_file = tmp_path / "obj.yaml"
        yaml_file.write_text("x: 1\n")
        with open(str(yaml_file), "r") as f:
            result = load_yaml(f)
        assert result["x"] == 1

    def test_float_value(self, tmp_path):
        yaml_file = tmp_path / "float.yaml"
        yaml_file.write_text("lr: 0.001\n")
        result = load_yaml(str(yaml_file))
        assert isinstance(result["lr"], float)
        assert abs(result["lr"] - 0.001) < 1e-9
