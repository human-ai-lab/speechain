"""Tests for speechain/utilbox/data_loading_util.py"""

import json
import os
import sys
from unittest.mock import MagicMock

import pytest

# Mock unavailable C-extension dependencies before any speechain imports
for _mod in ("GPUtil", "h5py", "soundfile"):
    sys.modules.setdefault(_mod, MagicMock())

torch = pytest.importorskip("torch")

from speechain.utilbox.data_loading_util import (
    get_file_birthtime,
    load_idx2data_file,
    search_file_in_subfolder,
)


class TestLoadIdx2DataFile:
    def test_basic_key_value(self, tmp_path):
        f = tmp_path / "idx2text.txt"
        f.write_text("utt1 hello world\nutt2 foo bar\n")
        result = load_idx2data_file(str(f))
        assert result["utt1"] == "hello world"
        assert result["utt2"] == "foo bar"

    def test_sorted_keys(self, tmp_path):
        f = tmp_path / "idx.txt"
        f.write_text("b_utt value_b\na_utt value_a\n")
        result = load_idx2data_file(str(f))
        keys = list(result.keys())
        assert keys == sorted(keys)

    def test_data_type_int(self, tmp_path):
        f = tmp_path / "idx2num.txt"
        f.write_text("utt1 42\nutt2 7\n")
        result = load_idx2data_file(str(f), data_type=int)
        assert result["utt1"] == 42
        assert result["utt2"] == 7

    def test_list_of_files(self, tmp_path):
        f1 = tmp_path / "file1.txt"
        f2 = tmp_path / "file2.txt"
        f1.write_text("a val_a\n")
        f2.write_text("b val_b\n")
        result = load_idx2data_file([str(f1), str(f2)])
        assert "a" in result
        assert "b" in result

    def test_custom_separator(self, tmp_path):
        f = tmp_path / "idx_tab.txt"
        f.write_text("key1\tvalue1\nkey2\tvalue2\n")
        result = load_idx2data_file(str(f), separator="\t")
        assert result["key1"] == "value1"

    def test_nonexistent_file_raises(self):
        with pytest.raises(AssertionError):
            load_idx2data_file("/nonexistent/path/file.txt")

    def test_json_file(self, tmp_path):
        f = tmp_path / "data.json"
        data = {"utt1": "hello", "utt2": "world"}
        f.write_text(json.dumps(data))
        result = load_idx2data_file(str(f))
        assert result["utt1"] == "hello"
        assert result["utt2"] == "world"


class TestSearchFileInSubfolder:
    def test_finds_files_in_dir(self, tmp_path):
        (tmp_path / "file1.txt").write_text("a")
        (tmp_path / "file2.txt").write_text("b")
        result = search_file_in_subfolder(str(tmp_path))
        assert len(result) == 2
        assert all(os.path.isfile(p) for p in result)

    def test_recursive_search(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.txt").write_text("n")
        (tmp_path / "top.txt").write_text("t")
        result = search_file_in_subfolder(str(tmp_path))
        assert len(result) == 2

    def test_with_match_fn(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.csv").write_text("b")
        result = search_file_in_subfolder(
            str(tmp_path), tgt_match_fn=lambda n: n.endswith(".txt")
        )
        assert len(result) == 1
        assert result[0].endswith(".txt")

    def test_return_name(self, tmp_path):
        (tmp_path / "file.txt").write_text("x")
        result = search_file_in_subfolder(str(tmp_path), return_name=True)
        assert result == ["file.txt"]

    def test_sorted_output(self, tmp_path):
        (tmp_path / "z.txt").write_text("z")
        (tmp_path / "a.txt").write_text("a")
        result = search_file_in_subfolder(str(tmp_path))
        assert result == sorted(result)

    def test_single_file_path(self, tmp_path):
        f = tmp_path / "single.txt"
        f.write_text("x")
        result = search_file_in_subfolder(str(f))
        assert len(result) == 1
        assert result[0] == str(f)


class TestGetFileBirthtime:
    def test_returns_float_by_default(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("x")
        result = get_file_birthtime(str(f))
        assert isinstance(result, float)

    def test_returns_string_when_readable(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("x")
        result = get_file_birthtime(str(f), readable_time=True)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_time_positive(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("x")
        result = get_file_birthtime(str(f))
        assert result > 0
