"""Tests for speechain/utilbox/md_util.py"""

import pytest

numpy = pytest.importorskip("numpy")

from speechain.utilbox.md_util import get_list_strings, get_table_strings


class TestGetTableStrings:
    def test_simple_single_row(self):
        contents = [["a", "b", "c"]]
        result = get_table_strings(contents)
        assert "a" in result
        assert "b" in result
        assert "c" in result
        assert "|" in result

    def test_multiple_rows(self):
        contents = [["a", "b"], ["c", "d"]]
        result = get_table_strings(contents)
        assert "a" in result
        assert "d" in result

    def test_with_headers(self):
        contents = [["val1", "val2"]]
        headers = ["Col1", "Col2"]
        result = get_table_strings(contents, headers=headers)
        assert "**Col1**" in result
        assert "**Col2**" in result

    def test_with_headers_not_bold(self):
        contents = [["val1", "val2"]]
        headers = ["Col1", "Col2"]
        result = get_table_strings(contents, headers=headers, header_bold=False)
        assert "Col1" in result
        assert "**Col1**" not in result

    def test_with_first_col(self):
        contents = [["val1", "val2"], ["val3", "val4"]]
        first_col = ["row1", "row2"]
        result = get_table_strings(contents, first_col=first_col)
        assert "**row1**" in result
        assert "**row2**" in result

    def test_with_first_col_not_bold(self):
        contents = [["val1", "val2"], ["val3", "val4"]]
        first_col = ["row1", "row2"]
        result = get_table_strings(contents, first_col=first_col, first_col_bold=False)
        assert "row1" in result
        assert "**row1**" not in result

    def test_with_headers_and_first_col(self):
        contents = [["v1", "v2"], ["v3", "v4"]]
        first_col = ["r1", "r2"]
        headers = ["", "A", "B"]
        result = get_table_strings(contents, first_col=first_col, headers=headers)
        assert "**A**" in result
        assert "**r1**" in result

    def test_separator_row_present(self):
        contents = [["a", "b"]]
        result = get_table_strings(contents)
        assert "---|" in result

    def test_flat_list_promoted_to_nested(self):
        # A flat list should be treated as a single row
        contents = ["a", "b", "c"]
        result = get_table_strings(contents)
        assert "a" in result

    def test_header_length_mismatch_raises(self):
        contents = [["a", "b"]]
        headers = ["H1"]  # wrong length
        with pytest.raises(AssertionError):
            get_table_strings(contents, headers=headers)

    def test_first_col_length_mismatch_raises(self):
        contents = [["a", "b"], ["c", "d"]]
        first_col = ["r1"]  # wrong length
        with pytest.raises(AssertionError):
            get_table_strings(contents, first_col=first_col)


class TestGetListStrings:
    def test_simple(self):
        result = get_list_strings({"Key": "Value"})
        assert "**Key:**" in result
        assert "Value" in result
        assert result.startswith("* ")

    def test_multiple_entries(self):
        result = get_list_strings({"A": "1", "B": "2"})
        assert "**A:**" in result
        assert "**B:**" in result
        assert "1" in result
        assert "2" in result

    def test_not_bold(self):
        result = get_list_strings({"Key": "Value"}, header_bold=False)
        assert "Key:" in result
        assert "**Key:**" not in result

    def test_empty_dict(self):
        result = get_list_strings({})
        assert result == ""

    def test_newline_per_entry(self):
        result = get_list_strings({"A": "1", "B": "2"})
        lines = [l for l in result.split("\n") if l]
        assert len(lines) == 2
