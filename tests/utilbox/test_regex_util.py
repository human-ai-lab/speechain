"""Tests for speechain/utilbox/regex_util.py"""

from speechain.utilbox.regex_util import (
    has_nested_structure,
    regex_angle_bracket,
    regex_brace,
    regex_square_bracket,
    regex_square_bracket_large,
)


class TestRegexAngleBracket:
    def test_simple(self):
        assert regex_angle_bracket.findall("<exp_root>") == ["<exp_root>"]

    def test_innermost_match(self):
        # Only innermost bracket (no nested angle brackets in regex_angle_bracket)
        assert regex_angle_bracket.findall("<exp<root>") == ["<root>"]

    def test_multiple(self):
        assert regex_angle_bracket.findall("<a> and <b>") == ["<a>", "<b>"]

    def test_no_match(self):
        assert regex_angle_bracket.findall("no brackets here") == []

    def test_empty_brackets(self):
        assert regex_angle_bracket.findall("<>") == ["<>"]


class TestRegexSquareBracket:
    def test_flat(self):
        assert regex_square_bracket.findall("[h,i,j,k]") == ["[h,i,j,k]"]

    def test_nested_returns_innermost(self):
        # Should match innermost bracket only
        result = regex_square_bracket.findall("[f,g,[h,i,j,k]]")
        assert "[h,i,j,k]" in result

    def test_no_match(self):
        assert regex_square_bracket.findall("no brackets") == []

    def test_multiple_flat(self):
        result = regex_square_bracket.findall("[a][b]")
        assert result == ["[a]", "[b]"]


class TestRegexSquareBracketLarge:
    def test_flat(self):
        assert regex_square_bracket_large.findall("[h,i,j,k]") == ["[h,i,j,k]"]

    def test_nested_returns_largest(self):
        result = regex_square_bracket_large.findall("a,b,c,[d,e,[f,g,[h,i,j,k]]]")
        assert result == ["[d,e,[f,g,[h,i,j,k]]]"]

    def test_no_match(self):
        assert regex_square_bracket_large.findall("no brackets") == []


class TestRegexBrace:
    def test_simple(self):
        assert regex_brace.findall("{h,i,j,k}") == ["{h,i,j,k}"]

    def test_nested_returns_innermost(self):
        result = regex_brace.findall("{f,g,{h,i,j,k}}")
        assert "{h,i,j,k}" in result
        assert "{f,g,{h,i,j,k}}" not in result

    def test_no_match(self):
        assert regex_brace.findall("no braces") == []

    def test_multiple(self):
        result = regex_brace.findall("{a},{b}")
        assert result == ["{a}", "{b}"]


class TestHasNestedStructure:
    def test_nested_true(self):
        assert has_nested_structure("[a,[b,c]]") is True

    def test_deeply_nested(self):
        assert has_nested_structure("a,b,c,[d,e,[f,g,[h,i,j,k]]]") is True

    def test_flat_false(self):
        assert has_nested_structure("[h,i,j,k]") is False

    def test_no_brackets(self):
        assert has_nested_structure("no brackets here") is False

    def test_empty_string(self):
        assert has_nested_structure("") is False
