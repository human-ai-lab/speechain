import pytest

from speechain.utilbox.type_util import (
    str2bool,
    str2dict,
    str2list,
    str2none,
    str_or_int,
)


class TestStr2Bool:
    def test_true_values(self):
        assert str2bool("true") is True
        assert str2bool("True") is True
        assert str2bool("TRUE") is True
        assert str2bool("ture") is True  # typo tolerance

    def test_false_values(self):
        assert str2bool("false") is False
        assert str2bool("False") is False
        assert str2bool("FALSE") is False
        assert str2bool("flase") is False  # typo tolerance

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            str2bool("yes")
        with pytest.raises(ValueError):
            str2bool("1")
        with pytest.raises(ValueError):
            str2bool("")


class TestStr2None:
    def test_none_strings(self):
        assert str2none("none") is None
        assert str2none("None") is None
        assert str2none("NONE") is None
        assert str2none("null") is None
        assert str2none("Null") is None
        assert str2none("") is None

    def test_non_none_strings(self):
        assert str2none("hello") == "hello"
        assert str2none("0") == "0"
        assert str2none("false") == "false"


class TestStrOrInt:
    def test_digit_strings(self):
        assert str_or_int("0") == 0
        assert str_or_int("42") == 42
        assert isinstance(str_or_int("5"), int)

    def test_empty_string(self):
        assert str_or_int("") is None

    def test_non_digit_strings(self):
        assert str_or_int("hello") == "hello"
        assert str_or_int("1.5") == "1.5"
        assert str_or_int("abc123") == "abc123"


class TestStr2List:
    def test_simple_list(self):
        assert str2list("[a,b,c]") == ["a", "b", "c"]

    def test_nested_list(self):
        result = str2list("[a,[1,2],c]")
        assert result == ["a", [1, 2], "c"]

    def test_type_casting_int(self):
        assert str2list("[1,2,3]") == [1, 2, 3]

    def test_type_casting_float(self):
        result = str2list("[1.1,2.2]")
        assert result == [1.1, 2.2]

    def test_type_casting_bool(self):
        result = str2list("[true,false]")
        assert result == [True, False]

    def test_comma_separated_no_brackets(self):
        assert str2list("a,b,c") == ["a", "b", "c"]

    def test_deeply_nested(self):
        result = str2list("[a,[1,2,[1.1,2.2]]]")
        assert result == ["a", [1, 2, [1.1, 2.2]]]

    def test_mismatched_brackets_raises(self):
        with pytest.raises(AssertionError):
            str2list("[a,b,c")


class TestStr2Dict:
    def test_empty_string(self):
        assert str2dict("") == {}

    def test_empty_braces(self):
        assert str2dict("{}") == {}

    def test_simple_key_value(self):
        assert str2dict("a:1") == {"a": 1}

    def test_string_value(self):
        assert str2dict("a:xyz") == {"a": "xyz"}

    def test_float_value(self):
        assert str2dict("a:1.5") == {"a": 1.5}

    def test_bool_value(self):
        assert str2dict("a:true") == {"a": True}
        assert str2dict("a:false") == {"a": False}

    def test_multiple_keys(self):
        result = str2dict("a:1,b:2")
        assert result == {"a": 1, "b": 2}

    def test_nested_dict(self):
        result = str2dict("a:{b:1,c:2}")
        assert result == {"a": {"b": 1, "c": 2}}

    def test_plain_string_no_colon(self):
        assert str2dict("somepath/file") == "somepath/file"

    def test_list_value(self):
        result = str2dict("a:{val:[1,2,3]}")
        assert result == {"a": {"val": [1, 2, 3]}}
