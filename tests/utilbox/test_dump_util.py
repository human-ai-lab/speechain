"""Tests for speechain/utilbox/dump_util.py"""

import pytest

from speechain.utilbox.dump_util import (
    en_text_process,
    get_readable_memory,
    get_readable_number,
    parse_readable_number,
)


class TestEnTextProcess:
    def test_lowercase(self):
        result = en_text_process("Hello World", txt_format="punc")
        assert result == result.lower()

    def test_punc_format_keeps_punctuation(self):
        result = en_text_process("Hello, world.", txt_format="punc")
        assert "," in result or "." in result

    def test_no_punc_removes_punctuation(self):
        result = en_text_process("Hello, world.", txt_format="no-punc")
        assert "," not in result
        assert "." not in result

    def test_no_punc_keeps_apostrophe(self):
        result = en_text_process("it's a test", txt_format="no-punc")
        assert "'" in result

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError):
            en_text_process("Hello", txt_format="invalid")

    def test_accented_chars_converted(self):
        result = en_text_process("café", txt_format="no-punc")
        assert "é" not in result
        assert "e" in result

    def test_quotes_normalized(self):
        result = en_text_process("she said \u201chello\u201d", txt_format="punc")
        assert "\u201c" not in result
        assert "\u201d" not in result

    def test_double_hyphen_to_comma(self):
        result = en_text_process("word--word", txt_format="punc")
        # double hyphen -> single hyphen -> comma
        assert "--" not in result

    def test_simple_sentence(self):
        result = en_text_process("The quick brown fox", txt_format="punc")
        assert result == "the quick brown fox"


class TestGetReadableNumber:
    def test_small_number(self):
        assert get_readable_number(42) == "42"

    def test_hundred(self):
        assert get_readable_number(300) == "3h"

    def test_kilo(self):
        assert get_readable_number(3000) == "3k"

    def test_kilo_remainder(self):
        assert get_readable_number(3500) == "3k5h"

    def test_million(self):
        assert get_readable_number(5_000_000) == "5m"

    def test_million_kilo(self):
        assert get_readable_number(5_003_000) == "5m3k"

    def test_billion(self):
        assert get_readable_number(1_000_000_000) == "1b"

    def test_float_input(self):
        assert get_readable_number(3000.0) == "3k"

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            get_readable_number("3000")


class TestParseReadableNumber:
    def test_small(self):
        assert parse_readable_number("42") == 42

    def test_kilo(self):
        assert parse_readable_number("3k") == 3000

    def test_million(self):
        assert parse_readable_number("5m") == 5_000_000

    def test_hundred(self):
        assert parse_readable_number("5h") == 500

    def test_roundtrip_kilo(self):
        # Only single-level strings roundtrip correctly in the current implementation
        original = 3000
        assert parse_readable_number(get_readable_number(original)) == original

    def test_roundtrip_million(self):
        original = 5_000_000
        assert parse_readable_number(get_readable_number(original)) == original


class TestGetReadableMemory:
    def test_bytes(self):
        assert get_readable_memory(512) == "512B"

    def test_kilobytes(self):
        assert get_readable_memory(1024) == "1KB "

    def test_megabytes(self):
        assert get_readable_memory(1024 * 1024) == "1MB "

    def test_gigabytes(self):
        assert get_readable_memory(1024**3) == "1GB "

    def test_terabytes(self):
        assert get_readable_memory(1024**4) == "1TB "

    def test_mixed(self):
        # 1MB + 512B
        result = get_readable_memory(1024 * 1024 + 512)
        assert "MB" in result
        assert "512B" in result

    def test_float_input(self):
        assert get_readable_memory(1024.0) == "1KB "

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            get_readable_memory("1024")
