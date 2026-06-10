import pytest

torch = pytest.importorskip("torch")

from speechain.criterion.error_rate import ErrorRate, text_preprocess


class _MockTokenizer:
    """Minimal mock tokenizer for string-input tests (tokenizer not actually called)."""

    ignore_idx = 0
    sos_eos_idx = 1
    unk_idx = 2

    def tensor2text(self, tensor):
        return "mock"


class TestTextPreprocess:
    def test_string_passthrough(self):
        tok = _MockTokenizer()
        assert text_preprocess("hello world", tok) == "hello world"

    def test_empty_string(self):
        tok = _MockTokenizer()
        assert text_preprocess("", tok) == ""

    def test_invalid_type_raises(self):
        tok = _MockTokenizer()
        with pytest.raises(RuntimeError):
            text_preprocess(123, tok)


class TestErrorRate:
    def test_instantiation(self):
        er = ErrorRate()
        assert er is not None

    def test_cer_identical_strings(self):
        er = ErrorRate()
        tok = _MockTokenizer()
        cer, wer = er(["hello"], ["hello"], tokenizer=tok)
        assert cer[0] == 0.0

    def test_wer_identical_strings(self):
        er = ErrorRate()
        tok = _MockTokenizer()
        cer, wer = er(["hello world"], ["hello world"], tokenizer=tok)
        assert wer[0] == 0.0

    def test_cer_completely_different(self):
        er = ErrorRate()
        tok = _MockTokenizer()
        cer, wer = er(["abc"], ["xyz"], tokenizer=tok)
        assert cer[0] == 1.0

    def test_wer_one_word_wrong(self):
        er = ErrorRate()
        tok = _MockTokenizer()
        cer, wer = er(["hello world"], ["hello there"], tokenizer=tok)
        assert wer[0] == pytest.approx(0.5)

    def test_returns_lists_without_do_aver(self):
        er = ErrorRate()
        tok = _MockTokenizer()
        cer, wer = er(["hello", "world"], ["hello", "earth"], tokenizer=tok)
        assert isinstance(cer, list)
        assert isinstance(wer, list)
        assert len(cer) == 2

    def test_do_aver_returns_scalar(self):
        er = ErrorRate()
        tok = _MockTokenizer()
        cer, wer = er(
            ["hello", "world"], ["hello", "earth"], tokenizer=tok, do_aver=True
        )
        assert isinstance(cer, float)
        assert isinstance(wer, float)

    def test_single_string_input(self):
        er = ErrorRate()
        tok = _MockTokenizer()
        cer, wer = er("hello", "hello", tokenizer=tok)
        assert cer[0] == 0.0

    def test_uses_instance_tokenizer(self):
        tok = _MockTokenizer()
        er = ErrorRate(tokenizer=tok)
        cer, wer = er(["abc"], ["abc"])
        assert cer[0] == 0.0
