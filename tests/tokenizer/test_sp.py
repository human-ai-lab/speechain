import pytest

sentencepiece = pytest.importorskip("sentencepiece")

from speechain.tokenizer.sp import SentencePieceTokenizer


class TestSentencePieceTokenizerImport:
    def test_class_exists(self):
        assert SentencePieceTokenizer is not None

    def test_is_class(self):
        import inspect

        assert inspect.isclass(SentencePieceTokenizer)
