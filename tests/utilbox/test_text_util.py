from speechain.utilbox.text_util import text2word_list


class TestText2WordList:
    def test_simple_words(self):
        assert text2word_list("hello world") == ["hello", "world"]

    def test_trailing_punctuation(self):
        assert text2word_list("hello world.") == ["hello", "world", "."]

    def test_leading_punctuation(self):
        assert text2word_list('"hello world') == ['"', "hello", "world"]

    def test_comma_separated(self):
        assert text2word_list("yes, no") == ["yes", ",", "no"]

    def test_multiple_sentences(self):
        result = text2word_list("Hello world. How are you?")
        assert "Hello" in result
        assert "." in result
        assert "How" in result
        assert "you" in result
        assert "?" in result

    def test_single_word(self):
        assert text2word_list("hello") == ["hello"]

    def test_single_word_with_punctuation(self):
        assert text2word_list("hello.") == ["hello", "."]
