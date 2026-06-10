import pytest


class TestGetWordEditAlignment:
    def test_perfect_match(self):
        from speechain.utilbox.eval_util import get_word_edit_alignment

        ins, dele, sub, table = get_word_edit_alignment("hello world", "hello world")
        assert ins == 0
        assert dele == 0
        assert sub == 0

    def test_insertion(self):
        from speechain.utilbox.eval_util import get_word_edit_alignment

        # hypo has fewer words than real -> insertion in real
        ins, dele, sub, table = get_word_edit_alignment("hello", "hello world")
        assert ins == 1
        assert dele == 0
        assert sub == 0

    def test_deletion(self):
        from speechain.utilbox.eval_util import get_word_edit_alignment

        # hypo has extra word -> deletion from real's perspective
        ins, dele, sub, table = get_word_edit_alignment("hello world", "hello")
        assert dele == 1
        assert ins == 0
        assert sub == 0

    def test_substitution(self):
        from speechain.utilbox.eval_util import get_word_edit_alignment

        ins, dele, sub, table = get_word_edit_alignment("hello world", "hello earth")
        assert sub == 1
        assert ins == 0
        assert dele == 0

    def test_returns_four_values(self):
        from speechain.utilbox.eval_util import get_word_edit_alignment

        result = get_word_edit_alignment("a b c", "a b c")
        assert len(result) == 4

    def test_table_is_string(self):
        from speechain.utilbox.eval_util import get_word_edit_alignment

        _, _, _, table = get_word_edit_alignment("foo", "bar")
        assert isinstance(table, str)
