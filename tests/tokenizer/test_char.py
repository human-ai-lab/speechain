import os

import pytest

torch = pytest.importorskip("torch")

from speechain.tokenizer.char import CharTokenizer


@pytest.fixture
def vocab_dir(tmp_path):
    tokens = ["<blank>", "<sos/eos>", "<unk>", "a", "b", "c"]
    vocab_file = tmp_path / "vocab"
    vocab_file.write_text("\n".join(tokens) + "\n")
    return str(tmp_path)


@pytest.fixture
def tokenizer(vocab_dir):
    return CharTokenizer(token_path=vocab_dir)


class TestCharTokenizer:
    def test_instantiation(self, tokenizer):
        assert tokenizer is not None
        assert tokenizer.vocab_size == 6

    def test_special_token_indices(self, tokenizer):
        assert tokenizer.ignore_idx == 0  # <blank>
        assert tokenizer.sos_eos_idx == 1  # <sos/eos>
        assert tokenizer.unk_idx == 2  # <unk>

    def test_text2tensor_returns_long_tensor(self, tokenizer):
        t = tokenizer.text2tensor("ab")
        assert isinstance(t, torch.Tensor)
        assert t.dtype == torch.long

    def test_text2tensor_with_sos_eos(self, tokenizer):
        t = tokenizer.text2tensor("ab")
        # first and last token should be sos/eos
        assert t[0].item() == tokenizer.sos_eos_idx
        assert t[-1].item() == tokenizer.sos_eos_idx

    def test_text2tensor_no_sos(self, tokenizer):
        t = tokenizer.text2tensor("ab", no_sos=True, no_eos=False)
        assert t[0].item() != tokenizer.sos_eos_idx
        assert t[-1].item() == tokenizer.sos_eos_idx

    def test_text2tensor_no_eos(self, tokenizer):
        t = tokenizer.text2tensor("ab", no_sos=False, no_eos=True)
        assert t[0].item() == tokenizer.sos_eos_idx
        assert t[-1].item() != tokenizer.sos_eos_idx

    def test_text2tensor_no_sos_no_eos(self, tokenizer):
        t = tokenizer.text2tensor("abc", no_sos=True, no_eos=True)
        assert len(t) == 3

    def test_text2tensor_unk_token(self, tokenizer):
        t = tokenizer.text2tensor("az", no_sos=True, no_eos=True)
        # 'z' is not in vocab, should map to unk_idx
        assert t[1].item() == tokenizer.unk_idx

    def test_tensor2text_roundtrip(self, tokenizer):
        text = "abc"
        t = tokenizer.text2tensor(text, no_sos=True, no_eos=True)
        decoded = tokenizer.tensor2text(t)
        assert decoded == text

    def test_tensor2text_skips_special_tokens(self, tokenizer):
        t = tokenizer.text2tensor("ab")
        decoded = tokenizer.tensor2text(t)
        assert "<sos/eos>" not in decoded
        assert decoded == "ab"

    def test_text2tensor_return_list(self, tokenizer):
        result = tokenizer.text2tensor("ab", return_tensor=False)
        assert isinstance(result, list)

    def test_copy_path(self, tmp_path, vocab_dir):
        copy_dir = tmp_path / "copy"
        copy_dir.mkdir()
        tok = CharTokenizer(token_path=vocab_dir, copy_path=str(copy_dir))
        assert os.path.exists(str(copy_dir / "token_vocab"))
