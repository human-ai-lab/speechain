import pytest

torch = pytest.importorskip("torch")

from speechain.tokenizer.abs import Tokenizer
from speechain.tokenizer.char import CharTokenizer


@pytest.fixture
def vocab_dir(tmp_path):
    tokens = ["<blank>", "<sos/eos>", "<unk>", "h", "e", "l", "o"]
    vocab_file = tmp_path / "vocab"
    vocab_file.write_text("\n".join(tokens) + "\n")
    return str(tmp_path)


@pytest.fixture
def tokenizer(vocab_dir):
    return CharTokenizer(token_path=vocab_dir)


class TestTokenizerAbs:
    def test_tokenizer_is_abstract(self):
        import inspect

        assert inspect.isabstract(Tokenizer)

    def test_text2tensor_is_abstract(self):
        import inspect

        assert "text2tensor" in [
            m[0] for m in inspect.getmembers(Tokenizer, predicate=inspect.isfunction)
        ]

    def test_idx2token_mapping(self, tokenizer):
        assert tokenizer.idx2token[0] == "<blank>"
        assert tokenizer.idx2token[1] == "<sos/eos>"
        assert tokenizer.idx2token[2] == "<unk>"
        assert tokenizer.idx2token[3] == "h"

    def test_token2idx_mapping(self, tokenizer):
        assert tokenizer.token2idx["<blank>"] == 0
        assert tokenizer.token2idx["<sos/eos>"] == 1
        assert tokenizer.token2idx["<unk>"] == 2
        assert tokenizer.token2idx["h"] == 3

    def test_special_indices(self, tokenizer):
        assert tokenizer.ignore_idx == 0
        assert tokenizer.sos_eos_idx == 1
        assert tokenizer.unk_idx == 2

    def test_vocab_size(self, tokenizer):
        assert tokenizer.vocab_size == 7

    def test_space_idx_none_when_no_space_token(self, tokenizer):
        assert tokenizer.space_idx is None

    def test_tensor2text_skips_special_tokens(self, tokenizer):
        tensor = [1, 3, 4, 5, 5, 6, 1]  # <sos/eos> h e l l o <sos/eos>
        result = tokenizer.tensor2text(tensor)
        assert result == "hello"

    def test_tensor2text_replaces_unk_with_star(self, tokenizer):
        tensor = [tokenizer.unk_idx]
        result = tokenizer.tensor2text(tensor)
        assert result == "*"

    def test_tensor2text_skips_blank(self, tokenizer):
        tensor = [tokenizer.ignore_idx, 3]
        result = tokenizer.tensor2text(tensor)
        assert result == "h"

    def test_tensor2text_accepts_torch_tensor(self, tokenizer):
        t = torch.LongTensor([3, 4, 5, 5, 6])
        result = tokenizer.tensor2text(t)
        assert result == "hello"
