import pytest

torch = pytest.importorskip("torch")


class TestCTCDecodingImport:
    def test_module_importable(self):
        import speechain.infer_func.ctc_decoding as ctc

        assert hasattr(ctc, "CTCPrefixScorer")

    def test_ctc_prefix_scorer_class_exists(self):
        from speechain.infer_func.ctc_decoding import CTCPrefixScorer

        assert CTCPrefixScorer is not None

    def test_ctc_prefix_scorer_instantiation(self):
        from speechain.infer_func.ctc_decoding import CTCPrefixScorer

        batch_size = 2
        beam_size = 3
        vocab_size = 10
        enc_len = 5
        x = torch.randn(batch_size, enc_len, vocab_size).log_softmax(dim=-1)
        enc_lens = torch.tensor([enc_len, enc_len - 1])
        scorer = CTCPrefixScorer(
            x=x,
            enc_lens=enc_lens,
            batch_size=batch_size,
            beam_size=beam_size,
            blank_index=0,
            eos_index=1,
        )
        assert scorer is not None
        assert scorer.vocab_size == vocab_size
        assert scorer.beam_size == beam_size
