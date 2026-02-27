import pytest


class TestG2PTokenizerImport:
    def test_module_import(self):
        g2p_en = pytest.importorskip("g2p_en")
        from speechain.tokenizer.g2p import GraphemeToPhonemeTokenizer

        assert GraphemeToPhonemeTokenizer is not None

    def test_abnormal_phns_list(self):
        g2p_en = pytest.importorskip("g2p_en")
        from speechain.tokenizer.g2p import abnormal_phns, cmu_phn_list

        assert isinstance(abnormal_phns, list)
        assert len(abnormal_phns) > 0
        assert isinstance(cmu_phn_list, list)
        assert len(cmu_phn_list) > 0
