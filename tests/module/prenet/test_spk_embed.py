import pytest

torch = pytest.importorskip("torch")

from speechain.module.prenet.spk_embed import SpeakerEmbedPrenet


class TestSpeakerEmbedPrenet:
    def test_lookup_only_forward(self):
        spk_emb = SpeakerEmbedPrenet(d_model=64, spk_emb_dim_lookup=32, spk_num=10)
        spk_ids = torch.randint(0, 10, (2,))
        lookup_feat, pretrain_feat = spk_emb(spk_ids=spk_ids)
        # forward returns raw embeddings before final projection (shape: (batch, spk_emb_dim_lookup))
        assert lookup_feat.shape == (2, 32)
        assert pretrain_feat is None

    def test_pretrain_only_forward(self):
        spk_emb = SpeakerEmbedPrenet(d_model=64, spk_emb_dim_pretrained=128)
        spk_feat = torch.randn(2, 128)
        lookup_feat, pretrain_feat = spk_emb(spk_feat=spk_feat)
        assert pretrain_feat.shape == (2, 128)
        assert lookup_feat is None

    def test_both_lookup_and_pretrain(self):
        spk_emb = SpeakerEmbedPrenet(
            d_model=64, spk_emb_dim_lookup=32, spk_num=10, spk_emb_dim_pretrained=128
        )
        spk_ids = torch.randint(0, 10, (2,))
        spk_feat = torch.randn(2, 128)
        lookup_feat, pretrain_feat = spk_emb(spk_ids=spk_ids, spk_feat=spk_feat)
        assert lookup_feat is not None
        assert pretrain_feat is not None

    def test_combine_spk_feat_concat(self):
        spk_emb = SpeakerEmbedPrenet(
            d_model=64, spk_emb_dim_lookup=32, spk_num=5, spk_emb_comb="concat"
        )
        spk_ids = torch.randint(0, 5, (2,))
        lookup_feat, pretrain_feat = spk_emb(spk_ids=spk_ids)
        enc_output = torch.randn(2, 10, 64)
        enc_out, dec_out = spk_emb.combine_spk_feat(
            spk_feat=pretrain_feat,
            spk_feat_lookup=lookup_feat,
            enc_output=enc_output,
        )
        assert enc_out.shape == (2, 10, 64)
        assert dec_out is None

    def test_combine_spk_feat_add(self):
        spk_emb = SpeakerEmbedPrenet(
            d_model=64,
            spk_emb_dim_lookup=64,
            spk_num=5,
            spk_emb_comb="add",
        )
        spk_ids = torch.randint(0, 5, (2,))
        lookup_feat, _ = spk_emb(spk_ids=spk_ids)
        enc_output = torch.randn(2, 10, 64)
        enc_out, _ = spk_emb.combine_spk_feat(
            spk_feat=None, spk_feat_lookup=lookup_feat, enc_output=enc_output
        )
        assert enc_out.shape == (2, 10, 64)

    def test_invalid_spk_emb_comb(self):
        with pytest.raises(AssertionError):
            SpeakerEmbedPrenet(
                d_model=64, spk_emb_dim_lookup=32, spk_num=5, spk_emb_comb="invalid"
            )

    def test_neither_lookup_nor_pretrain_raises(self):
        with pytest.raises(AssertionError):
            SpeakerEmbedPrenet(d_model=64)
