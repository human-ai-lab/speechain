import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")

from speechain.module.standalone.snr_predictor import SNRPredictor

FRONTEND = dict(
    type="frontend.speech2mel.Speech2MelSpec",
    conf=dict(sr=16000, hop_length=0.01, win_length=0.025, n_mels=40),
)


class TestSNRPredictor:
    def test_waveform_input_shapes(self):
        model = SNRPredictor(
            snr_classes=["clean", "0", "-10"],
            frontend=FRONTEND,
            conv_dims=[8, 8],
            emb_dim=16,
        )
        model.eval()
        wav = torch.randn(2, 8000, 1)
        wav_len = torch.tensor([8000, 4000])
        out = model(wav, wav_len)
        assert out["logits"].shape == (2, 3)
        assert out["emb"].shape == (2, 16)
        assert out["frame_logits"].shape[0] == 2 and out["frame_logits"].shape[-1] == 3
        assert out["frame_emb"].shape[-1] == 16
        assert out["feat_len"][0] > out["feat_len"][1]
        # the length tensor given by the caller must not be modified
        assert wav_len.tolist() == [8000, 4000]

    def test_feature_input(self):
        model = SNRPredictor(
            input_size=40, snr_classes=["a", "b"], conv_dims=[8], emb_dim=8
        )
        out = model(torch.randn(3, 20, 40), torch.tensor([20, 15, 10]))
        assert out["logits"].shape == (3, 2)

    def test_class_ids_and_training(self):
        model = SNRPredictor(
            input_size=40, snr_classes=["clean", "0"], conv_dims=[8], emb_dim=8
        )
        assert model.class_ids(["0", "clean"]).tolist() == [1, 0]
        out = model(torch.randn(2, 20, 40), torch.tensor([20, 20]))
        loss = torch.nn.functional.cross_entropy(out["logits"], torch.tensor([0, 1]))
        loss.backward()
        assert model.classifier.weight.grad is not None
