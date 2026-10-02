import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")

from speechain.datasets.lombard import LombardSpeechTextDataset


class TestTrimRatios:
    def test_ratios(self):
        main_data = dict(
            text="['<space>', 'a', 'b', '<space>']", duration="[0.5, 1.0, 1.5, 1.0]"
        )
        front, tail = LombardSpeechTextDataset._trim_ratios(main_data)
        assert front == pytest.approx(0.125) and tail == pytest.approx(0.25)

    def test_no_silence(self):
        assert LombardSpeechTextDataset._trim_ratios(
            dict(text="['a', 'b']", duration="[1, 1]")
        ) == (0.0, 0.0)
        assert LombardSpeechTextDataset._trim_ratios(dict(text="abc")) == (0.0, 0.0)


class TestExtractAndCollate:
    @pytest.fixture
    def files(self, tmp_path):
        import soundfile as sf

        sr = 16000
        for name, seconds in [("tgt", 1.0), ("ref", 0.5)]:
            sf.write(
                str(tmp_path / f"{name}.wav"),
                (torch.rand(int(sr * seconds)) - 0.5).numpy(),
                sr,
            )
        idx2 = dict(
            feat=str(tmp_path / "tgt.wav"),
            feat_ref=str(tmp_path / "ref.wav"),
            text="['<space>', 'a', 'b']",
            duration="[0.5, 1.0, 0.5]",
            text_asr="a b",
            snr_cond="snr0",
        )
        for key, value in idx2.items():
            (tmp_path / f"idx2{key}").write_text(f"utt1 {value}\nutt2 {value}\n")
        return {key: str(tmp_path / f"idx2{key}") for key in idx2.keys()}

    def test_pipeline(self, files):
        dataset = LombardSpeechTextDataset(main_data=files, sample_rate=16000)
        item = dataset["utt1"]
        # the leading silence (25% of the durations) is trimmed from both waveforms
        assert item["feat"].shape[0] == 16000 - int(0.25 * 16000)
        assert item["feat_ref"].shape[0] == 8000 - int(0.25 * 8000)
        assert (
            item["text"] == ["a", "b"]
            and item["text_asr"] == "a b"
            and item["snr_cond"] == "snr0"
        )
        batch = dataset.collate_fn([dataset["utt1"], dataset["utt2"]])
        assert batch["feat_ref"].shape == (2, 6000, 1)
        assert batch["feat_ref_len"].tolist() == [6000, 6000]
        assert batch["text_asr"] == ["a b", "a b"] and batch["snr_cond"] == [
            "snr0",
            "snr0",
        ]
        assert "extra_keys" in repr(dataset)
