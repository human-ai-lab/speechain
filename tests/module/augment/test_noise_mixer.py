import pytest

torch = pytest.importorskip("torch")

from speechain.module.augment.noise_mixer import NoiseMixer


@pytest.fixture
def mixer():
    return NoiseMixer(sample_rate=16000)


def _speech(batch=2, length=16000):
    torch.manual_seed(0)
    return torch.randn(batch, length) * 0.3, torch.tensor([length, length // 2])


class TestNoiseMixer:
    def test_static_snr_is_reached(self, mixer):
        wav, wav_len = _speech()
        noisy, out_len, applied = mixer(wav, wav_len, 5.0)
        measured = NoiseMixer.measure_snr(wav, noisy, wav_len)
        assert torch.allclose(measured, torch.full((2,), 5.0), atol=0.5)
        assert torch.allclose(applied, torch.full((2,), 5.0), atol=0.5)
        assert (out_len == wav_len).all()
        # padding of the short utterance stays untouched
        assert torch.equal(noisy[1, wav_len[1] :], wav[1, wav_len[1] :])

    def test_clean_condition_is_identity(self, mixer):
        wav, wav_len = _speech()
        noisy, _, applied = mixer(wav, wav_len, [None, -10.0])
        assert torch.equal(noisy[0], wav[0])
        assert torch.isinf(applied[0]) and abs(float(applied[1]) + 10) < 0.5

    def test_dynamic_profile(self, mixer):
        wav, wav_len = _speech(batch=1)
        wav_len = torch.tensor([16000])
        noisy, _, _ = mixer(wav, wav_len, [(0.0, 20.0), (0.5, -10.0)])
        noise = noisy - wav
        first, second = noise[0, :8000].pow(2).mean(), noise[0, 8000:].pow(2).mean()
        # the second half must be 30 dB noisier than the first half
        assert abs(10 * torch.log10(second / first).item() - 30) < 1.0

    def test_three_dim_input_and_noise_file(self, tmp_path):
        import soundfile as sf

        sf.write(str(tmp_path / "hum.wav"), (torch.rand(8000) - 0.5).numpy(), 8000)
        mixer = NoiseMixer(
            sample_rate=16000, noise_files=dict(hum=str(tmp_path / "hum.wav"))
        )
        assert mixer.noise_types == ["white", "hum"]
        wav, wav_len = _speech()
        noisy, _, applied = mixer(wav.unsqueeze(-1), wav_len, 0.0, noise_type="hum")
        assert noisy.shape == (2, 16000, 1)
        assert torch.allclose(applied, torch.zeros(2), atol=0.5)

    def test_unknown_noise_type(self, mixer):
        wav, wav_len = _speech()
        with pytest.raises(AssertionError):
            mixer(wav, wav_len, 0.0, noise_type="unknown")

    def test_snr_profile_to_frames(self):
        curve = NoiseMixer.snr_profile_to_frames(
            [(0.0, 0.0), (0.25, 10.0)], 8, torch.device("cpu")
        )
        assert curve.tolist() == [0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
        assert NoiseMixer.snr_profile_to_frames(None, 8, torch.device("cpu")) is None
