import numpy as np
import pytest


class TestFeatUtil:
    def test_preemphasize_wav(self):
        pytest.importorskip("pyworld")
        from speechain.utilbox.feat_util import preemphasize_wav

        wav = np.ones(100, dtype=np.float32)
        result = preemphasize_wav(wav, coeff=0.97)
        assert result.shape == wav.shape
        assert isinstance(result, np.ndarray)

    def test_preemphasize_wav_zero_coeff(self):
        pytest.importorskip("pyworld")
        from speechain.utilbox.feat_util import preemphasize_wav

        wav = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = preemphasize_wav(wav, coeff=0.0)
        np.testing.assert_array_almost_equal(result, wav)

    def test_feat_util_module_importable(self):
        pytest.importorskip("pyworld")
        import speechain.utilbox.feat_util as m

        assert m is not None

    def test_convert_wav_to_stft_shape(self):
        pytest.importorskip("pyworld")
        from speechain.utilbox.feat_util import convert_wav_to_stft

        wav = np.random.randn(16000).astype(np.float32)
        linear_spec = convert_wav_to_stft(wav, hop_length=160, win_length=400)
        assert linear_spec.shape == (101, 201)

    def test_convert_wav_to_logmel_shape(self):
        pytest.importorskip("pyworld")
        from speechain.utilbox.feat_util import convert_wav_to_logmel

        wav = np.random.randn(16000).astype(np.float32)
        mel_spec = convert_wav_to_logmel(
            wav, n_mels=80, hop_length=160, win_length=400
        )
        assert mel_spec.shape == (101, 80)

    def test_convert_wav_to_logmel_with_deltas_shape(self):
        pytest.importorskip("pyworld")
        from speechain.utilbox.feat_util import convert_wav_to_logmel

        wav = np.random.randn(16000).astype(np.float32)
        mel_spec = convert_wav_to_logmel(
            wav, n_mels=80, hop_length=160, win_length=400, delta_order=2
        )
        assert mel_spec.shape == (101, 240)

    def test_convert_wav_to_mfcc_shape(self):
        pytest.importorskip("pyworld")
        from speechain.utilbox.feat_util import convert_wav_to_mfcc

        wav = np.random.randn(16000).astype(np.float32)
        mfcc = convert_wav_to_mfcc(wav, hop_length=160, win_length=400, n_mfcc=20)
        assert mfcc.shape == (101, 20)

    def test_mel_filterbank_shape(self):
        from speechain.utilbox.feat_util import mel_filterbank

        mel_basis = mel_filterbank(sr=16000, n_fft=400, n_mels=80)
        assert mel_basis.shape == (80, 201)
