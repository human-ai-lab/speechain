import numpy as np
import pytest


class TestFeatUtil:
    def test_preemphasize_wav(self):
        pytest.importorskip("librosa")
        pytest.importorskip("pyworld")
        from speechain.utilbox.feat_util import preemphasize_wav

        wav = np.ones(100, dtype=np.float32)
        result = preemphasize_wav(wav, coeff=0.97)
        assert result.shape == wav.shape
        assert isinstance(result, np.ndarray)

    def test_preemphasize_wav_zero_coeff(self):
        pytest.importorskip("librosa")
        pytest.importorskip("pyworld")
        from speechain.utilbox.feat_util import preemphasize_wav

        wav = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = preemphasize_wav(wav, coeff=0.0)
        np.testing.assert_array_almost_equal(result, wav)

    def test_feat_util_module_importable(self):
        pytest.importorskip("librosa")
        pytest.importorskip("pyworld")
        import speechain.utilbox.feat_util as m

        assert m is not None
