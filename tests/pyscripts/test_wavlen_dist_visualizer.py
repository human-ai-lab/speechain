import pytest


class TestWavlenDistVisualizerImport:
    def test_import(self):
        try:
            # an incompatible torch/torchvision pairing raises RuntimeError here, not ImportError
            import torchvision
        except (ImportError, OSError, AttributeError, RuntimeError):
            pytest.skip("torchvision not available")
        import speechain.pyscripts.wavlen_dist_visualizer as m

        assert m is not None
