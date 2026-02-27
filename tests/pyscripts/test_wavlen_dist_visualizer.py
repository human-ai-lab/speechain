import pytest


class TestWavlenDistVisualizerImport:
    def test_import(self):
        try:
            import torchvision
        except (ImportError, OSError, AttributeError):
            pytest.skip("torchvision not available")
        import speechain.pyscripts.wavlen_dist_visualizer as m

        assert m is not None
