import pytest


class TestPhnDurationVisualizerImport:
    def test_import(self):
        try:
            import torchvision
        except (ImportError, OSError, AttributeError):
            pytest.skip("torchvision not available")
        import speechain.pyscripts.phn_duaration_visualizer as m

        assert m is not None
