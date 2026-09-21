import pytest


class TestTextDistVisualizerImport:
    def test_import(self):
        pytest.importorskip("g2p_en")
        # the target module also depends on matplotlib and seaborn (via speechain.snapshooter)
        pytest.importorskip("matplotlib")
        pytest.importorskip("seaborn")
        try:
            # speechain.snapshooter transitively imports torchvision; an incompatible
            # torch/torchvision pairing raises RuntimeError (not ImportError) here
            import speechain.pyscripts.text_dist_visualizer as m
        except (ImportError, OSError, AttributeError, RuntimeError):
            pytest.skip("torchvision not available")

        assert m is not None
