import pytest


class TestTextDistVisualizerImport:
    def test_import(self):
        pytest.importorskip("g2p_en")
        # the target module also depends on matplotlib and seaborn (via speechain.snapshooter)
        pytest.importorskip("matplotlib")
        pytest.importorskip("seaborn")
        try:
            # speechain.snapshooter (imported by the target module) transitively imports
            # torchvision; an incompatible torch/torchvision pairing raises RuntimeError
            # (not ImportError) here. Probe torchvision directly rather than wrapping the
            # target module's own import, so a real regression there isn't silently skipped.
            import torchvision  # noqa: F401
        except (ImportError, OSError, AttributeError, RuntimeError):
            pytest.skip("torchvision not available")
        import speechain.pyscripts.text_dist_visualizer as m

        assert m is not None
