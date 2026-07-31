import pytest


class TestTextDistVisualizerImport:
    def test_import(self):
        pytest.importorskip("g2p_en")
        # the target module also depends on matplotlib and seaborn (via speechain.snapshooter)
        pytest.importorskip("matplotlib")
        pytest.importorskip("seaborn")
        import speechain.pyscripts.text_dist_visualizer as m

        assert m is not None
