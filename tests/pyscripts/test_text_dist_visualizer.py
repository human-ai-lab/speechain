import pytest


class TestTextDistVisualizerImport:
    def test_import(self):
        pytest.importorskip("g2p_en")
        import speechain.pyscripts.text_dist_visualizer as m

        assert m is not None
