import pytest

torch = pytest.importorskip("torch")
try:
    import torchvision  # noqa: F401
except Exception:
    pytest.skip("torchvision not available or incompatible", allow_module_level=True)


class TestSnapshooterImport:
    def test_module_importable(self):
        import speechain.snapshooter as snap

        assert snap is not None

    def test_plotter_classes_exist(self):
        from speechain.snapshooter import CurvePlotter, HistPlotter, MatrixPlotter

        assert CurvePlotter is not None
        assert MatrixPlotter is not None
        assert HistPlotter is not None

    def test_snapshooter_class_exists(self):
        from speechain.snapshooter import SnapShooter

        assert SnapShooter is not None


class TestCurvePlotter:
    def test_instantiation_defaults(self):
        from speechain.snapshooter import CurvePlotter

        plotter = CurvePlotter()
        assert plotter is not None
        assert "linestyle" in plotter.plot_conf

    def test_instantiation_custom_conf(self):
        from speechain.snapshooter import CurvePlotter

        plotter = CurvePlotter(plot_conf={"linewidth": 2})
        assert plotter.plot_conf["linewidth"] == 2

    def test_grid_conf_defaults(self):
        from speechain.snapshooter import CurvePlotter

        plotter = CurvePlotter()
        assert "linestyle" in plotter.grid_conf


class TestMatrixPlotter:
    def test_instantiation(self):
        from speechain.snapshooter import MatrixPlotter

        plotter = MatrixPlotter()
        assert plotter is not None


class TestHistPlotter:
    def test_instantiation(self):
        from speechain.snapshooter import HistPlotter

        plotter = HistPlotter()
        assert plotter is not None
