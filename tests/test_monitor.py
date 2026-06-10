import pytest

try:
    import torchvision  # noqa: F401
except Exception:
    pytest.skip("torchvision not available or incompatible", allow_module_level=True)


class TestMonitorImport:
    def test_module_importable(self):
        import speechain.monitor as monitor

        assert monitor is not None

    def test_monitor_classes_exist(self):
        from speechain.monitor import Monitor, TestMonitor, TrainValidMonitor

        assert Monitor is not None
        assert TrainValidMonitor is not None
        assert TestMonitor is not None
