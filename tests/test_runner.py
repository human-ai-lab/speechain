import pytest

try:
    import torchvision  # noqa: F401
except Exception:
    pytest.skip("torchvision not available or incompatible", allow_module_level=True)


class TestRunnerImport:
    def test_module_importable(self):
        import speechain.runner as runner

        assert runner is not None

    def test_runner_class_exists(self):
        from speechain.runner import Runner

        assert Runner is not None
