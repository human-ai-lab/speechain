"""Tests for speechain/utilbox/log_util.py"""

import logging
import os
import sys
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")

# Mock heavy optional third-party dependency GPUtil only
try:
    import GPUtil  # noqa: F401
except (ImportError, OSError):
    sys.modules["GPUtil"] = MagicMock()

from speechain.utilbox.log_util import distributed_zero_first, logger_stdout_file


class TestLoggerStdoutFile:
    def test_returns_logger(self, tmp_path):
        logger = logger_stdout_file(str(tmp_path))
        assert isinstance(logger, logging.Logger)

    def test_creates_log_directory(self, tmp_path):
        log_dir = tmp_path / "logs"
        logger_stdout_file(str(log_dir))
        assert log_dir.exists()

    def test_with_file_name_creates_file(self, tmp_path):
        logger = logger_stdout_file(str(tmp_path), file_name="test_run")
        log_file = tmp_path / "test_run.log"
        assert log_file.exists()
        assert isinstance(logger, logging.Logger)

    def test_second_log_file_gets_suffix(self, tmp_path):
        logger_stdout_file(str(tmp_path), file_name="run")
        logger_stdout_file(str(tmp_path), file_name="run")
        files = list(tmp_path.iterdir())
        log_files = [f for f in files if f.suffix == ".log"]
        assert len(log_files) == 2

    def test_no_file_name_returns_empty_logger(self, tmp_path):
        logger = logger_stdout_file(str(tmp_path), file_name=None)
        # No file handlers when file_name is None
        assert len(logger.handlers) == 0

    def test_non_master_rank_no_file_handler(self, tmp_path):
        logger = logger_stdout_file(
            str(tmp_path), file_name="run", distributed=True, rank=1
        )
        # Non-master rank should not write to file
        assert not any(isinstance(h, logging.FileHandler) for h in logger.handlers)

    def test_logger_can_log(self, tmp_path):
        logger = logger_stdout_file(str(tmp_path), file_name="log_test")
        # Should not raise
        logger.info("test message")


class TestDistributedZeroFirst:
    def test_non_distributed_no_barrier(self):
        # Should run without error in non-distributed mode
        with distributed_zero_first(distributed=False, rank=0):
            result = 42
        assert result == 42

    def test_yields_correctly(self):
        executed = []
        with distributed_zero_first(distributed=False, rank=0):
            executed.append(1)
        assert executed == [1]

    def test_distributed_rank_zero_no_barrier_before(self):
        # In distributed=True, rank=0 should not call barrier before yield
        # (only after). We just verify it doesn't raise without actual dist.
        # Mocking torch.distributed.barrier
        original = getattr(torch, "distributed", None)
        mock_dist = MagicMock()
        torch.distributed = mock_dist

        try:
            with distributed_zero_first(distributed=True, rank=0):
                pass
            # barrier called once (after yield, for rank 0)
            assert mock_dist.barrier.call_count == 1
        finally:
            if original is not None:
                torch.distributed = original
