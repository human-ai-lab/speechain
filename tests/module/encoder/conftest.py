"""
conftest.py for speechain/module/encoder/

Adds a skip marker to test_speaker.py when torchaudio is not available
(only mocked by the parent conftest.py).  This preserves the original
"not runnable" state of test_speaker when torchaudio is absent, rather
than allowing it to collect and then fail with a misleading TypeError.
"""

import sys
from unittest.mock import MagicMock

import pytest


def pytest_collection_modifyitems(config, items):
    torchaudio_mocked = isinstance(sys.modules.get("torchaudio"), MagicMock)
    if torchaudio_mocked:
        skip_marker = pytest.mark.skip(
            reason="torchaudio not available in this environment"
        )
        for item in items:
            if "test_speaker" in str(item.fspath):
                item.add_marker(skip_marker)
