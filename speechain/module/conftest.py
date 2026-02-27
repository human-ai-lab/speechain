"""
conftest.py for speechain/module/

Pre-mocks unavailable optional dependencies (e.g., torchaudio, GPUtil) so that
sub-packages whose __init__.py references those libs can still be imported
in CPU-only CI environments.  Tests that actually exercise torchaudio
functionality use their own HAS_TORCHAUDIO guard and pytestmark.skipif.
"""

import sys
from unittest.mock import MagicMock


def _mock_lib(name: str) -> None:
    """Insert a MagicMock for *name* and ensure parent packages know about it."""
    mock = MagicMock()
    sys.modules.setdefault(name, mock)
    parts = name.split(".")
    if len(parts) > 1:
        parent_name = ".".join(parts[:-1])
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, parts[-1], mock)


try:
    import torchaudio  # noqa: F401
except (ImportError, OSError):
    for _sub in [
        "torchaudio",
        "torchaudio.functional",
        "torchaudio.transforms",
        "torchaudio.pipelines",
        "torchaudio._extension",
    ]:
        _mock_lib(_sub)

try:
    import GPUtil  # noqa: F401
except (ImportError, OSError):
    _mock_lib("GPUtil")
