"""Import paths for VQC/MNIST reconstruction run."""
from __future__ import annotations

import sys
from pathlib import Path

RUN_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = RUN_ROOT.parent.parent
VENDOR_PARENT = WORKSPACE / "vendor"
QTERM = WORKSPACE / "research-artifacts" / "qfl-terminal-snapshot"
QTERM_CODE = QTERM / "code"

for p in (VENDOR_PARENT, WORKSPACE / "code", QTERM_CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

ARTIFACTS = RUN_ROOT / "artifacts"
LOGS = RUN_ROOT / "logs"
DATA = WORKSPACE / "data"
