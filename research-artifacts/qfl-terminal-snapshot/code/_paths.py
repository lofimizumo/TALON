"""Import paths for parent-repo SHARD/QFL simulator."""
from __future__ import annotations

import sys
from pathlib import Path

RUN_ROOT = Path(__file__).resolve().parents[1]
PARENT = RUN_ROOT.parent.parent  # /workspace
VENDOR = PARENT / "vendor" / "shard_sim"

if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR.parent))

ARTIFACTS = RUN_ROOT / "artifacts"
LOGS = RUN_ROOT / "logs"
ROUNDS = RUN_ROOT / "rounds"
