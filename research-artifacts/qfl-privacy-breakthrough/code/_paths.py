"""Paths to parent workspace SHARD/QFL simulator and sibling runs."""
from __future__ import annotations

import sys
from pathlib import Path

RUN_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = RUN_ROOT.parent.parent
VENDOR = WORKSPACE / "vendor" / "shard_sim"
QTERM = WORKSPACE / "research-artifacts" / "qfl-terminal-snapshot"

if str(VENDOR.parent) not in sys.path:
    sys.path.insert(0, str(VENDOR.parent))

ARTIFACTS = RUN_ROOT / "artifacts"
LOGS = RUN_ROOT / "logs"
ROUNDS = RUN_ROOT / "rounds"
