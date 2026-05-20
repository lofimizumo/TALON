"""Canonical paths for the TALON research run."""
from __future__ import annotations

from pathlib import Path

RUN_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = RUN_ROOT / "vendor"

# Optional read-only paper reference (not required to run benchmarks)
WORKSPACE_ROOT = RUN_ROOT.parent
SHARD_REPO = WORKSPACE_ROOT / "01.SHARD"
SHARD_PAPER = SHARD_REPO / "shard_neurips2026"
