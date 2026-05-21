#!/usr/bin/env python3
"""Round-03 RESUME: complete L3 after root-cause fix (no LAPIN on d=784 grid).

Uses l3_budget mnist28_resume profile. Writes round03_metrics.json.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path

RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "code"))
from _paths import ARTIFACTS, LOGS, WORKSPACE  # noqa: E402
from l3_budget import estimate_joli_seconds, joli_profile  # noqa: E402

os.chdir(WORKSPACE)

_spec = importlib.util.spec_from_file_location(
    "r03", RUN_ROOT / "code" / "benchmark_round03.py"
)
r03 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(r03)

PROFILE = joli_profile(784, research_round=3)
SEEDS = r03.SEEDS
RESIZE = r03.RESIZE
DIM_G = r03.DIM_G


def setup_logging() -> logging.Logger:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOGS / "experiment_round03_resume.log", mode="w"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return logging.getLogger("r03_resume")


def main() -> None:
    logger = setup_logging()
    d = RESIZE * RESIZE
    logger.info(
        "Round-03 RESUME profile=%s adam_grid=%s n_batch_cap=%d lapin_on_d<=%s "
        "parallel_workers=%d est_joli_sec~%.0f",
        PROFILE.name,
        PROFILE.adam_grid,
        PROFILE.n_batch_cap,
        PROFILE.lapin_only_if_d_le,
        PROFILE.parallel_seed_workers,
        estimate_joli_seconds(d, DIM_G, r03.N_SAMPLES, PROFILE.adam_steps, PROFILE.n_batch_cap)
        * len(PROFILE.adam_grid),
    )
    logger.info(
        "ROOT_CAUSE: see paper/root_cause_l3_stall.md — killed stalled R03 workers 2026-05-21"
    )

    device = r03.pick_device()
    per_seed: list[dict] = []

    for seed in SEEDS:
        logger.info("=== seed %d L3 phase start ===", seed)
        rec = r03.run_single_seed(
            seed,
            logger,
            device,
            capture_grid=(seed == r03.GRID_SEED),
            l3_profile=PROFILE,
        )
        per_seed.append(rec)

    # Aggregate pass rates
    pass_cells = []
    for rec in per_seed:
        for key, cell in rec.get("l3_grid", {}).items():
            if cell.get("targets", {}).get("both_ok"):
                pass_cells.append((rec["seed"], key))

    payload = {
        "benchmark": "round03_resume",
        "root_cause_doc": "paper/root_cause_l3_stall.md",
        "l3_profile": PROFILE.__dict__,
        "per_seed": per_seed,
        "any_seed_pass_both_targets": len(pass_cells) > 0,
        "pass_cells": pass_cells,
    }
    out = ARTIFACTS / "round03_metrics.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s passes=%s", out, pass_cells)


if __name__ == "__main__":
    main()
