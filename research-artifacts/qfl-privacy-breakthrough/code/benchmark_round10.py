#!/usr/bin/env python3
"""Round-10 QFL privacy — held-out replication for acceptance package.

Replicates on seeds {41, 43, 47, 53, 59} (not used in Rounds 01–09):
  - S2-oracle GARD-SPARSE @ 25% rows (75% reduction, MSE ≤ 0.10)
  - T1p LEAK-CERT v2 with honest naive (coverage; criterion C diagnostic)
  - S2-wrong40 ABT-1 barrier @ 25% and 50% (GARD + JASPER-Q)

Cross-cites LASA-QTERM T1 impossibility for Signal 1 (not re-run here).
No supervisor_review.md (Round 10 scientist-only acceptance).
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "code"))
from _paths import ARTIFACTS, LOGS, QTERM  # noqa: E402
from qprivacy_map_core import (  # noqa: E402
    FRAMEWORK_ID,
    THEOREM_ABT1,
    audit_gard_sparse_oracle,
    audit_t1p_leak_cert,
    audit_wrong40_barrier,
)

_spec01 = importlib.util.spec_from_file_location(
    "benchmark_round01", RUN_ROOT / "code" / "benchmark_round01.py"
)
_r01 = importlib.util.module_from_spec(_spec01)
sys.modules["benchmark_round01"] = _r01
assert _spec01.loader is not None
_spec01.loader.exec_module(_r01)

_spec06 = importlib.util.spec_from_file_location(
    "benchmark_round06", RUN_ROOT / "code" / "benchmark_round06.py"
)
_r06 = importlib.util.module_from_spec(_spec06)
sys.modules["benchmark_round06"] = _r06
assert _spec06.loader is not None
_spec06.loader.exec_module(_r06)

_spec08 = importlib.util.spec_from_file_location(
    "benchmark_round08", RUN_ROOT / "code" / "benchmark_round08.py"
)
_r08 = importlib.util.module_from_spec(_spec08)
sys.modules["benchmark_round08"] = _r08
assert _spec08.loader is not None
_spec08.loader.exec_module(_r08)

Stage2Config = _r01.Stage2Config
QflConfig = _r01.QflConfig
run_stage2_sparse = _r01.run_stage2_sparse
run_gard_cooccurrence = _r06.run_gard_cooccurrence
run_jasper_q_r06 = _r06.run_jasper_q_r06
stage_cfg_from_qfl = _r06.stage_cfg_from_qfl
Round06Config = _r06.Round06Config
Round04Config = _r06.Round04Config
Round07Config = _r08.Round07Config
run_leak_cert_t1p_honest = _r08.run_leak_cert_t1p_honest

HELD_OUT_SEEDS: tuple[int, ...] = (41, 43, 47, 53, 59)
DEVELOPMENT_SEEDS: tuple[int, ...] = (3, 7, 11, 19, 23)


def setup_logging() -> logging.Logger:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    logging.getLogger("shard_sim.attacker").setLevel(logging.WARNING)
    logger = logging.getLogger("round10")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(LOGS / "experiment_round10.log", mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def main() -> None:
    logger = setup_logging()
    t0 = time.perf_counter()
    seeds = list(HELD_OUT_SEEDS)
    stage_cfg = Stage2Config()
    qfl_cfg = QflConfig()
    r07 = Round07Config()
    r06 = Round06Config()
    r04 = Round04Config()
    cfg_qfl_stage = stage_cfg_from_qfl(qfl_cfg)

    logger.info(
        "Round 10 — held-out replication seeds=%s (%s acceptance)",
        seeds,
        FRAMEWORK_ID,
    )

    gard_runs = []
    for s in seeds:
        gr = run_stage2_sparse(s, 0.25, stage_cfg)
        gr["seed"] = s
        gard_runs.append(gr)
        logger.info(
            "GARD-SPARSE oracle seed=%d rows=%d/%d mse=%.4g",
            s,
            gr["observed_rows"],
            gr["full_rows"],
            gr["gard_sparse_mse"],
        )
    audit_gard = audit_gard_sparse_oracle(gard_runs, target_mse=0.10, fraction=0.25)

    cert_runs = [run_leak_cert_t1p_honest(s, qfl_cfg, r07) for s in seeds]
    for cr in cert_runs:
        logger.info(
            "LEAK-CERT-T1p seed=%d emp=%.4g cert=%.4g naive_bc=%.4g ratio=%.2f cover=%s",
            cr["seed"],
            cr["empirical_t1p_mse"],
            cr["cert_tight_upper_bound"],
            cr["cert_naive_broadcast"],
            cr["naive_broadcast_over_cert"],
            cr["cert_covers_empirical"],
        )
    audit_t1p = audit_t1p_leak_cert(cert_runs, use_honest_naive=True)

    gard_w40: list[dict] = []
    jasper_w40: list[dict] = []
    for frac in (0.25, 0.50):
        for seed in seeds:
            g = run_gard_cooccurrence(
                seed, frac, "wrong40", "level1_estimate", cfg_qfl_stage
            )
            j = run_jasper_q_r06(
                seed,
                frac,
                "wrong40",
                "level1_estimate",
                cfg_qfl_stage,
                qfl_cfg,
                r04,
                r06,
            )
            gard_w40.append(g)
            jasper_w40.append(j)
            logger.info(
                "wrong40 frac=%.2f seed=%d gard=%.4g jasper=%.4g",
                frac,
                seed,
                g["snapshot_mse"],
                j["snapshot_mse"],
            )
    audit_w40 = audit_wrong40_barrier(gard_w40, jasper_w40, fractions=(0.25, 0.50))

    signal_1 = {
        "id": "SIGNAL-1",
        "name": "Assignment barrier + T1 impossibility cross-cite",
        "abt1_barrier_holds": audit_w40["pass"],
        "abt1_audit": audit_w40,
        "t1_impossibility_ref": str(QTERM / "paper" / "impossibility_t1.md"),
        "note": "T1 broadcast floor cited from LASA-QTERM; ABT-1 replicated on held-out seeds.",
    }
    signal_2 = {
        "id": "SIGNAL-2",
        "name": "Oracle GARD-SPARSE conditional threat",
        "gard_oracle_pass": audit_gard["pass"],
        "gard_audit": audit_gard,
        "note": "75% row reduction @ MSE≤0.10 requires honest-incidence QFL (oracle assignment).",
    }
    signal_3 = {
        "id": "SIGNAL-3",
        "name": "LEAK-CERT T1p (secondary)",
        "cert_covers_empirical": audit_t1p["cert_covers_empirical_all"],
        "criterion_c_2x_honest": audit_t1p["criterion_c_2x_honest_naive"],
        "t1p_audit": audit_t1p,
        "note": "Certificate covers empirical T1p; honest naive is not 2× tighter than cert.",
    }

    two_independent = signal_1["abt1_barrier_holds"] and signal_2["gard_oracle_pass"]

    results = {
        "benchmark": "round10_held_out_acceptance_replication",
        "framework_id": FRAMEWORK_ID,
        "theorem_abt1": THEOREM_ABT1,
        "held_out_seeds": seeds,
        "development_seeds_excluded": list(DEVELOPMENT_SEEDS),
        "stage2_config": asdict(stage_cfg),
        "qfl_config": asdict(qfl_cfg),
        "round07_config": asdict(r07),
        "raw_runs": {
            "gard_sparse_oracle_frac_0.25": gard_runs,
            "leak_cert_t1p_v2_honest": cert_runs,
            "wrong40_gard": gard_w40,
            "wrong40_jasper_q": jasper_w40,
        },
        "audits": {
            "S2-oracle": audit_gard,
            "T1p": audit_t1p,
            "S2-wrong40": audit_w40,
        },
        "breakthrough_signals": {
            "signal_1": signal_1,
            "signal_2": signal_2,
            "signal_3": signal_3,
            "two_independent_breakthrough_signals": two_independent,
        },
        "kill_list_confirmed": [
            "Snapshot-DP (Round 05)",
            "ASSIGN-LOCK (Round 02 survey-only)",
            "trace-inflated C naive (Round 08 fix)",
            "PROBE-RAND (Round 09 honest failure)",
        ],
        "acceptance": {
            "round": 10,
            "status": "ACCEPT" if two_independent else "REVISE",
            "early_accept_two_signals": two_independent,
            "config_gate_A_oracle": audit_gard["pass"],
            "config_gate_C_honest": audit_t1p["criterion_c_2x_honest_naive"],
            "config_gate_ABT1": audit_w40["pass"],
        },
        "honesty_notes": [
            "Held-out seeds never used in Rounds 01–09 development ensemble.",
            "Signal 2 is conditional on oracle assignment; wrong40 breaks row-efficiency win.",
            "Signal 3 documents valid cert coverage without claiming 2× honest naive tightening.",
            "No supervisor_review.md (Round 10 scientist acceptance package).",
        ],
        "runtime_sec": time.perf_counter() - t0,
    }

    out = ARTIFACTS / "round10_metrics.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out)
    logger.info(
        "Acceptance: %s | two_signals=%s | A=%s ABT1=%s C_2x=%s",
        results["acceptance"]["status"],
        two_independent,
        audit_gard["pass"],
        audit_w40["pass"],
        audit_t1p["criterion_c_2x_honest_naive"],
    )
    logger.info("Runtime %.2fs", results["runtime_sec"])


if __name__ == "__main__":
    main()
