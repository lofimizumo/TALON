#!/usr/bin/env python3
"""Round-08 QFL-PRIVACY-MAP — reproducible defender/auditor audit bundle.

Integrates Rounds 01–07 into a single tier table:
  - T1 impossible (LASA-QTERM terminal-only)
  - T1p LEAK-CERT v2 with **honest** naive (broadcast mean only)
  - S2-oracle GARD-SPARSE conditional threat (75% row reduction @ MSE 0.10)
  - S2-wrong40 assignment barrier theorem ABT-1

Fixes Round 07 LEAK-CERT criterion-C gaming (trace-inflated naive baseline).
No supervisor_review artifact (Round 08 scientist-only).
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
import torch

RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "code"))
from _paths import ARTIFACTS, LOGS, QTERM, VENDOR  # noqa: E402
from qprivacy_map_core import (  # noqa: E402
    FRAMEWORK_ID,
    THEOREM_ABT1,
    audit_gard_sparse_oracle,
    audit_t1_impossibility,
    audit_t1p_leak_cert,
    audit_wrong40_barrier,
    build_privacy_map_payload,
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

_spec07 = importlib.util.spec_from_file_location(
    "benchmark_round07", RUN_ROOT / "code" / "benchmark_round07.py"
)
_r07 = importlib.util.module_from_spec(_spec07)
sys.modules["benchmark_round07"] = _r07
assert _spec07.loader is not None
_spec07.loader.exec_module(_r07)

sys.path.insert(0, str(VENDOR.parent))
sys.path.insert(0, str(QTERM / "code"))

from qterm_attack import QtermAttack, QtermConfig, QtermTier  # noqa: E402

Stage2Config = _r01.Stage2Config
QflConfig = _r01.QflConfig
run_stage2_sparse = _r01.run_stage2_sparse
hungarian_snapshot_mse = _r01.hungarian_snapshot_mse
make_epoch_batches = _r01.make_epoch_batches
simulate_gradients = _r01.simulate_gradients

run_gard_cooccurrence = _r06.run_gard_cooccurrence
run_jasper_q_r06 = _r06.run_jasper_q_r06
Round06Config = _r06.Round06Config
Round04Config = _r06.Round04Config
stage_cfg_from_qfl = _r06.stage_cfg_from_qfl

recover_t1p_multi_epoch = _r07.recover_t1p_multi_epoch
cert_t1p_trace_upper = _r07.cert_t1p_trace_upper
cert_naive_mean_only = _r07.cert_naive_mean_only
cert_naive_trace_inflated = _r07.cert_naive_trace_inflated
Round07Config = _r07.Round07Config
SurrogateQFL = _r07.SurrogateQFL


def setup_logging() -> logging.Logger:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    logging.getLogger("shard_sim.attacker").setLevel(logging.WARNING)
    logger = logging.getLogger("round08")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(LOGS / "experiment_round08.log", mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def run_t1_audit(seed: int, qfl_cfg: QflConfig) -> dict:
    """T1 impossibility: epoch-terminal-only attack vs true snapshots."""
    torch.manual_seed(seed)
    cfg_s2 = Stage2Config(
        n_samples=qfl_cfg.n_samples,
        dim_g=qfl_cfg.dim_g,
        true_rank=qfl_cfg.true_rank,
        n_epochs=qfl_cfg.n_epochs,
    )
    true_snapshots = _r01.make_low_rank_snapshots(cfg_s2, seed)
    surrogate = SurrogateQFL(
        input_dim=64,
        dim_g=qfl_cfg.dim_g,
        n_params=qfl_cfg.dim_g,
        noise_level=qfl_cfg.noise_level,
        seed=seed,
    )
    coeff: list[np.ndarray] = []
    terminal_grads: list[list[np.ndarray]] = []
    for e in range(qfl_cfg.n_epochs):
        batches = make_epoch_batches(qfl_cfg.n_samples, qfl_cfg.batch_size, seed + e)
        a_e, _ = simulate_gradients(true_snapshots, batches, surrogate)
        coeff.append(a_e)
        mean_snap = true_snapshots.mean(axis=0)
        g_term = a_e @ mean_snap
        if surrogate.noise_level > 0:
            g_term = g_term + surrogate._np_rng.normal(0, surrogate.noise_level, size=g_term.shape)
        terminal_grads.append([g_term])

    e_bar = np.mean([g for epoch in terminal_grads for g in epoch], axis=0)
    qterm_t1 = QtermAttack(
        QtermConfig(
            tier=QtermTier.T1,
            n_samples=qfl_cfg.n_samples,
            batch_size=qfl_cfg.batch_size,
            random_seed=seed,
        )
    )
    r1 = qterm_t1.recover(e_bar, coeff, batch_gradients=terminal_grads)
    mse = hungarian_snapshot_mse(r1.snapshots, true_snapshots)
    return {"seed": seed, "t1_mse": mse, "n_terminal_epochs": qfl_cfg.n_epochs}


def run_leak_cert_t1p_honest(seed: int, qfl_cfg: QflConfig, r07: Round07Config) -> dict:
    """LEAK-CERT-T1p v2 with honest naive (broadcast mean only) for criterion C."""
    true_snapshots, _, batch_grads = recover_t1p_multi_epoch(seed, qfl_cfg, r07)
    surrogate = SurrogateQFL(
        input_dim=64,
        dim_g=qfl_cfg.dim_g,
        n_params=qfl_cfg.dim_g,
        noise_level=qfl_cfg.noise_level,
        seed=seed,
    )
    coeff: list[np.ndarray] = []
    for e in range(qfl_cfg.n_epochs):
        batches = make_epoch_batches(qfl_cfg.n_samples, qfl_cfg.batch_size, seed + e)
        a_e, _ = simulate_gradients(true_snapshots, batches, surrogate)
        coeff.append(a_e)
    e_bar = np.mean([g for grads_e in batch_grads for g in grads_e], axis=0)

    qterm_t1p = QtermAttack(
        QtermConfig(
            tier=QtermTier.T1P,
            n_samples=qfl_cfg.n_samples,
            batch_size=qfl_cfg.batch_size,
            partial_rows_per_epoch=qfl_cfg.partial_rows,
            random_seed=seed,
        )
    )
    r1p = qterm_t1p.recover(e_bar, coeff, batch_gradients=batch_grads)
    empirical = hungarian_snapshot_mse(r1p.snapshots, true_snapshots)

    naive_broadcast = cert_naive_mean_only(e_bar, true_snapshots)
    naive_trace = cert_naive_trace_inflated(e_bar, true_snapshots, qfl_cfg.true_rank)
    k_batches = qfl_cfg.n_samples // qfl_cfg.batch_size
    cert_trace = cert_t1p_trace_upper(
        true_snapshots,
        e_bar,
        qfl_cfg.true_rank,
        r07.t1p_rows,
        r07.t1p_partial_per_epoch,
        k_batches,
        qfl_cfg.batch_size,
        r07.cert_noise_std,
        r07,
    )
    cert_tight = cert_trace
    ratio_broadcast = naive_broadcast / max(cert_tight, 1e-12)
    ratio_trace = naive_trace / max(cert_tight, 1e-12)

    return {
        "seed": seed,
        "empirical_t1p_mse": empirical,
        "n_obs_t1p_rows": r07.t1p_rows,
        "cert_naive_broadcast": naive_broadcast,
        "cert_naive_trace_inflated": naive_trace,
        "cert_t1p_trace_upper": cert_trace,
        "cert_tight_upper_bound": cert_tight,
        "naive_broadcast_over_cert": ratio_broadcast,
        "naive_trace_over_cert": ratio_trace,
        "cert_covers_empirical": cert_tight >= empirical * 0.95,
        "criterion_c_2x_honest_naive": ratio_broadcast >= 2.0,
        "criterion_c_2x_trace_inflated": ratio_trace >= 2.0,
    }


def main() -> None:
    logger = setup_logging()
    t0 = time.perf_counter()
    seeds = [3, 7, 11, 19, 23]
    stage_cfg = Stage2Config()
    qfl_cfg = QflConfig()
    r07 = Round07Config()
    r06 = Round06Config()
    r04 = Round04Config()

    logger.info("Round 08 — %s defender/auditor audit bundle", FRAMEWORK_ID)

    # --- T1 ---
    t1_runs = [run_t1_audit(s, qfl_cfg) for s in seeds]
    t1_mean = float(np.mean([r["t1_mse"] for r in t1_runs]))
    for r in t1_runs:
        logger.info("T1 seed=%d mse=%.4g", r["seed"], r["t1_mse"])
    audit_t1 = audit_t1_impossibility(t1_mean)
    logger.info("T1 audit: %s", audit_t1["verdict"])

    # --- T1p LEAK-CERT (honest naive) ---
    cert_runs = [run_leak_cert_t1p_honest(s, qfl_cfg, r07) for s in seeds]
    for cr in cert_runs:
        logger.info(
            "LEAK-CERT-T1p v2 seed=%d emp=%.4g cert=%.4g naive_bc=%.4g ratio_bc=%.2f "
            "naive_tr=%.4g ratio_tr=%.2f cover=%s honest_2x=%s",
            cr["seed"],
            cr["empirical_t1p_mse"],
            cr["cert_tight_upper_bound"],
            cr["cert_naive_broadcast"],
            cr["naive_broadcast_over_cert"],
            cr["cert_naive_trace_inflated"],
            cr["naive_trace_over_cert"],
            cr["cert_covers_empirical"],
            cr["criterion_c_2x_honest_naive"],
        )
    audit_t1p = audit_t1p_leak_cert(cert_runs, use_honest_naive=True)
    logger.info("T1p audit: %s", audit_t1p["verdict"])

    # --- S2-oracle GARD-SPARSE @ 25% (75% reduction) ---
    gard_runs = []
    for s in seeds:
        gr = run_stage2_sparse(s, 0.25, stage_cfg)
        gr["seed"] = s
        gard_runs.append(gr)
        logger.info(
            "GARD-SPARSE oracle seed=%d rows=%d mse=%.4g",
            s,
            gr["observed_rows"],
            gr["gard_sparse_mse"],
        )
    audit_gard = audit_gard_sparse_oracle(gard_runs, target_mse=0.10, fraction=0.25)
    logger.info("S2-oracle audit: %s", audit_gard["verdict"])

    # --- S2-wrong40 ABT-1 @ 25% and 50% ---
    cfg_qfl_stage = stage_cfg_from_qfl(qfl_cfg)
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
    logger.info("S2-wrong40 audit: %s", audit_w40["verdict"])

    audits = {
        "T1": audit_t1,
        "T1p": audit_t1p,
        "S2-oracle": audit_gard,
        "S2-wrong40": audit_w40,
    }
    privacy_map = build_privacy_map_payload(audits)

    results = {
        "benchmark": "round08_qfl_privacy_map_audit_bundle",
        "framework_id": FRAMEWORK_ID,
        "theorem_abt1": THEOREM_ABT1,
        "theorem_abt1_path": str(RUN_ROOT / "paper" / "assignment_barrier_theorem.md"),
        "privacy_map_path": str(RUN_ROOT / "paper" / "privacy_map.md"),
        "seeds": seeds,
        "stage2_config": asdict(stage_cfg),
        "qfl_config": asdict(qfl_cfg),
        "round07_config": asdict(r07),
        "raw_runs": {
            "t1": t1_runs,
            "leak_cert_t1p_v2_honest": cert_runs,
            "gard_sparse_oracle_frac_0.25": gard_runs,
            "wrong40_gard": gard_w40,
            "wrong40_jasper_q": jasper_w40,
        },
        "privacy_map": privacy_map,
        "leak_cert_fix": {
            "round07_issue": "trace-inflated naive baseline gamed criterion C (~14.8×)",
            "round08_fix": "criterion C uses cert_naive_broadcast (mean-only) only",
            "trace_inflated_retained_for_audit": True,
        },
        "honesty_notes": [
            "QFL-PRIVACY-MAP: defender chooses publish tier; auditor runs max attack for that tier.",
            "T1 impossibility cites qfl-terminal-snapshot; not relitigated here.",
            "GARD-SPARSE 75% reduction requires oracle assignment + chain graph (Round 01).",
            "ABT-1 wrong40 barrier: proof sketch in paper/assignment_barrier_theorem.md.",
            "LEAK-CERT v2 cert covers empirical T1p; honest 2× criterion C fails (naive≈cert).",
            "No supervisor_review.md for Round 08 (scientist-only integration).",
        ],
        "runtime_sec": time.perf_counter() - t0,
    }

    out = ARTIFACTS / "round08_metrics.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out)
    logger.info(
        "Gates: T1=%s A_oracle=%s C_honest=%s ABT1=%s",
        privacy_map["config_breakthrough"]["T1_impossibility_confirmed"],
        privacy_map["config_breakthrough"]["A_gard_oracle_75pct_reduction_at_mse_0.10"],
        privacy_map["config_breakthrough"]["C_leak_cert_2x_honest_naive"],
        privacy_map["config_breakthrough"]["ABT1_wrong40_barrier"],
    )
    logger.info("Runtime %.2fs", results["runtime_sec"])


if __name__ == "__main__":
    main()
