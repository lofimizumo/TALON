#!/usr/bin/env python3
"""Round-03 QFL privacy benchmark — break assignment barrier.

Implements (Round 03 mission):
  1. INCIDENCE-REFINE — iterative hard incidence correction from batch means
  2. HYBRID — GARD sparse Stage-2 + LASA-QTERM T1p (terminal + intermediate rows)
  3. DP-mean anchor — Laplace-noised Level-1 mean vs oracle anchor
  4. Co-occurrence graph — explicit stress under wrong assignment vs oracle chain

Reuses Round-02 simulator, parent-aligned threat reduction, and honest criterion gates.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "code"))
from _paths import ARTIFACTS, LOGS, QTERM, VENDOR  # noqa: E402

# Load round02 as a module (shared Stage-2 + QFL helpers)
_spec = importlib.util.spec_from_file_location(
    "benchmark_round02", RUN_ROOT / "code" / "benchmark_round02.py"
)
_r02 = importlib.util.module_from_spec(_spec)
sys.modules["benchmark_round02"] = _r02
assert _spec.loader is not None
_spec.loader.exec_module(_r02)

sys.path.insert(0, str(VENDOR.parent))
sys.path.insert(0, str(QTERM / "code"))

from shard_sim.surrogate_model import SurrogateQFL  # noqa: E402
from qterm_attack import QtermAttack, QtermConfig, QtermTier  # noqa: E402

MSE_TARGETS = _r02.MSE_TARGETS
CRITERION_A_GATE = _r02.CRITERION_A_GATE
Stage2Config = _r02.Stage2Config
QflConfig = _r02.QflConfig

make_low_rank_snapshots = _r02.make_low_rank_snapshots
make_incidence = _r02.make_incidence
select_rows = _r02.select_rows
corrupt_incidence = _r02.corrupt_incidence
split_train_val = _r02.split_train_val
cooccurrence_laplacian = _r02.cooccurrence_laplacian
chain_laplacian = _r02.chain_laplacian
solve_map = _r02.solve_map
select_graph_map = _r02.select_graph_map
snapshot_mse = _r02.snapshot_mse
threat_reduction_tables = _r02.threat_reduction_tables
row_members = _r02.row_members
row_from_members = _r02.row_from_members
resolve_mean_anchor = _r02.resolve_mean_anchor
anchor_weight_for_mode = _r02.anchor_weight_for_mode
hungarian_snapshot_mse = _r02.hungarian_snapshot_mse
make_epoch_batches = _r02.make_epoch_batches
simulate_gradients = _r02.simulate_gradients


@dataclass(frozen=True)
class Round03Config:
    incidence_refine_iters: int = 10
    incidence_refine_graph_lambda: float = 0.1
    dp_epsilon: float = 1.0
    dp_l2_sensitivity: float = 0.15
    hybrid_sparse_fraction: float = 0.25
    hybrid_blend_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


def setup_logging() -> logging.Logger:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    logging.getLogger("shard_sim.attacker").setLevel(logging.WARNING)
    logger = logging.getLogger("round03")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(LOGS / "experiment_round03.log", mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def assignment_overlap(h_assumed: np.ndarray, h_true: np.ndarray, batch_size: int) -> float:
    overlaps = []
    for a, t in zip(h_assumed, h_true):
        pa = set(row_members(a).tolist())
        pt = set(row_members(t).tolist())
        overlaps.append(len(pa.intersection(pt)) / batch_size)
    return float(np.mean(overlaps))


def dp_mean_anchor(
    m_obs: np.ndarray,
    cfg: Stage2Config,
    seed: int,
    r03: Round03Config,
) -> np.ndarray:
    """Laplace mechanism on Level-1 batch-mean average (no snapshot oracle)."""
    mu = m_obs.mean(axis=0)
    rng = np.random.default_rng(seed + 88_000)
    scale = r03.dp_l2_sensitivity / max(r03.dp_epsilon, 1e-6)
    noise = rng.laplace(0.0, scale / np.sqrt(cfg.dim_g), size=(cfg.dim_g,))
    return mu + noise


def refine_incidence_hard(
    h_init: np.ndarray,
    m_obs: np.ndarray,
    mean_snapshot: np.ndarray,
    cfg: Stage2Config,
    anchor_weight: float,
    r03: Round03Config,
    *,
    use_cooccurrence_graph: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Alternate snapshot MAP solve and per-row greedy membership relabeling."""
    h = h_init.copy()
    s = solve_map(h, m_obs, mean_snapshot, cfg, anchor_weight=anchor_weight)
    for _ in range(r03.incidence_refine_iters):
        if use_cooccurrence_graph:
            lap = cooccurrence_laplacian(h)
            s = solve_map(
                h,
                m_obs,
                mean_snapshot,
                cfg,
                anchor_weight=anchor_weight,
                graph_lambda=r03.incidence_refine_graph_lambda,
                laplacian=lap,
            )
        else:
            s = solve_map(h, m_obs, mean_snapshot, cfg, anchor_weight=anchor_weight)

        rows = []
        for i in range(h.shape[0]):
            scores = s @ m_obs[i]
            members = np.argsort(scores)[-cfg.batch_size :]
            rows.append(row_from_members(members, cfg.n_samples))
        h = np.vstack(rows)
    if use_cooccurrence_graph:
        lap = cooccurrence_laplacian(h)
        s = solve_map(
            h,
            m_obs,
            mean_snapshot,
            cfg,
            anchor_weight=anchor_weight,
            graph_lambda=r03.incidence_refine_graph_lambda,
            laplacian=lap,
        )
    else:
        s = solve_map(h, m_obs, mean_snapshot, cfg, anchor_weight=anchor_weight)
    return s, h


def run_assignment_barrier_stage2(
    seed: int,
    fraction: float,
    assignment_regime: str,
    mean_anchor_mode: str,
    cfg: Stage2Config,
    r03: Round03Config,
) -> dict:
    s_true = make_low_rank_snapshots(cfg, seed)
    h_full, _meta = make_incidence(cfg, seed)
    keep = select_rows(h_full, fraction, seed)
    h_true = h_full[keep]
    rng = np.random.default_rng(seed + 50_000 + len(keep))
    m_obs = h_true @ s_true + cfg.noise_std * rng.normal(size=(len(keep), cfg.dim_g))
    h_assumed, prior_overlap = corrupt_incidence(h_true, cfg, assignment_regime, seed)

    if mean_anchor_mode == "dp_mean":
        mean_snapshot = dp_mean_anchor(m_obs, cfg, seed, r03)
        anchor_weight = cfg.mean_anchor_weight
    else:
        mean_snapshot = resolve_mean_anchor(mean_anchor_mode, s_true, h_assumed, m_obs, cfg, seed)
        anchor_weight = anchor_weight_for_mode(mean_anchor_mode, cfg)

    train_idx, val_idx = split_train_val(len(keep), cfg, seed)
    h_train, m_train = h_assumed[train_idx], m_obs[train_idx]
    h_val, m_val = h_assumed[val_idx], m_obs[val_idx]
    rank_aug = int(
        np.linalg.matrix_rank(
            np.vstack([h_assumed, np.ones((1, cfg.n_samples), dtype=np.float64) / cfg.n_samples])
        )
    )

    methods: dict[str, float] = {}

    s_ls = solve_map(h_assumed, m_obs, mean_snapshot, cfg, anchor_weight=anchor_weight)
    methods["shard_style_ls_wrong_h"] = snapshot_mse(s_ls, s_true)

    lap_co = cooccurrence_laplacian(h_assumed)
    s_co, _ = select_graph_map(
        h_train,
        m_train,
        h_val,
        m_val,
        h_assumed,
        m_obs,
        mean_snapshot,
        lap_co,
        cfg,
        anchor_weight,
        rank_aug,
    )
    methods["gard_cooccurrence_wrong_h"] = snapshot_mse(s_co, s_true)

    lap_or = chain_laplacian(cfg.n_samples)
    s_or, _ = select_graph_map(
        h_train,
        m_train,
        h_val,
        m_val,
        h_assumed,
        m_obs,
        mean_snapshot,
        lap_or,
        cfg,
        anchor_weight,
        rank_aug,
    )
    methods["gard_oracle_chain_wrong_h"] = snapshot_mse(s_or, s_true)

    s_ref, h_ref = refine_incidence_hard(
        h_assumed,
        m_obs,
        mean_snapshot,
        cfg,
        anchor_weight,
        r03,
        use_cooccurrence_graph=True,
    )
    methods["incidence_refine_cooccurrence"] = snapshot_mse(s_ref, s_true)
    methods["incidence_refine_overlap"] = assignment_overlap(h_ref, h_true, cfg.batch_size)

    # Oracle-assignment upper bound (true H; same anchor vector as experiment)
    if mean_anchor_mode == "dp_mean":
        mean_ub = mean_snapshot
        ub_anchor_w = cfg.mean_anchor_weight
    else:
        mean_ub = resolve_mean_anchor("oracle_true", s_true, h_true, m_obs, cfg, seed)
        ub_anchor_w = anchor_weight_for_mode("oracle_true", cfg)
    s_ub, _ = select_graph_map(
        h_train,
        m_train,
        h_val,
        m_val,
        h_true,
        m_obs,
        mean_ub,
        lap_or,
        cfg,
        ub_anchor_w,
        rank_aug,
    )
    methods["gard_oracle_h_upper_bound"] = snapshot_mse(s_ub, s_true)

    return {
        "seed": seed,
        "fraction": fraction,
        "observed_rows": int(len(keep)),
        "full_rows": int(h_full.shape[0]),
        "assignment_regime": assignment_regime,
        "assignment_overlap_prior": prior_overlap,
        "mean_anchor_mode": mean_anchor_mode,
        "methods": methods,
    }


def build_sparse_incidence_from_batches(
    snapshots: np.ndarray,
    batches_per_epoch: list[list[list[int]]],
    cfg: Stage2Config,
    seed: int,
    fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Stage-2 rows from simulated minibatches (true membership)."""
    rows = []
    for epoch_batches in batches_per_epoch:
        for members in epoch_batches:
            row = np.zeros(cfg.n_samples, dtype=np.float64)
            row[members] = 1.0 / len(members)
            rows.append(row)
    h_full = np.vstack(rows)
    keep = select_rows(h_full, fraction, seed)
    h = h_full[keep]
    rng = np.random.default_rng(seed + 61_000)
    m = h @ snapshots + cfg.noise_std * rng.normal(size=(len(keep), cfg.dim_g))
    return h, m


def run_hybrid_t1p_gard(
    seed: int,
    assignment_regime: str,
    cfg: Stage2Config,
    qfl_cfg: QflConfig,
    r03: Round03Config,
) -> dict:
    """Fuse LASA-QTERM T1p (terminal partial rows) with sparse Stage-2 GARD."""
    torch.manual_seed(seed)
    n = qfl_cfg.n_samples
    stage_cfg = Stage2Config(
        n_samples=n,
        batch_size=qfl_cfg.batch_size,
        dim_g=qfl_cfg.dim_g,
        true_rank=qfl_cfg.true_rank,
        n_epochs=qfl_cfg.n_epochs,
        noise_std=qfl_cfg.noise_level,
    )
    snapshots = make_low_rank_snapshots(stage_cfg, seed)
    surrogate = SurrogateQFL(
        input_dim=64,
        dim_g=qfl_cfg.dim_g,
        n_params=qfl_cfg.dim_g,
        noise_level=qfl_cfg.noise_level,
        seed=seed,
    )

    coeff: list[np.ndarray] = []
    batch_grads: list[list[np.ndarray]] = []
    batches_per_epoch: list[list[list[int]]] = []
    for e in range(qfl_cfg.n_epochs):
        batches = make_epoch_batches(n, qfl_cfg.batch_size, seed + e)
        batches_per_epoch.append(batches)
        a_e, grads_e = simulate_gradients(snapshots, batches, surrogate)
        coeff.append(a_e)
        batch_grads.append(grads_e)

    e_bar = np.mean(
        [g for grads_e in batch_grads for g in grads_e],
        axis=0,
    )
    qterm = QtermAttack(
        QtermConfig(
            tier=QtermTier.T1P,
            n_samples=n,
            batch_size=qfl_cfg.batch_size,
            partial_rows_per_epoch=qfl_cfg.partial_rows,
            random_seed=seed,
        )
    )
    t1p = qterm.recover(e_bar, coeff, batch_gradients=batch_grads)
    s_t1p = t1p.snapshots
    t1p_mse = hungarian_snapshot_mse(s_t1p, snapshots)

    h_sparse_true, m_sparse = build_sparse_incidence_from_batches(
        snapshots,
        batches_per_epoch,
        stage_cfg,
        seed,
        r03.hybrid_sparse_fraction,
    )
    h_sparse_assumed, overlap = corrupt_incidence(
        h_sparse_true, stage_cfg, assignment_regime, seed
    )
    mean_l1 = m_sparse.mean(axis=0)
    anchor_w = stage_cfg.mean_anchor_weight
    lap_co = cooccurrence_laplacian(h_sparse_assumed)
    train_idx, val_idx = split_train_val(h_sparse_assumed.shape[0], stage_cfg, seed)
    s_gard, _ = select_graph_map(
        h_sparse_assumed[train_idx],
        m_sparse[train_idx],
        h_sparse_assumed[val_idx],
        m_sparse[val_idx],
        h_sparse_assumed,
        m_sparse,
        mean_l1,
        lap_co,
        stage_cfg,
        anchor_w,
        n,
    )
    gard_mse = hungarian_snapshot_mse(s_gard, snapshots)

    best_blend = 0.0
    best_hybrid_mse = float("inf")
    for alpha in r03.hybrid_blend_grid:
        blended = alpha * s_t1p + (1.0 - alpha) * s_gard
        mse = hungarian_snapshot_mse(blended, snapshots)
        if mse < best_hybrid_mse:
            best_hybrid_mse = mse
            best_blend = alpha

    # GARD with T1p mean as anchor (terminal informs sparse solve)
    mean_t1p = s_t1p.mean(axis=0)
    s_gard_t1p_anchor, _ = select_graph_map(
        h_sparse_assumed[train_idx],
        m_sparse[train_idx],
        h_sparse_assumed[val_idx],
        m_sparse[val_idx],
        h_sparse_assumed,
        m_sparse,
        mean_t1p,
        lap_co,
        stage_cfg,
        anchor_w,
        n,
    )
    gard_t1p_anchor_mse = hungarian_snapshot_mse(s_gard_t1p_anchor, snapshots)

    return {
        "seed": seed,
        "assignment_regime": assignment_regime,
        "t1p_partial_rows_per_epoch": qfl_cfg.partial_rows,
        "sparse_fraction": r03.hybrid_sparse_fraction,
        "sparse_observed_rows": int(h_sparse_true.shape[0]),
        "assignment_overlap_sparse": overlap,
        "t1p_mse": t1p_mse,
        "gard_sparse_cooccurrence_mse": gard_mse,
        "hybrid_best_blend": best_blend,
        "hybrid_best_mse": best_hybrid_mse,
        "gard_t1p_mean_anchor_mse": gard_t1p_anchor_mse,
    }


def aggregate_barrier_runs(runs: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for run in runs:
        key = (run["fraction"], run["assignment_regime"], run["mean_anchor_mode"])
        groups.setdefault(key, []).append(run)
    out = []
    for key, items in sorted(groups.items()):
        frac, assign, anchor = key
        for method in items[0]["methods"]:
            mse = np.array([r["methods"][method] for r in items], dtype=np.float64)
            out.append(
                {
                    "fraction": frac,
                    "assignment_regime": assign,
                    "mean_anchor_mode": anchor,
                    "method": method,
                    "snapshot_mse_mean": float(mse.mean()),
                    "snapshot_mse_std": float(mse.std(ddof=0)),
                    "n_seeds": len(items),
                }
            )
    return out


def aggregate_hybrid(runs: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for run in runs:
        groups.setdefault(run["assignment_regime"], []).append(run)
    out = []
    for regime, items in sorted(groups.items()):
        for field in (
            "t1p_mse",
            "gard_sparse_cooccurrence_mse",
            "hybrid_best_mse",
            "gard_t1p_mean_anchor_mse",
        ):
            vals = np.array([r[field] for r in items], dtype=np.float64)
            out.append(
                {
                    "assignment_regime": regime,
                    "metric": field,
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=0)),
                    "n_seeds": len(items),
                }
            )
        blends = [r["hybrid_best_blend"] for r in items]
        out.append(
            {
                "assignment_regime": regime,
                "metric": "hybrid_best_blend_mean",
                "mean": float(np.mean(blends)),
                "std": float(np.std(blends)),
                "n_seeds": len(items),
            }
        )
    return out


def barrier_threat_tables(
    runs: list[dict],
    cfg: Stage2Config,
    *,
    assignment: str,
    mean_anchor_mode: str,
    method: str,
) -> dict:
    """Adapt round02 threat_reduction to round03 run dict shape."""
    adapted = [
        {
            "seed": r["seed"],
            "fraction": r["fraction"],
            "observed_rows": r["observed_rows"],
            "full_rows": r["full_rows"],
            "assignment_regime": r["assignment_regime"],
            "mean_anchor_mode": r["mean_anchor_mode"],
            "methods": {method: r["methods"][method]},
        }
        for r in runs
        if r["assignment_regime"] == assignment and r["mean_anchor_mode"] == mean_anchor_mode
    ]
    return threat_reduction_tables(
        adapted, cfg, assignment=assignment, mean_anchor_mode=mean_anchor_mode, method=method
    )


def cooccurrence_vs_oracle_summary(aggregate: list[dict]) -> list[dict]:
    """Delta MSE: co-occurrence minus oracle chain, grouped by assignment."""
    rows = []
    for assign in ("wrong20", "wrong40", "unknown_random"):
        co = next(
            (
                a
                for a in aggregate
                if a["assignment_regime"] == assign
                and a["method"] == "gard_cooccurrence_wrong_h"
                and a["fraction"] == 0.25
            ),
            None,
        )
        orc = next(
            (
                a
                for a in aggregate
                if a["assignment_regime"] == assign
                and a["method"] == "gard_oracle_chain_wrong_h"
                and a["fraction"] == 0.25
            ),
            None,
        )
        ref = next(
            (
                a
                for a in aggregate
                if a["assignment_regime"] == assign
                and a["method"] == "incidence_refine_cooccurrence"
                and a["fraction"] == 0.25
            ),
            None,
        )
        if co and orc:
            rows.append(
                {
                    "assignment_regime": assign,
                    "fraction": 0.25,
                    "cooccurrence_mse": co["snapshot_mse_mean"],
                    "oracle_chain_mse": orc["snapshot_mse_mean"],
                    "cooccurrence_minus_oracle": co["snapshot_mse_mean"]
                    - orc["snapshot_mse_mean"],
                    "incidence_refine_mse": ref["snapshot_mse_mean"] if ref else None,
                }
            )
    return rows


def main() -> None:
    logger = setup_logging()
    t0 = time.perf_counter()
    cfg = Stage2Config()
    qfl_cfg = QflConfig()
    r03 = Round03Config()
    seeds = [3, 7, 11, 19, 23]
    assignments_barrier = ["wrong20", "wrong40", "unknown_random"]
    fractions_barrier = (0.25, 0.15)
    anchor_modes = ["level1_estimate", "dp_mean", "oracle_true"]

    logger.info("Round 03 assignment-barrier stress")
    barrier_runs: list[dict] = []
    for frac in fractions_barrier:
        for assign in assignments_barrier:
            for anchor in anchor_modes:
                for seed in seeds:
                    run = run_assignment_barrier_stage2(
                        seed, frac, assign, anchor, cfg, r03
                    )
                    barrier_runs.append(run)
                    logger.info(
                        "barrier seed=%d frac=%.2f %s anchor=%s refine=%.4g co=%.4g",
                        seed,
                        frac,
                        assign,
                        anchor,
                        run["methods"]["incidence_refine_cooccurrence"],
                        run["methods"]["gard_cooccurrence_wrong_h"],
                    )

    barrier_agg = aggregate_barrier_runs(barrier_runs)
    co_vs_oracle = cooccurrence_vs_oracle_summary(barrier_agg)

    # Threat reduction @ parent gate for key methods (wrong40, level1 + dp)
    threat_wrong40_refine = barrier_threat_tables(
        barrier_runs,
        cfg,
        assignment="wrong40",
        mean_anchor_mode="level1_estimate",
        method="incidence_refine_cooccurrence",
    )
    threat_wrong40_co = barrier_threat_tables(
        barrier_runs,
        cfg,
        assignment="wrong40",
        mean_anchor_mode="level1_estimate",
        method="gard_cooccurrence_wrong_h",
    )
    # Oracle-assignment replication for criterion A (level1 + cooccurrence)
    oracle_runs: list[dict] = []
    for frac in cfg.random_fractions:
        for seed in seeds:
            oracle_runs.append(
                run_assignment_barrier_stage2(
                    seed, frac, "oracle", "level1_estimate", cfg, r03
                )
            )
    threat_oracle_level1_co = barrier_threat_tables(
        oracle_runs,
        cfg,
        assignment="oracle",
        mean_anchor_mode="level1_estimate",
        method="gard_cooccurrence_wrong_h",
    )

    # Hybrid T1p + sparse GARD
    hybrid_runs = []
    for assign in ("oracle", "wrong20", "wrong40"):
        for seed in seeds:
            hybrid_runs.append(run_hybrid_t1p_gard(seed, assign, cfg, qfl_cfg, r03))
            logger.info(
                "hybrid seed=%d %s t1p=%.4g gard=%.4g best=%.4g blend=%.2f",
                seed,
                assign,
                hybrid_runs[-1]["t1p_mse"],
                hybrid_runs[-1]["gard_sparse_cooccurrence_mse"],
                hybrid_runs[-1]["hybrid_best_mse"],
                hybrid_runs[-1]["hybrid_best_blend"],
            )
    hybrid_agg = aggregate_hybrid(hybrid_runs)

    # DP vs oracle anchor ablation @ wrong40 fraction 0.25
    dp_ablation = []
    for seed in seeds:
        for mode in ("oracle_true", "level1_estimate", "dp_mean"):
            r = run_assignment_barrier_stage2(seed, 0.25, "wrong40", mode, cfg, r03)
            dp_ablation.append(
                {
                    "seed": seed,
                    "mean_anchor_mode": mode,
                    "cooccurrence_mse": r["methods"]["gard_cooccurrence_wrong_h"],
                    "refine_mse": r["methods"]["incidence_refine_cooccurrence"],
                }
            )

    def mean_field(rows: list[dict], field: str, mode: str) -> float:
        vals = [x[field] for x in rows if x["mean_anchor_mode"] == mode]
        return float(np.mean(vals))

    dp_summary = {
        "fraction": 0.25,
        "assignment": "wrong40",
        "mean_cooccurrence_mse": {
            "oracle_true": mean_field(dp_ablation, "cooccurrence_mse", "oracle_true"),
            "level1_estimate": mean_field(dp_ablation, "cooccurrence_mse", "level1_estimate"),
            "dp_mean": mean_field(dp_ablation, "cooccurrence_mse", "dp_mean"),
        },
        "mean_refine_mse": {
            "oracle_true": mean_field(dp_ablation, "refine_mse", "oracle_true"),
            "level1_estimate": mean_field(dp_ablation, "refine_mse", "level1_estimate"),
            "dp_mean": mean_field(dp_ablation, "refine_mse", "dp_mean"),
        },
        "dp_epsilon": r03.dp_epsilon,
        "dp_l2_sensitivity": r03.dp_l2_sensitivity,
    }

    # Criterion A check (oracle, level1 cooccurrence — honest parent gate)
    row_a = next(
        r for r in threat_oracle_level1_co["parent_mean_mse_gate"] if r["target_mse"] == 0.10
    )
    criterion_a_oracle_level1 = (
        row_a.get("reaches_target", False) and row_a.get("reduction_vs_full", 0) >= 0.25
    )

    row_wrong40 = next(
        r for r in threat_wrong40_refine["parent_mean_mse_gate"] if r["target_mse"] == 0.10
    )
    criterion_a_wrong40_refine = (
        row_wrong40.get("reaches_target", False)
        and row_wrong40.get("reduction_vs_full", 0) >= 0.25
    )

    mse_methods = {
        "shard_style_ls_wrong_h",
        "gard_cooccurrence_wrong_h",
        "gard_oracle_chain_wrong_h",
        "incidence_refine_cooccurrence",
        "gard_oracle_h_upper_bound",
    }
    wrong40_at_025 = [
        a
        for a in barrier_agg
        if a["assignment_regime"] == "wrong40"
        and a["fraction"] == 0.25
        and a["method"] in mse_methods
        and a["mean_anchor_mode"] == "level1_estimate"
    ]
    wrong_best = (
        min(wrong40_at_025, key=lambda a: a["snapshot_mse_mean"]) if wrong40_at_025 else None
    )

    results = {
        "benchmark": "round03_assignment_barrier_incidence_dp_hybrid",
        "criterion_A_preregistered_gate": CRITERION_A_GATE,
        "mse_targets": list(MSE_TARGETS),
        "round03_config": asdict(r03),
        "stage2_config": asdict(cfg),
        "qfl_config": asdict(qfl_cfg),
        "seeds": seeds,
        "assignment_barrier": {
            "per_run": barrier_runs,
            "aggregate": barrier_agg,
            "cooccurrence_vs_oracle_at_0.25": co_vs_oracle,
            "dp_anchor_ablation_wrong40": dp_summary,
            "threat_reduction": {
                "wrong40_incidence_refine_level1": threat_wrong40_refine,
                "wrong40_cooccurrence_level1": threat_wrong40_co,
            },
        },
        "hybrid_t1p_sparse_gard": {
            "per_run": hybrid_runs,
            "aggregate": hybrid_agg,
        },
        "config_breakthrough": {
            "A_row_reduction_25pct_oracle_parent_gate_level1_co": criterion_a_oracle_level1,
            "A_row_reduction_25pct_wrong40_parent_gate_refine": criterion_a_wrong40_refine,
            "B_defense_50pct_at_10pct_utility": False,
            "C_cert_2x_tighter_than_naive": None,
        },
        "breakthrough_status": {
            "assignment_barrier_broken": criterion_a_wrong40_refine,
            "best_wrong40_method_at_0.25": wrong_best,
            "cooccurrence_beats_oracle_chain_wrong40": any(
                r["cooccurrence_minus_oracle"] < -0.05
                for r in co_vs_oracle
                if r["assignment_regime"] == "wrong40"
            ),
            "hybrid_beats_t1p_fraction_wrong40": float(
                np.mean(
                    [
                        r["hybrid_best_mse"] < r["t1p_mse"]
                        for r in hybrid_runs
                        if r["assignment_regime"] == "wrong40"
                    ]
                )
            ),
        },
        "honesty_notes": [
            "INCIDENCE-REFINE uses greedy top-B relabeling; no access to true H during refine.",
            "DP-mean is Laplace on batch-mean average; epsilon=%s not a deployed DP-SGD guarantee."
            % r03.dp_epsilon,
            "Hybrid T1p uses true batch membership for partial rows (T1p tier); sparse rows still wrong-H stressed.",
            "Criterion A parent gate still fails for wrong40/unknown under level1+refine unless noted in JSON.",
        ],
        "runtime_sec": time.perf_counter() - t0,
    }

    out = ARTIFACTS / "round03_metrics.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out)
    logger.info(
        "Criterion A oracle level1 co: %s wrong40 refine: %s",
        criterion_a_oracle_level1,
        criterion_a_wrong40_refine,
    )
    logger.info("Cooccurrence vs oracle: %s", co_vs_oracle)
    logger.info("Runtime %.2fs", results["runtime_sec"])


if __name__ == "__main__":
    main()
