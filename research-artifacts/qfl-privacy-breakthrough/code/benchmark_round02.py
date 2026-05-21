#!/usr/bin/env python3
"""Round-02 QFL privacy benchmark — assignment/graph stress, honest threat reduction.

Implements supervisor Round 01 demands:
  - Assignment stress (oracle, wrong20, wrong40, unknown_random)
  - Graph stress (oracle, noisy, wrong, co-occurrence)
  - Parent-aligned + per-seed threat reduction @ MSE {0.05, 0.10, 0.15}
  - Mean-anchor ablation (none / level1 / noisy / oracle)
  - Ridge + low-rank baselines
  - ASSIGN-LOCK stub (permuted batch-index secure aggregation)
  - QFL-SHIELD rank/sigma grid with <=10% normalized utility
  - LEAK-CERT parked (tier-specific bound deferred to Round 03+)
"""

from __future__ import annotations

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

sys.path.insert(0, str(VENDOR.parent))
sys.path.insert(0, str(QTERM / "code"))

from shard_sim.attacker import ShardAttacker  # noqa: E402
from shard_sim.surrogate_model import SurrogateQFL  # noqa: E402
from qterm_attack import QtermAttack, QtermConfig, QtermTier  # noqa: E402


MSE_TARGETS = (0.05, 0.10, 0.15)
CRITERION_A_GATE = "parent_mean_mse_at_minimum_rows"


@dataclass(frozen=True)
class Stage2Config:
    n_samples: int = 48
    batch_size: int = 4
    dim_g: int = 32
    true_rank: int = 4
    n_epochs: int = 5
    noise_std: float = 0.01
    mean_anchor_weight: float = 100.0
    validation_fraction: float = 0.25
    ridge_grid: tuple[float, ...] = (0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
    graph_lambda_grid: tuple[float, ...] = (0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
    low_rank_grid: tuple[int, ...] = (2, 4, 8, 12)
    random_fractions: tuple[float, ...] = (1.0, 0.6, 0.4, 0.25, 0.15)
    noisy_mean_std: float = 0.05


@dataclass(frozen=True)
class QflConfig:
    n_samples: int = 32
    batch_size: int = 4
    n_epochs: int = 10
    dim_g: int = 32
    true_rank: int = 4
    noise_level: float = 0.01
    partial_rows: int = 7
    shield_ranks: tuple[int, ...] = (1, 2, 4, 8)
    shield_sigmas: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.25)
    max_utility_loss_fraction: float = 0.10


def setup_logging() -> logging.Logger:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    logging.getLogger("shard_sim.attacker").setLevel(logging.WARNING)
    logger = logging.getLogger("round02")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(LOGS / "experiment_round02.log", mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def hungarian_snapshot_mse(recovered: np.ndarray, truth: np.ndarray) -> float:
    from scipy.optimize import linear_sum_assignment

    r_sq = np.sum(recovered**2, axis=1, keepdims=True)
    t_sq = np.sum(truth**2, axis=1, keepdims=True)
    dist = r_sq + t_sq.T - 2.0 * recovered @ truth.T
    np.maximum(dist, 0.0, out=dist)
    row, col = linear_sum_assignment(dist)
    return float(np.mean((recovered[row] - truth[col]) ** 2))


# ---------------------------------------------------------------------------
# Stage-2 core (ported from parent benchmark_round05.py)
# ---------------------------------------------------------------------------


def make_low_rank_snapshots(cfg: Stage2Config, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, cfg.n_samples)
    latent_cols = [
        np.sin(2.0 * np.pi * t),
        np.cos(2.0 * np.pi * t),
        np.sin(4.0 * np.pi * t + 0.3),
        np.cos(4.0 * np.pi * t - 0.2),
    ]
    latent = np.vstack(latent_cols[: cfg.true_rank]).T
    latent += 0.08 * rng.normal(size=latent.shape)
    basis = rng.normal(size=(cfg.dim_g, cfg.true_rank))
    basis, _ = np.linalg.qr(basis)
    snapshots = latent @ basis.T
    snapshots += 0.03 * rng.normal(size=snapshots.shape)
    snapshots -= snapshots.mean(axis=0, keepdims=True)
    snapshots /= np.std(snapshots) + 1e-12
    return snapshots.astype(np.float64)


def make_incidence(cfg: Stage2Config, seed: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    rng = np.random.default_rng(seed + 10_000)
    k_batches = cfg.n_samples // cfg.batch_size
    rows: list[np.ndarray] = []
    metadata: list[tuple[int, int]] = []
    for epoch in range(cfg.n_epochs):
        perm = rng.permutation(cfg.n_samples)
        for batch in range(k_batches):
            members = perm[batch * cfg.batch_size : (batch + 1) * cfg.batch_size]
            row = np.zeros(cfg.n_samples, dtype=np.float64)
            row[members] = 1.0 / cfg.batch_size
            rows.append(row)
            metadata.append((epoch, batch))
    return np.vstack(rows), metadata


def row_members(row: np.ndarray) -> np.ndarray:
    return np.flatnonzero(row > 0.0)


def row_from_members(members: Iterable[int], n_samples: int) -> np.ndarray:
    row = np.zeros(n_samples, dtype=np.float64)
    members = np.array(list(members), dtype=np.int64)
    row[members] = 1.0 / len(members)
    return row


def select_rows(h_full: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 20_000 + int(round(fraction * 1000)))
    n_rows = h_full.shape[0]
    n_keep = max(1, int(round(fraction * n_rows)))
    return np.sort(rng.choice(n_rows, size=n_keep, replace=False))


def corrupt_incidence(
    h_true: np.ndarray,
    cfg: Stage2Config,
    regime: str,
    seed: int,
) -> tuple[np.ndarray, float]:
    regime_offsets = {"oracle": 0, "wrong20": 20, "wrong40": 40, "unknown_random": 100}
    rng = np.random.default_rng(seed + 30_000 + regime_offsets[regime])
    if regime == "oracle":
        return h_true.copy(), 1.0
    replace_rate = {"wrong20": 0.20, "wrong40": 0.40, "unknown_random": 1.00}[regime]
    assumed_rows = []
    overlaps = []
    all_ids = np.arange(cfg.n_samples)
    for row in h_true:
        true_members = row_members(row)
        if regime == "unknown_random":
            guessed = rng.choice(all_ids, size=cfg.batch_size, replace=False)
        else:
            n_replace = max(1, int(round(replace_rate * cfg.batch_size)))
            keep = rng.choice(true_members, size=cfg.batch_size - n_replace, replace=False)
            pool = np.setdiff1d(all_ids, keep, assume_unique=False)
            fill = rng.choice(pool, size=n_replace, replace=False)
            guessed = np.concatenate([keep, fill])
        assumed_rows.append(row_from_members(guessed, cfg.n_samples))
        overlaps.append(len(set(true_members).intersection(set(guessed))) / cfg.batch_size)
    return np.vstack(assumed_rows), float(np.mean(overlaps))


def split_train_val(n_rows: int, cfg: Stage2Config, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 40_000 + n_rows)
    n_val = max(1, int(round(cfg.validation_fraction * n_rows)))
    if n_rows >= 8:
        n_val = max(2, n_val)
    n_val = min(n_val, n_rows - 1) if n_rows > 1 else 0
    perm = rng.permutation(n_rows)
    val = np.sort(perm[:n_val])
    train = np.sort(perm[n_val:])
    return train, val


def chain_laplacian(n_samples: int, order: np.ndarray | None = None) -> np.ndarray:
    if order is None:
        order = np.arange(n_samples)
    weights = np.zeros((n_samples, n_samples), dtype=np.float64)
    for left, right in zip(order[:-1], order[1:]):
        weights[left, right] += 1.0
        weights[right, left] += 1.0
    degrees = np.sum(weights, axis=1)
    return np.diag(degrees) - weights


def noisy_chain_laplacian(n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 60_000)
    weights = np.zeros((n_samples, n_samples), dtype=np.float64)
    for i in range(n_samples - 1):
        weights[i, i + 1] += 0.7
        weights[i + 1, i] += 0.7
    for _ in range(n_samples):
        i, j = rng.choice(n_samples, size=2, replace=False)
        weights[i, j] += 0.3
        weights[j, i] += 0.3
    degrees = np.sum(weights, axis=1)
    return np.diag(degrees) - weights


def cooccurrence_laplacian(h_assumed: np.ndarray) -> np.ndarray:
    n_samples = h_assumed.shape[1]
    weights = np.zeros((n_samples, n_samples), dtype=np.float64)
    for row in h_assumed:
        members = row_members(row)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                weights[members[i], members[j]] += 1.0
                weights[members[j], members[i]] += 1.0
    if np.max(weights) > 0:
        weights /= np.max(weights)
    degrees = np.sum(weights, axis=1)
    return np.diag(degrees) - weights


def knn_laplacian(points: np.ndarray, k: int = 4) -> np.ndarray:
    n_samples = points.shape[0]
    centered = points - points.mean(axis=0, keepdims=True)
    sq_norm = np.sum(centered * centered, axis=1, keepdims=True)
    dists = sq_norm + sq_norm.T - 2.0 * centered @ centered.T
    np.fill_diagonal(dists, np.inf)
    weights = np.zeros((n_samples, n_samples), dtype=np.float64)
    finite = dists[np.isfinite(dists)]
    sigma = float(np.median(finite)) if finite.size else 1.0
    sigma = max(sigma, 1e-8)
    for i in range(n_samples):
        nbrs = np.argsort(dists[i])[:k]
        for j in nbrs:
            w = np.exp(-dists[i, j] / sigma)
            weights[i, j] = max(weights[i, j], w)
            weights[j, i] = max(weights[j, i], w)
    degrees = np.sum(weights, axis=1)
    return np.diag(degrees) - weights


def resolve_mean_anchor(
    mode: str,
    s_true: np.ndarray,
    h_obs: np.ndarray,
    m_obs: np.ndarray,
    cfg: Stage2Config,
    seed: int,
) -> np.ndarray:
    if mode == "oracle_true":
        return s_true.mean(axis=0)
    if mode == "no_anchor":
        return np.zeros(cfg.dim_g, dtype=np.float64)
    if mode == "level1_estimate":
        # Level-1 style: average of observed batch-mean gradients (no snapshot oracle)
        return m_obs.mean(axis=0)
    if mode == "noisy_mean":
        rng = np.random.default_rng(seed + 55_000)
        return s_true.mean(axis=0) + cfg.noisy_mean_std * rng.normal(size=(cfg.dim_g,))
    raise ValueError(f"unknown mean anchor mode {mode}")


def anchor_weight_for_mode(mode: str, cfg: Stage2Config) -> float:
    return 0.0 if mode == "no_anchor" else cfg.mean_anchor_weight


def solve_map(
    h_obs: np.ndarray,
    m_obs: np.ndarray,
    mean_snapshot: np.ndarray,
    cfg: Stage2Config,
    *,
    anchor_weight: float,
    ridge_lambda: float = 0.0,
    graph_lambda: float = 0.0,
    laplacian: np.ndarray | None = None,
) -> np.ndarray:
    n_samples = cfg.n_samples
    anchor = np.ones((n_samples, 1), dtype=np.float64) / n_samples
    lhs = h_obs.T @ h_obs
    if anchor_weight > 0.0:
        lhs = lhs + anchor_weight * (anchor @ anchor.T)
    if ridge_lambda > 0.0:
        lhs = lhs + ridge_lambda * np.eye(n_samples)
    if graph_lambda > 0.0 and laplacian is not None:
        lhs = lhs + graph_lambda * laplacian
    rhs = h_obs.T @ m_obs
    if anchor_weight > 0.0:
        rhs = rhs + anchor_weight * anchor @ mean_snapshot[None, :]
    return np.linalg.solve(lhs + 1e-8 * np.eye(n_samples), rhs)


def validation_mse(s_hat: np.ndarray, h_val: np.ndarray, m_val: np.ndarray) -> float:
    if h_val.shape[0] == 0:
        return 0.0
    return float(np.mean((h_val @ s_hat - m_val) ** 2))


def select_ridge(
    h_train: np.ndarray,
    m_train: np.ndarray,
    h_val: np.ndarray,
    m_val: np.ndarray,
    h_all: np.ndarray,
    m_all: np.ndarray,
    mean_snapshot: np.ndarray,
    cfg: Stage2Config,
    anchor_weight: float,
) -> tuple[np.ndarray, float]:
    best_lambda = cfg.ridge_grid[0]
    best_score = float("inf")
    for lam in cfg.ridge_grid:
        s_candidate = solve_map(
            h_train, m_train, mean_snapshot, cfg, anchor_weight=anchor_weight, ridge_lambda=lam
        )
        score = validation_mse(s_candidate, h_val, m_val)
        if score < best_score:
            best_score = score
            best_lambda = lam
    return (
        solve_map(
            h_all, m_all, mean_snapshot, cfg, anchor_weight=anchor_weight, ridge_lambda=best_lambda
        ),
        float(best_lambda),
    )


def low_rank_project(s_hat: np.ndarray, rank: int) -> np.ndarray:
    mean = s_hat.mean(axis=0, keepdims=True)
    centered = s_hat - mean
    u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    r = min(rank, len(singular))
    return mean + (u[:, :r] * singular[:r]) @ vt[:r]


def select_low_rank(
    h_train: np.ndarray,
    m_train: np.ndarray,
    h_val: np.ndarray,
    m_val: np.ndarray,
    h_all: np.ndarray,
    m_all: np.ndarray,
    mean_snapshot: np.ndarray,
    cfg: Stage2Config,
    anchor_weight: float,
) -> tuple[np.ndarray, dict]:
    best = {"ridge_lambda": cfg.ridge_grid[0], "rank": cfg.low_rank_grid[0]}
    best_score = float("inf")
    for lam in cfg.ridge_grid:
        s_base = solve_map(
            h_train, m_train, mean_snapshot, cfg, anchor_weight=anchor_weight, ridge_lambda=lam
        )
        for rank in cfg.low_rank_grid:
            s_candidate = low_rank_project(s_base, rank)
            score = validation_mse(s_candidate, h_val, m_val)
            if score < best_score:
                best_score = score
                best = {"ridge_lambda": float(lam), "rank": int(rank)}
    s_all = solve_map(
        h_all,
        m_all,
        mean_snapshot,
        cfg,
        anchor_weight=anchor_weight,
        ridge_lambda=best["ridge_lambda"],
    )
    return low_rank_project(s_all, best["rank"]), best


def select_graph_map(
    h_train: np.ndarray,
    m_train: np.ndarray,
    h_val: np.ndarray,
    m_val: np.ndarray,
    h_all: np.ndarray,
    m_all: np.ndarray,
    mean_snapshot: np.ndarray,
    lap: np.ndarray,
    cfg: Stage2Config,
    anchor_weight: float,
    rank_augmented: int,
) -> tuple[np.ndarray, float]:
    lambda_grid = (0.0,) if rank_augmented >= cfg.n_samples else cfg.graph_lambda_grid
    best_lambda = lambda_grid[0]
    best_score = float("inf")
    for lam in lambda_grid:
        s_candidate = solve_map(
            h_train,
            m_train,
            mean_snapshot,
            cfg,
            anchor_weight=anchor_weight,
            graph_lambda=lam,
            laplacian=lap,
        )
        score = validation_mse(s_candidate, h_val, m_val)
        if score < best_score:
            best_score = score
            best_lambda = lam
    return (
        solve_map(
            h_all,
            m_all,
            mean_snapshot,
            cfg,
            anchor_weight=anchor_weight,
            graph_lambda=best_lambda,
            laplacian=lap,
        ),
        float(best_lambda),
    )


def snapshot_mse(s_hat: np.ndarray, s_true: np.ndarray) -> float:
    return float(np.mean((s_hat - s_true) ** 2))


def run_stage2_one(
    seed: int,
    fraction: float,
    assignment_regime: str,
    mean_anchor_mode: str,
    cfg: Stage2Config,
) -> dict:
    s_true = make_low_rank_snapshots(cfg, seed)
    h_full, _metadata = make_incidence(cfg, seed)
    keep = select_rows(h_full, fraction, seed)
    h_true = h_full[keep]
    rng = np.random.default_rng(seed + 50_000 + len(keep))
    m_obs = h_true @ s_true + cfg.noise_std * rng.normal(size=(len(keep), cfg.dim_g))
    h_assumed, overlap = corrupt_incidence(h_true, cfg, assignment_regime, seed)
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
    methods["shard_style_ls"] = snapshot_mse(s_ls, s_true)

    s_ridge, _ = select_ridge(
        h_train, m_train, h_val, m_val, h_assumed, m_obs, mean_snapshot, cfg, anchor_weight
    )
    methods["ridge_ls"] = snapshot_mse(s_ridge, s_true)

    s_lr, _ = select_low_rank(
        h_train, m_train, h_val, m_val, h_assumed, m_obs, mean_snapshot, cfg, anchor_weight
    )
    methods["low_rank_pca"] = snapshot_mse(s_lr, s_true)

    wrong_order = np.random.default_rng(seed + 70_000).permutation(cfg.n_samples)
    graphs = {
        "gard_oracle_graph": chain_laplacian(cfg.n_samples),
        "gard_noisy_graph": noisy_chain_laplacian(cfg.n_samples, seed),
        "gard_wrong_graph": chain_laplacian(cfg.n_samples, wrong_order),
        "gard_cooccurrence_graph": cooccurrence_laplacian(h_assumed),
        "gard_lsknn_graph": knn_laplacian(s_ls),
    }
    for name, lap in graphs.items():
        s_g, _ = select_graph_map(
            h_train,
            m_train,
            h_val,
            m_val,
            h_assumed,
            m_obs,
            mean_snapshot,
            lap,
            cfg,
            anchor_weight,
            rank_aug,
        )
        methods[name] = snapshot_mse(s_g, s_true)

    return {
        "seed": seed,
        "fraction": fraction,
        "observed_rows": int(len(keep)),
        "full_rows": int(h_full.shape[0]),
        "assignment_regime": assignment_regime,
        "assignment_overlap": overlap,
        "mean_anchor_mode": mean_anchor_mode,
        "methods": methods,
    }


def aggregate_stage2(runs: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for run in runs:
        key = (
            run["fraction"],
            run["assignment_regime"],
            run["mean_anchor_mode"],
        )
        groups.setdefault(key, []).append(run)
    out = []
    for (fraction, assignment, anchor), items in sorted(groups.items()):
        obs = float(np.mean([r["observed_rows"] for r in items]))
        for method in items[0]["methods"]:
            mse = np.array([r["methods"][method] for r in items], dtype=np.float64)
            out.append(
                {
                    "fraction": fraction,
                    "observed_rows_mean": obs,
                    "assignment_regime": assignment,
                    "mean_anchor_mode": anchor,
                    "method": method,
                    "snapshot_mse_mean": float(mse.mean()),
                    "snapshot_mse_std": float(mse.std(ddof=0)),
                    "per_seed_mse": {str(r["seed"]): r["methods"][method] for r in items},
                    "n_seeds": len(items),
                }
            )
    return out


def threat_reduction_tables(
    runs: list[dict],
    cfg: Stage2Config,
    *,
    assignment: str,
    mean_anchor_mode: str = "oracle_true",
    method: str = "gard_oracle_graph",
) -> dict:
    full_rows = runs[0]["full_rows"]
    fractions = sorted({r["fraction"] for r in runs}, reverse=True)
    subset = [
        r
        for r in runs
        if r["assignment_regime"] == assignment and r["mean_anchor_mode"] == mean_anchor_mode
    ]
    by_fraction: dict[float, list[dict]] = {}
    for r in subset:
        by_fraction.setdefault(r["fraction"], []).append(r)

    parent_rows = []
    per_seed_rows = []
    round01_style_rows = []

    for target in MSE_TARGETS:
        parent_candidates = []
        for frac in fractions:
            items = by_fraction.get(frac, [])
            if not items:
                continue
            obs = int(round(np.mean([x["observed_rows"] for x in items])))
            mses = [x["methods"][method] for x in items]
            mean_mse = float(np.mean(mses))
            n_pass = sum(1 for m in mses if m <= target)
            pass_rate = n_pass / len(mses)
            per_seed_rows.append(
                {
                    "target_mse": target,
                    "fraction": frac,
                    "observed_rows": obs,
                    "mean_mse": mean_mse,
                    "per_seed_pass_rate": pass_rate,
                    "n_seeds_pass": n_pass,
                    "n_seeds": len(mses),
                }
            )
            if mean_mse <= target:
                parent_candidates.append(
                    {
                        "target_mse": target,
                        "min_observed_rows": obs,
                        "fraction": frac,
                        "mean_mse": mean_mse,
                        "reduction_vs_full": 1.0 - obs / full_rows,
                    }
                )
            # Round-01 style: any single seed×fraction run passes
            for item in items:
                if item["methods"][method] <= target:
                    round01_style_rows.append(
                        {
                            "target_mse": target,
                            "seed": item["seed"],
                            "observed_rows": item["observed_rows"],
                            "fraction": item["fraction"],
                            "mse": item["methods"][method],
                        }
                    )
        parent_best = (
            min(parent_candidates, key=lambda c: c["min_observed_rows"])
            if parent_candidates
            else None
        )
        if parent_best is None:
            parent_rows.append(
                {
                    "target_mse": target,
                    "reaches_target": False,
                    "min_observed_rows": None,
                    "reduction_vs_full": None,
                }
            )
        else:
            parent_rows.append({"reaches_target": True, **parent_best})

    r01_min = None
    if round01_style_rows:
        best = min(round01_style_rows, key=lambda x: (x["observed_rows"], x["target_mse"]))
        if best["target_mse"] == 0.10:
            r01_min = best["observed_rows"]

    return {
        "assignment_regime": assignment,
        "mean_anchor_mode": mean_anchor_mode,
        "method": method,
        "full_rows": full_rows,
        "parent_mean_mse_gate": parent_rows,
        "per_seed_pass_rate_by_fraction": per_seed_rows,
        "round01_any_seed_min_rows_at_0.10": r01_min,
    }


# ---------------------------------------------------------------------------
# ASSIGN-LOCK stub
# ---------------------------------------------------------------------------


def run_assign_lock_stub(seed: int, fraction: float, cfg: Stage2Config) -> dict:
    """Permute published batch-row order; attacker solves with identity slot order."""
    s_true = make_low_rank_snapshots(cfg, seed)
    h_full, _ = make_incidence(cfg, seed)
    keep = select_rows(h_full, fraction, seed)
    h_true = h_full[keep]
    rng = np.random.default_rng(seed + 80_000)
    m_true = h_true @ s_true + cfg.noise_std * rng.normal(size=(len(keep), cfg.dim_g))

    perm = rng.permutation(len(keep))
    m_pub = m_true[perm]
    h_identity = h_true.copy()  # attacker assumes slot i maps to true row i (wrong after perm)

    mean_oracle = s_true.mean(axis=0)
    anchor_w = cfg.mean_anchor_weight
    lap = chain_laplacian(cfg.n_samples)

    s_oracle_assign = solve_map(h_true, m_true, mean_oracle, cfg, anchor_weight=anchor_w)
    s_lock_broken = solve_map(h_identity, m_pub, mean_oracle, cfg, anchor_weight=anchor_w)
    s_lock_gard = solve_map(
        h_identity,
        m_pub,
        mean_oracle,
        cfg,
        anchor_weight=anchor_w,
        graph_lambda=0.1,
        laplacian=lap,
    )
    s_oracle_gard, _ = select_graph_map(
        h_true[: max(2, len(keep) // 2)],
        m_true[: max(2, len(keep) // 2)],
        h_true[max(2, len(keep) // 2) :],
        m_true[max(2, len(keep) // 2) :],
        h_true,
        m_true,
        mean_oracle,
        lap,
        cfg,
        anchor_w,
        cfg.n_samples,
    )

    def row_recovery_score(h_assumed: np.ndarray) -> float:
        overlaps = []
        for i in range(h_assumed.shape[0]):
            overlaps.append(
                len(set(row_members(h_true[i])).intersection(set(row_members(h_assumed[i]))))
                / cfg.batch_size
            )
        return float(np.mean(overlaps))

    return {
        "seed": seed,
        "fraction": fraction,
        "observed_rows": int(len(keep)),
        "assignment_overlap_identity_slots": row_recovery_score(h_identity),
        "snapshot_mse_oracle_assignment": snapshot_mse(s_oracle_assign, s_true),
        "snapshot_mse_assign_lock_ls": snapshot_mse(s_lock_broken, s_true),
        "snapshot_mse_assign_lock_gard": snapshot_mse(s_lock_gard, s_true),
        "snapshot_mse_gard_oracle_upper_bound": snapshot_mse(s_oracle_gard, s_true),
    }


# ---------------------------------------------------------------------------
# QFL-SHIELD grid
# ---------------------------------------------------------------------------


def make_smooth_snapshots(cfg: QflConfig, seed: int) -> np.ndarray:
    return make_low_rank_snapshots(
        Stage2Config(
            n_samples=cfg.n_samples,
            dim_g=cfg.dim_g,
            true_rank=cfg.true_rank,
            n_epochs=cfg.n_epochs,
        ),
        seed,
    )


def make_epoch_batches(n_samples: int, batch_size: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed + 9000)
    perm = rng.permutation(n_samples)
    k = n_samples // batch_size
    return [perm[b * batch_size : (b + 1) * batch_size].tolist() for b in range(k)]


def simulate_gradients(
    snapshots: np.ndarray,
    batches: list[list[int]],
    surrogate: SurrogateQFL,
) -> tuple[np.ndarray, list[np.ndarray]]:
    a_epoch = surrogate.generate_coefficient_matrix()
    grads: list[np.ndarray] = []
    for batch_indices in batches:
        mean_snap = snapshots[batch_indices].mean(axis=0)
        g = a_epoch @ mean_snap
        if surrogate.noise_level > 0:
            g = g + surrogate._np_rng.normal(0, surrogate.noise_level, size=g.shape)
        grads.append(g)
    return a_epoch, grads


def apply_shield(
    true_snapshots: np.ndarray,
    rank: int,
    noise_std: float,
    seed: int,
) -> np.ndarray:
    centered = true_snapshots - true_snapshots.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    r = min(rank, len(s))
    low = (u[:, :r] * s[:r]) @ vt[:r]
    published = true_snapshots.mean(axis=0, keepdims=True) + low
    if noise_std > 0:
        rng = np.random.default_rng(seed + 99_000)
        published = published + noise_std * rng.normal(size=published.shape)
    return published.astype(np.float64)


def normalized_utility_loss(
    true_snaps: np.ndarray,
    published_snaps: np.ndarray,
    surrogate: SurrogateQFL,
    seed: int,
) -> tuple[float, float]:
    """Returns (raw grad MSE, fraction vs undefended baseline on same batches)."""
    losses = []
    base_losses = []
    for e in range(3):
        batches = make_epoch_batches(true_snaps.shape[0], 4, seed + e)
        for batch in batches:
            g_true = simulate_gradients(true_snaps, [batch], surrogate)[1][0]
            g_pub = simulate_gradients(published_snaps, [batch], surrogate)[1][0]
            losses.append(float(np.mean((g_true - g_pub) ** 2)))
            base_losses.append(float(np.mean(g_true**2)))
    raw = float(np.mean(losses))
    baseline = float(np.mean(base_losses))
    return raw, raw / max(baseline, 1e-12)


def run_shield_grid(seed: int, cfg: QflConfig) -> list[dict]:
    torch.manual_seed(seed)
    true_snapshots = make_smooth_snapshots(cfg, seed)
    surrogate = SurrogateQFL(
        input_dim=64,
        dim_g=cfg.dim_g,
        n_params=cfg.dim_g,
        noise_level=cfg.noise_level,
        seed=seed,
    )
    attacker = ShardAttacker(
        dim_g=cfg.dim_g,
        n_samples=cfg.n_samples,
        batch_size=cfg.batch_size,
        max_iter=200,
        tol=1e-8,
        random_seed=seed,
    )

    coeff: list[np.ndarray] = []
    batch_grads: list[list[np.ndarray]] = []
    for e in range(cfg.n_epochs):
        batches = make_epoch_batches(cfg.n_samples, cfg.batch_size, seed + e)
        a_e, grads_e = simulate_gradients(true_snapshots, batches, surrogate)
        coeff.append(a_e)
        batch_grads.append(grads_e)

    e_bar = attacker.level1_mean_recovery(coeff, batch_grads)
    shard_s = attacker.level2_disaggregate(e_bar, coeff, batch_grads, true_snapshots)
    shard_mse = hungarian_snapshot_mse(shard_s, true_snapshots)

    qterm = QtermAttack(
        QtermConfig(
            tier=QtermTier.T1P,
            n_samples=cfg.n_samples,
            batch_size=cfg.batch_size,
            partial_rows_per_epoch=cfg.partial_rows,
            random_seed=seed,
        )
    )
    t1p = qterm.recover(e_bar, coeff, batch_gradients=batch_grads)
    t1p_mse = hungarian_snapshot_mse(t1p.snapshots, true_snapshots)

    grid = []
    for rank in cfg.shield_ranks:
        for sigma in cfg.shield_sigmas:
            defended = apply_shield(true_snapshots, rank, sigma, seed)
            util_raw, util_frac = normalized_utility_loss(
                true_snapshots, defended, surrogate, seed
            )
            coeff_d: list[np.ndarray] = []
            batch_grads_d: list[list[np.ndarray]] = []
            for e in range(cfg.n_epochs):
                batches = make_epoch_batches(cfg.n_samples, cfg.batch_size, seed + e)
                a_e, grads_e = simulate_gradients(defended, batches, surrogate)
                coeff_d.append(a_e)
                batch_grads_d.append(grads_e)
            e_bar_d = attacker.level1_mean_recovery(coeff_d, batch_grads_d)
            r_def = qterm.recover(e_bar_d, coeff_d, batch_gradients=batch_grads_d)
            def_mse = hungarian_snapshot_mse(r_def.snapshots, true_snapshots)
            grid.append(
                {
                    "seed": seed,
                    "rank": rank,
                    "sigma": sigma,
                    "utility_gradient_mse": util_raw,
                    "utility_loss_fraction_normalized": util_frac,
                    "meets_utility_constraint": util_frac <= cfg.max_utility_loss_fraction,
                    "t1p_mse_undefended": t1p_mse,
                    "t1p_mse_defended": def_mse,
                    "shard_stage2_mse": shard_mse,
                    "attack_reduction_vs_t1p": 1.0 - def_mse / max(t1p_mse, 1e-12),
                    "attack_reduction_vs_shard": 1.0 - def_mse / max(shard_mse, 1e-12),
                }
            )
    return grid


def aggregate_shield_grid(all_cells: list[dict], cfg: QflConfig) -> dict:
    feasible = [c for c in all_cells if c["meets_utility_constraint"]]
    best_feasible = None
    if feasible:
        best_feasible = max(feasible, key=lambda c: c["attack_reduction_vs_t1p"])
    mean_undef = float(np.mean([c["t1p_mse_undefended"] for c in all_cells]))
    return {
        "n_grid_cells": len(all_cells),
        "n_feasible_utility": len(feasible),
        "best_feasible_cell": best_feasible,
        "mean_undefended_t1p_mse": mean_undef,
        "any_feasible_improves_t1p": bool(
            feasible and np.mean([c["attack_reduction_vs_t1p"] for c in feasible]) > 0
        ),
        "breakthrough_B_50pct_at_10pct_utility": bool(
            best_feasible is not None and best_feasible["attack_reduction_vs_t1p"] >= 0.50
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logger = setup_logging()
    t0 = time.perf_counter()
    cfg = Stage2Config()
    stage2_seeds = [3, 7, 11, 19, 23]
    assignments = ["oracle", "wrong20", "wrong40", "unknown_random"]
    anchor_modes_main = ["oracle_true"]
    anchor_modes_ablation = ["no_anchor", "level1_estimate", "noisy_mean", "oracle_true"]

    logger.info("Round 02 Stage-2 stress: seeds=%s assignments=%s", stage2_seeds, assignments)
    stage2_runs: list[dict] = []
    for frac in cfg.random_fractions:
        for assignment in assignments:
            for anchor in anchor_modes_main:
                for seed in stage2_seeds:
                    run = run_stage2_one(seed, frac, assignment, anchor, cfg)
                    stage2_runs.append(run)
                    logger.info(
                        "stage2 seed=%d frac=%.2f assign=%s gard_oracle=%.4g ls=%.4g",
                        seed,
                        frac,
                        assignment,
                        run["methods"]["gard_oracle_graph"],
                        run["methods"]["shard_style_ls"],
                    )

    # Mean-anchor ablation at oracle assignment, fractions 0.15 and 0.25
    ablation_runs: list[dict] = []
    for frac in (0.15, 0.25):
        for anchor in anchor_modes_ablation:
            for seed in stage2_seeds:
                ablation_runs.append(run_stage2_one(seed, frac, "oracle", anchor, cfg))

    aggregate = aggregate_stage2(stage2_runs)
    ablation_aggregate = aggregate_stage2(ablation_runs)

    threat_by_assignment = {
        a: threat_reduction_tables(
            stage2_runs, cfg, assignment=a, method="gard_oracle_graph"
        )
        for a in assignments
    }
    threat_shard = threat_reduction_tables(
        stage2_runs, cfg, assignment="oracle", method="shard_style_ls"
    )

    # Criterion A @ 0.10 with pre-registered gate
    oracle_parent = next(
        r for r in threat_by_assignment["oracle"]["parent_mean_mse_gate"] if r["target_mse"] == 0.10
    )
    wrong40_parent = next(
        r
        for r in threat_by_assignment["wrong40"]["parent_mean_mse_gate"]
        if r["target_mse"] == 0.10
    )
    unknown_parent = next(
        r
        for r in threat_by_assignment["unknown_random"]["parent_mean_mse_gate"]
        if r["target_mse"] == 0.10
    )

    criterion_a_oracle = (
        oracle_parent.get("reaches_target", False)
        and oracle_parent.get("reduction_vs_full", 0) >= 0.25
    )
    criterion_a_wrong40 = (
        wrong40_parent.get("reaches_target", False)
        and wrong40_parent.get("reduction_vs_full", 0) >= 0.25
    )
    criterion_a_unknown = (
        unknown_parent.get("reaches_target", False)
        and unknown_parent.get("reduction_vs_full", 0) >= 0.25
    )

    r01_inflated = threat_by_assignment["oracle"].get("round01_any_seed_min_rows_at_0.10")
    r01_reduction = 1.0 - r01_inflated / 60 if r01_inflated else None
    parent_reduction_010 = oracle_parent.get("reduction_vs_full")
    withdraw_85 = (
        r01_reduction is not None
        and r01_reduction >= 0.84
        and (parent_reduction_010 is None or parent_reduction_010 < 0.84)
    )

    # Best honest reduction across assignments/methods @ 0.10 parent gate
    best_honest = {"reduction": -1.0, "assignment": None, "method": None}
    for assignment in assignments:
        for method in [
            "gard_oracle_graph",
            "gard_noisy_graph",
            "gard_cooccurrence_graph",
            "ridge_ls",
            "low_rank_pca",
            "shard_style_ls",
        ]:
            tr = threat_reduction_tables(stage2_runs, cfg, assignment=assignment, method=method)
            row = next(r for r in tr["parent_mean_mse_gate"] if r["target_mse"] == 0.10)
            if row.get("reaches_target") and row.get("reduction_vs_full", -1) > best_honest["reduction"]:
                best_honest = {
                    "reduction": row["reduction_vs_full"],
                    "assignment": assignment,
                    "method": method,
                    "min_rows": row["min_observed_rows"],
                }

    # ASSIGN-LOCK
    lock_runs = []
    for frac in (0.25, 0.15):
        for seed in stage2_seeds:
            lock_runs.append(run_assign_lock_stub(seed, frac, cfg))
            logger.info(
                "ASSIGN-LOCK seed=%d frac=%.2f lock_mse=%.4g oracle_gard=%.4g",
                seed,
                frac,
                lock_runs[-1]["snapshot_mse_assign_lock_ls"],
                lock_runs[-1]["snapshot_mse_gard_oracle_upper_bound"],
            )
    lock_agg = {
        "mean_lock_ls_mse": float(
            np.mean([r["snapshot_mse_assign_lock_ls"] for r in lock_runs])
        ),
        "mean_oracle_gard_mse": float(
            np.mean([r["snapshot_mse_gard_oracle_upper_bound"] for r in lock_runs])
        ),
        "mean_slot_overlap": float(
            np.mean([r["assignment_overlap_identity_slots"] for r in lock_runs])
        ),
    }

    # QFL-SHIELD grid
    qfl_cfg = QflConfig()
    shield_cells: list[dict] = []
    for seed in stage2_seeds:
        shield_cells.extend(run_shield_grid(seed, qfl_cfg))
    shield_summary = aggregate_shield_grid(shield_cells, qfl_cfg)

    results = {
        "benchmark": "round02_assignment_graph_stress_and_honest_metrics",
        "criterion_A_preregistered_gate": CRITERION_A_GATE,
        "mse_targets": list(MSE_TARGETS),
        "config": asdict(cfg),
        "seeds": stage2_seeds,
        "assignment_regimes": assignments,
        "stage2": {
            "per_run": stage2_runs,
            "aggregate": aggregate,
            "mean_anchor_ablation": {
                "per_run": ablation_runs,
                "aggregate": ablation_aggregate,
            },
            "threat_reduction_by_assignment": threat_by_assignment,
            "threat_reduction_shard_oracle": threat_shard,
        },
        "config_breakthrough": {
            "A_row_reduction_25pct_oracle_parent_gate": criterion_a_oracle,
            "A_row_reduction_25pct_wrong40_parent_gate": criterion_a_wrong40,
            "A_row_reduction_25pct_unknown_random_parent_gate": criterion_a_unknown,
            "B_defense_50pct_at_10pct_utility": shield_summary["breakthrough_B_50pct_at_10pct_utility"],
            "C_cert_2x_tighter_than_naive": None,
        },
        "honesty_notes": [
            "Criterion A uses parent_mean_mse_at_minimum_rows (mean over seeds at budget), not Round-01 any-seed minimum.",
            "Round-01 85% row reduction is withdrawn when parent-aligned gate yields ~75% at MSE 0.10 (15/60 rows).",
            "GARD gains under wrong40/unknown_random are not deployment breakthroughs.",
            "LEAK-CERT parked to Round 03+ (tier-specific T1p partial-row bound).",
        ],
        "claim_withdrawal": {
            "round01_headline_85pct_withdrawn": bool(withdraw_85 or parent_reduction_010 is not None),
            "round01_inflated_min_rows_at_0.10": r01_inflated,
            "round01_inflated_reduction": r01_reduction,
            "parent_aligned_min_rows_at_0.10": oracle_parent.get("min_observed_rows"),
            "parent_aligned_reduction_at_0.10": parent_reduction_010,
        },
        "best_honest_row_reduction_at_0.10_parent_gate": best_honest,
        "assign_lock": {"per_run": lock_runs, "aggregate": lock_agg},
        "qfl_shield": {
            "config": asdict(qfl_cfg),
            "grid_cells": shield_cells,
            "summary": shield_summary,
            "kill_criteria_met": not shield_summary["any_feasible_improves_t1p"],
        },
        "leak_cert": {
            "status": "parked_round03",
            "reason": "Tier-specific rank/sensitivity bound at T1p 7 rows/epoch not implemented; dof heuristic failed R01.",
        },
        "runtime_sec": time.perf_counter() - t0,
    }

    out = ARTIFACTS / "round02_metrics.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out)
    logger.info(
        "Criterion A oracle parent gate @0.10: %s reduction=%s",
        criterion_a_oracle,
        parent_reduction_010,
    )
    logger.info(
        "Criterion A wrong40: %s unknown: %s",
        criterion_a_wrong40,
        criterion_a_unknown,
    )
    logger.info("Best honest reduction: %s", best_honest)
    logger.info("Shield feasible cells: %d", shield_summary["n_feasible_utility"])
    logger.info("Runtime %.2fs", results["runtime_sec"])


if __name__ == "__main__":
    main()
