#!/usr/bin/env python3
"""Round-05 SHARD Stage-2 benchmark with assignment and graph-prior stress tests.

This script extends the Round-04 oracle-incidence benchmark in four ways:

1. observed batch means are still generated from the true hidden incidence, but
   the attacker may receive oracle, partially wrong, or random/unknown incidence;
2. graph MAP is tested with oracle, noisy, wrong, co-occurrence, and LS-kNN graphs;
3. ridge, low-rank PCA, SHARD-style LS, and wrong-graph MAP baselines are included;
4. all ridge/graph regularization strengths are selected by held-out observed
   batch rows, never by snapshot MSE.

The benchmark remains a Stage-2 synthetic proxy. It does not claim to solve
gradient-to-snapshot Stage 1 or image inversion Stage 3.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


RUN_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = RUN_ROOT / "artifacts"
LOGS = RUN_ROOT / "logs"


@dataclass(frozen=True)
class SimConfig:
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
    random_fractions: tuple[float, ...] = (1.0, 0.8, 0.6, 0.4, 0.25, 0.15)
    prefix_epochs: tuple[int, ...] = (5, 4, 3, 2, 1)
    target_snapshot_mse: float = 0.10


def setup_logging() -> logging.Logger:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("round05")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(LOGS / "experiment_round05.log", mode="w")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def make_low_rank_snapshots(cfg: SimConfig, seed: int) -> np.ndarray:
    """Smooth, low-rank snapshots with enough noise to penalize over-smoothing."""
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


def make_incidence(cfg: SimConfig, seed: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
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


def select_rows(
    h_full: np.ndarray,
    metadata: list[tuple[int, int]],
    protocol: str,
    budget: float | int,
    seed: int,
) -> np.ndarray:
    if protocol == "random_fraction":
        rng = np.random.default_rng(seed + 20_000 + int(round(float(budget) * 1000)))
        n_rows = h_full.shape[0]
        n_keep = max(1, int(round(float(budget) * n_rows)))
        return np.sort(rng.choice(n_rows, size=n_keep, replace=False))
    if protocol == "prefix_epochs":
        return np.array([i for i, (epoch, _) in enumerate(metadata) if epoch < int(budget)])
    raise ValueError(f"unknown protocol {protocol}")


def corrupt_incidence(
    h_true: np.ndarray,
    cfg: SimConfig,
    regime: str,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Return attacker-assumed incidence and mean member overlap with truth."""
    regime_offsets = {"oracle": 0, "wrong20": 20, "wrong40": 40, "unknown_random": 100}
    rng = np.random.default_rng(seed + 30_000 + regime_offsets[regime])
    if regime == "oracle":
        return h_true.copy(), 1.0

    replace_rate = {
        "wrong20": 0.20,
        "wrong40": 0.40,
        "unknown_random": 1.00,
    }[regime]
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


def split_train_val(n_rows: int, cfg: SimConfig, seed: int) -> tuple[np.ndarray, np.ndarray]:
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
    return laplacian_from_weights(weights)


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
    return laplacian_from_weights(weights)


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
    return laplacian_from_weights(weights)


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
    return laplacian_from_weights(weights)


def laplacian_from_weights(weights: np.ndarray) -> np.ndarray:
    weights = 0.5 * (weights + weights.T)
    degrees = np.sum(weights, axis=1)
    return np.diag(degrees) - weights


def solve_map(
    h_obs: np.ndarray,
    m_obs: np.ndarray,
    mean_snapshot: np.ndarray,
    cfg: SimConfig,
    ridge_lambda: float = 0.0,
    graph_lambda: float = 0.0,
    laplacian: np.ndarray | None = None,
) -> np.ndarray:
    n_samples = cfg.n_samples
    anchor = np.ones((n_samples, 1), dtype=np.float64) / n_samples
    lhs = h_obs.T @ h_obs + cfg.mean_anchor_weight * (anchor @ anchor.T)
    if ridge_lambda > 0.0:
        lhs = lhs + ridge_lambda * np.eye(n_samples)
    if graph_lambda > 0.0 and laplacian is not None:
        lhs = lhs + graph_lambda * laplacian
    rhs = h_obs.T @ m_obs + cfg.mean_anchor_weight * anchor @ mean_snapshot[None, :]
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
    cfg: SimConfig,
) -> tuple[np.ndarray, float, float]:
    best_lambda = cfg.ridge_grid[0]
    best_score = float("inf")
    for lam in cfg.ridge_grid:
        s_candidate = solve_map(h_train, m_train, mean_snapshot, cfg, ridge_lambda=lam)
        score = validation_mse(s_candidate, h_val, m_val)
        if score < best_score:
            best_score = score
            best_lambda = lam
    return (
        solve_map(h_all, m_all, mean_snapshot, cfg, ridge_lambda=best_lambda),
        float(best_lambda),
        float(best_score),
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
    cfg: SimConfig,
) -> tuple[np.ndarray, dict, float]:
    best = {"ridge_lambda": cfg.ridge_grid[0], "rank": cfg.low_rank_grid[0]}
    best_score = float("inf")
    for lam in cfg.ridge_grid:
        s_base = solve_map(h_train, m_train, mean_snapshot, cfg, ridge_lambda=lam)
        for rank in cfg.low_rank_grid:
            s_candidate = low_rank_project(s_base, rank)
            score = validation_mse(s_candidate, h_val, m_val)
            if score < best_score:
                best_score = score
                best = {"ridge_lambda": float(lam), "rank": int(rank)}
    s_all = solve_map(h_all, m_all, mean_snapshot, cfg, ridge_lambda=best["ridge_lambda"])
    return low_rank_project(s_all, best["rank"]), best, float(best_score)


def select_graph_map(
    h_train: np.ndarray,
    m_train: np.ndarray,
    h_val: np.ndarray,
    m_val: np.ndarray,
    h_all: np.ndarray,
    m_all: np.ndarray,
    mean_snapshot: np.ndarray,
    lap: np.ndarray,
    cfg: SimConfig,
    rank_augmented: int,
) -> tuple[np.ndarray, float, float]:
    # If observed incidence plus the mean anchor is full-rank, exact SHARD LS is
    # already identifiable; the graph prior is only a missing-row fallback.
    lambda_grid = (0.0,) if rank_augmented >= cfg.n_samples else cfg.graph_lambda_grid
    best_lambda = lambda_grid[0]
    best_score = float("inf")
    for lam in lambda_grid:
        s_candidate = solve_map(
            h_train,
            m_train,
            mean_snapshot,
            cfg,
            graph_lambda=lam,
            laplacian=lap,
        )
        score = validation_mse(s_candidate, h_val, m_val)
        if score < best_score:
            best_score = score
            best_lambda = lam
    return (
        solve_map(h_all, m_all, mean_snapshot, cfg, graph_lambda=best_lambda, laplacian=lap),
        float(best_lambda),
        float(best_score),
    )


def snapshot_metrics(
    s_hat: np.ndarray,
    s_true: np.ndarray,
    h_true: np.ndarray,
    h_assumed: np.ndarray,
    m_obs: np.ndarray,
) -> dict:
    return {
        "snapshot_mse": float(np.mean((s_hat - s_true) ** 2)),
        "relative_fro_error": float(
            np.linalg.norm(s_hat - s_true, "fro") / (np.linalg.norm(s_true, "fro") + 1e-12)
        ),
        "true_observed_batch_mse": float(np.mean((h_true @ s_hat - m_obs) ** 2)),
        "assumed_observed_batch_mse": float(np.mean((h_assumed @ s_hat - m_obs) ** 2)),
    }


def run_one(
    seed: int,
    protocol: str,
    budget: float | int,
    assignment_regime: str,
    cfg: SimConfig,
) -> dict:
    s_true = make_low_rank_snapshots(cfg, seed)
    mean_snapshot = s_true.mean(axis=0)
    h_full, metadata = make_incidence(cfg, seed)
    keep = select_rows(h_full, metadata, protocol, budget, seed)
    h_true = h_full[keep]
    rng = np.random.default_rng(seed + 50_000 + len(keep))
    m_obs = h_true @ s_true + cfg.noise_std * rng.normal(size=(len(keep), cfg.dim_g))
    h_assumed, assignment_overlap = corrupt_incidence(h_true, cfg, assignment_regime, seed)
    train_idx, val_idx = split_train_val(len(keep), cfg, seed)
    h_train, m_train = h_assumed[train_idx], m_obs[train_idx]
    h_val, m_val = h_assumed[val_idx], m_obs[val_idx]
    rank_augmented = int(
        np.linalg.matrix_rank(
            np.vstack([h_assumed, np.ones((1, cfg.n_samples), dtype=np.float64) / cfg.n_samples])
        )
    )

    method_outputs: dict[str, dict] = {}

    s_ls = solve_map(h_assumed, m_obs, mean_snapshot, cfg)
    method_outputs["shard_style_ls"] = {
        **snapshot_metrics(s_ls, s_true, h_true, h_assumed, m_obs),
        "selected_lambda": 0.0,
    }

    s_ridge, ridge_lam, ridge_val = select_ridge(
        h_train, m_train, h_val, m_val, h_assumed, m_obs, mean_snapshot, cfg
    )
    method_outputs["ridge_ls"] = {
        **snapshot_metrics(s_ridge, s_true, h_true, h_assumed, m_obs),
        "selected_lambda": ridge_lam,
        "validation_mse": ridge_val,
    }

    s_low_rank, low_rank_params, low_rank_val = select_low_rank(
        h_train, m_train, h_val, m_val, h_assumed, m_obs, mean_snapshot, cfg
    )
    method_outputs["low_rank_pca"] = {
        **snapshot_metrics(s_low_rank, s_true, h_true, h_assumed, m_obs),
        "selected": low_rank_params,
        "selected_lambda": low_rank_params["ridge_lambda"],
        "selected_rank": low_rank_params["rank"],
        "validation_mse": low_rank_val,
    }

    wrong_order_rng = np.random.default_rng(seed + 70_000)
    wrong_order = wrong_order_rng.permutation(cfg.n_samples)
    graphs = {
        "gard_oracle_graph": chain_laplacian(cfg.n_samples),
        "gard_noisy_graph": noisy_chain_laplacian(cfg.n_samples, seed),
        "gard_wrong_graph": chain_laplacian(cfg.n_samples, wrong_order),
        "gard_cooccurrence_graph": cooccurrence_laplacian(h_assumed),
        "gard_lsknn_graph": knn_laplacian(s_ls),
    }
    for method, lap in graphs.items():
        s_graph, graph_lam, graph_val = select_graph_map(
            h_train,
            m_train,
            h_val,
            m_val,
            h_assumed,
            m_obs,
            mean_snapshot,
            lap,
            cfg,
            rank_augmented,
        )
        method_outputs[method] = {
            **snapshot_metrics(s_graph, s_true, h_true, h_assumed, m_obs),
            "selected_lambda": graph_lam,
            "validation_mse": graph_val,
        }

    s_wrong_forced = solve_map(
        h_assumed,
        m_obs,
        mean_snapshot,
        cfg,
        graph_lambda=0.3,
        laplacian=graphs["gard_wrong_graph"],
    )
    method_outputs["wrong_graph_forced_lam0.3"] = {
        **snapshot_metrics(s_wrong_forced, s_true, h_true, h_assumed, m_obs),
        "selected_lambda": 0.3,
    }

    if protocol == "random_fraction":
        budget_label = f"{float(budget):.2f}"
    else:
        budget_label = f"{int(budget)}epoch"
    return {
        "seed": seed,
        "protocol": protocol,
        "budget": budget,
        "budget_label": budget_label,
        "assignment_regime": assignment_regime,
        "assignment_member_overlap": assignment_overlap,
        "observed_batch_gradients": int(len(keep)),
        "full_batch_gradients": int(h_full.shape[0]),
        "observed_epochs": sorted({int(metadata[i][0]) for i in keep}),
        "incidence_rank_with_mean_anchor": rank_augmented,
        "rank_deficient_vs_n": rank_augmented < cfg.n_samples,
        "kept_rows": keep.tolist(),
        "train_rows": train_idx.tolist(),
        "validation_rows": val_idx.tolist(),
        "methods": method_outputs,
    }


def aggregate_runs(runs: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for run in runs:
        for method, metrics in run["methods"].items():
            key = (
                run["protocol"],
                run["budget_label"],
                run["assignment_regime"],
                method,
            )
            groups.setdefault(key, []).append((run, metrics))

    aggregate = []
    for (protocol, budget_label, assignment, method), items in sorted(groups.items()):
        rows = np.array([item[0]["observed_batch_gradients"] for item in items], dtype=np.float64)
        mse = np.array([item[1]["snapshot_mse"] for item in items], dtype=np.float64)
        rel = np.array([item[1]["relative_fro_error"] for item in items], dtype=np.float64)
        true_resid = np.array([item[1]["true_observed_batch_mse"] for item in items], dtype=np.float64)
        assumed_resid = np.array(
            [item[1]["assumed_observed_batch_mse"] for item in items], dtype=np.float64
        )
        selected_lambda = np.array(
            [float(item[1].get("selected_lambda", np.nan)) for item in items], dtype=np.float64
        )
        aggregate.append(
            {
                "protocol": protocol,
                "budget_label": budget_label,
                "assignment_regime": assignment,
                "method": method,
                "observed_batch_gradients_mean": float(rows.mean()),
                "snapshot_mse_mean": float(mse.mean()),
                "snapshot_mse_std": float(mse.std(ddof=0)),
                "relative_fro_error_mean": float(rel.mean()),
                "true_observed_batch_mse_mean": float(true_resid.mean()),
                "assumed_observed_batch_mse_mean": float(assumed_resid.mean()),
                "selected_lambda_mean": float(np.nanmean(selected_lambda)),
                "n_seeds": len(items),
            }
        )
    return aggregate


def compute_threat_reduction(aggregate: list[dict], cfg: SimConfig) -> list[dict]:
    rows = []
    for protocol in ["random_fraction", "prefix_epochs"]:
        for assignment in ["oracle", "wrong20", "wrong40", "unknown_random"]:
            for method in [
                "shard_style_ls",
                "ridge_ls",
                "low_rank_pca",
                "gard_oracle_graph",
                "gard_noisy_graph",
                "gard_wrong_graph",
                "gard_cooccurrence_graph",
                "gard_lsknn_graph",
            ]:
                candidates = [
                    row
                    for row in aggregate
                    if row["protocol"] == protocol
                    and row["assignment_regime"] == assignment
                    and row["method"] == method
                    and row["snapshot_mse_mean"] <= cfg.target_snapshot_mse
                ]
                if not candidates:
                    rows.append(
                        {
                            "protocol": protocol,
                            "assignment_regime": assignment,
                            "method": method,
                            "target_snapshot_mse": cfg.target_snapshot_mse,
                            "reaches_target": False,
                            "min_observed_batch_gradients": None,
                            "reduction_vs_full_observation": None,
                        }
                    )
                    continue
                best = min(candidates, key=lambda row: row["observed_batch_gradients_mean"])
                rows.append(
                    {
                        "protocol": protocol,
                        "assignment_regime": assignment,
                        "method": method,
                        "target_snapshot_mse": cfg.target_snapshot_mse,
                        "reaches_target": True,
                        "min_observed_batch_gradients": best["observed_batch_gradients_mean"],
                        "budget_label": best["budget_label"],
                        "reduction_vs_full_observation": float(
                            1.0 - best["observed_batch_gradients_mean"] / (cfg.n_epochs * cfg.n_samples / cfg.batch_size)
                        ),
                    }
                )
    return rows


def plot_random_oracle(aggregate: list[dict]) -> None:
    methods = [
        "shard_style_ls",
        "ridge_ls",
        "low_rank_pca",
        "gard_oracle_graph",
        "gard_noisy_graph",
        "gard_wrong_graph",
        "gard_lsknn_graph",
    ]
    labels = {
        "shard_style_ls": "SHARD LS",
        "ridge_ls": "Ridge LS",
        "low_rank_pca": "Low-rank PCA",
        "gard_oracle_graph": "GARD oracle graph",
        "gard_noisy_graph": "GARD noisy graph",
        "gard_wrong_graph": "GARD wrong graph",
        "gard_lsknn_graph": "GARD LS-kNN graph",
    }
    plt.figure(figsize=(8.0, 5.0))
    for method in methods:
        rows = [
            row
            for row in aggregate
            if row["protocol"] == "random_fraction"
            and row["assignment_regime"] == "oracle"
            and row["method"] == method
        ]
        rows.sort(key=lambda row: float(row["budget_label"]), reverse=True)
        xs = [float(row["budget_label"]) for row in rows]
        ys = [row["snapshot_mse_mean"] for row in rows]
        plt.plot(xs, ys, marker="o", label=labels[method])
    plt.yscale("log")
    plt.gca().invert_xaxis()
    plt.axhline(0.10, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    plt.xlabel("Observed batch-gradient fraction")
    plt.ylabel("Snapshot MSE (mean over seeds)")
    plt.title("Round 05: Graph and Baseline Stress Test (Oracle Assignment)")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "round05_random_oracle_assignment.png", dpi=180)
    plt.close()


def plot_assignment_stress(aggregate: list[dict]) -> None:
    methods = ["shard_style_ls", "ridge_ls", "gard_oracle_graph", "gard_noisy_graph", "gard_wrong_graph"]
    assignments = ["oracle", "wrong20", "wrong40", "unknown_random"]
    x = np.arange(len(assignments), dtype=np.float64)
    width = 0.15
    plt.figure(figsize=(8.0, 5.0))
    for idx, method in enumerate(methods):
        ys = []
        for assignment in assignments:
            row = next(
                row
                for row in aggregate
                if row["protocol"] == "random_fraction"
                and row["budget_label"] == "0.40"
                and row["assignment_regime"] == assignment
                and row["method"] == method
            )
            ys.append(row["snapshot_mse_mean"])
        plt.bar(x + (idx - 2) * width, ys, width=width, label=method)
    plt.yscale("log")
    plt.axhline(0.10, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    plt.xticks(x, assignments)
    plt.ylabel("Snapshot MSE at 40% observed rows")
    plt.title("Round 05: Assignment Error Stress Test")
    plt.grid(True, axis="y", which="both", alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "round05_assignment_stress.png", dpi=180)
    plt.close()


def main() -> None:
    logger = setup_logging()
    cfg = SimConfig()
    seeds = [3, 7, 11, 19, 23, 29, 31, 37]
    assignment_regimes = ["oracle", "wrong20", "wrong40", "unknown_random"]
    scenarios: list[tuple[str, float | int]] = [
        *[("random_fraction", fraction) for fraction in cfg.random_fractions],
        *[("prefix_epochs", epochs) for epochs in cfg.prefix_epochs],
    ]
    logger.info("Round 05 config: %s", cfg)
    logger.info("Seeds=%s", seeds)
    logger.info("Assignment regimes=%s", assignment_regimes)
    logger.info("Scenarios=%s", scenarios)

    t0 = time.perf_counter()
    all_runs = []
    for protocol, budget in scenarios:
        for assignment in assignment_regimes:
            logger.info("=== protocol=%s budget=%s assignment=%s ===", protocol, budget, assignment)
            for seed in seeds:
                run = run_one(seed, protocol, budget, assignment, cfg)
                all_runs.append(run)
                methods = run["methods"]
                logger.info(
                    "seed=%d rows=%d/%d overlap=%.2f rank=%d "
                    "ls=%.4g ridge=%.4g lowrank=%.4g gard_oracle=%.4g gard_noisy=%.4g gard_wrong=%.4g",
                    seed,
                    run["observed_batch_gradients"],
                    run["full_batch_gradients"],
                    run["assignment_member_overlap"],
                    run["incidence_rank_with_mean_anchor"],
                    methods["shard_style_ls"]["snapshot_mse"],
                    methods["ridge_ls"]["snapshot_mse"],
                    methods["low_rank_pca"]["snapshot_mse"],
                    methods["gard_oracle_graph"]["snapshot_mse"],
                    methods["gard_noisy_graph"]["snapshot_mse"],
                    methods["gard_wrong_graph"]["snapshot_mse"],
                )

    aggregate = aggregate_runs(all_runs)
    threat_reduction = compute_threat_reduction(aggregate, cfg)
    results = {
        "benchmark": "round05_assignment_and_graph_stress_stage2",
        "method_hypothesis": "GARD is useful only as a validation-gated graph prior under good assignment and graph side information.",
        "config": asdict(cfg),
        "seeds": seeds,
        "assignment_regimes": assignment_regimes,
        "scenarios": [{"protocol": protocol, "budget": budget} for protocol, budget in scenarios],
        "aggregate": aggregate,
        "threat_reduction": threat_reduction,
        "per_seed": all_runs,
        "runtime_sec": time.perf_counter() - t0,
        "honesty_notes": [
            "Regularization lambdas are selected by held-out observed batch rows, not snapshot MSE.",
            "Assignment-corruption regimes generate observations from true hidden incidence but solve with attacker-assumed incidence.",
            "The oracle graph is deliberately favorable; noisy, wrong, co-occurrence, and LS-kNN graphs stress mismatch.",
            "The benchmark is still synthetic Stage 2 and does not include Stage 1 gradient inversion or Stage 3 input inversion.",
        ],
    }
    (ARTIFACTS / "round05_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot_random_oracle(aggregate)
    plot_assignment_stress(aggregate)
    logger.info("Wrote %s", ARTIFACTS / "round05_metrics.json")
    logger.info("Wrote %s", ARTIFACTS / "round05_random_oracle_assignment.png")
    logger.info("Wrote %s", ARTIFACTS / "round05_assignment_stress.png")
    logger.info("Runtime %.2fs", results["runtime_sec"])


if __name__ == "__main__":
    main()
