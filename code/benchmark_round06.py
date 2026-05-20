#!/usr/bin/env python3
"""Round-06 Stage-2 benchmark: assignment-first soft incidence recovery.

This script tests JASPER (Joint Assignment Sinkhorn Projection and Estimation
Recovery), an alternating Stage-2 solver for settings where true per-batch
membership is missing or corrupted.  It intentionally treats true snapshots as
latent: methods receive only observed batch-average snapshots, epoch/batch
metadata, and an optional noisy incidence prior.

GARD is retained as a conditional graph prior baseline, not as the main claim.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


RUN_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = RUN_ROOT / "artifacts"
LOGS = RUN_ROOT / "logs"


@dataclass(frozen=True)
class SimConfig:
    n_samples: int = 36
    batch_size: int = 4
    dim_g: int = 24
    true_rank: int = 4
    n_epochs: int = 5
    noise_std: float = 0.01
    mean_anchor_weight: float = 20.0
    ridge_lambda: float = 0.03
    gard_lambda: float = 0.3
    jasper_graph_lambda: float = 0.03
    sinkhorn_tau: float = 0.35
    sinkhorn_iters: int = 80
    jasper_iters: int = 16
    jasper_damping: float = 0.45
    random_fractions: tuple[float, ...] = (1.0, 0.8, 0.6, 0.4, 0.25)
    prefix_epochs: tuple[int, ...] = (5, 4, 3, 2, 1)
    target_snapshot_mse: float = 0.10


ASSIGNMENT_REGIMES = ("oracle", "corrupt25", "corrupt50", "unknown_epoch")
SEEDS = (3, 7, 11, 19, 23, 29)


def setup_logging() -> logging.Logger:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("round06")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(LOGS / "experiment_round06.log", mode="w")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def make_low_rank_snapshots(cfg: SimConfig, seed: int) -> np.ndarray:
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


def row_members(row: np.ndarray) -> np.ndarray:
    return np.flatnonzero(row > 0.0)


def row_from_members(members: Iterable[int], n_samples: int) -> np.ndarray:
    members = np.array(list(members), dtype=np.int64)
    row = np.zeros(n_samples, dtype=np.float64)
    row[members] = 1.0 / len(members)
    return row


def make_incidence(cfg: SimConfig, seed: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    rng = np.random.default_rng(seed + 10_000)
    k_batches = cfg.n_samples // cfg.batch_size
    rows = []
    metadata: list[tuple[int, int]] = []
    for epoch in range(cfg.n_epochs):
        perm = rng.permutation(cfg.n_samples)
        for batch in range(k_batches):
            members = perm[batch * cfg.batch_size : (batch + 1) * cfg.batch_size]
            rows.append(row_from_members(members, cfg.n_samples))
            metadata.append((epoch, batch))
    return np.vstack(rows), metadata


def select_rows(
    h_full: np.ndarray,
    metadata: list[tuple[int, int]],
    protocol: str,
    budget: float | int,
    seed: int,
) -> np.ndarray:
    if protocol == "random_fraction":
        rng = np.random.default_rng(seed + 20_000 + int(round(float(budget) * 1000)))
        n_keep = max(1, int(round(float(budget) * h_full.shape[0])))
        return np.sort(rng.choice(h_full.shape[0], size=n_keep, replace=False))
    if protocol == "prefix_epochs":
        return np.array([i for i, (epoch, _) in enumerate(metadata) if epoch < int(budget)])
    raise ValueError(f"unknown protocol {protocol}")


def corrupt_or_hide_incidence(
    h_true: np.ndarray,
    metadata_obs: list[tuple[int, int]],
    cfg: SimConfig,
    regime: str,
    seed: int,
) -> tuple[np.ndarray, float, float]:
    """Return attacker incidence prior plus true-member overlap and known fraction."""
    rng = np.random.default_rng(seed + 30_000 + {"oracle": 0, "corrupt25": 25, "corrupt50": 50, "unknown_epoch": 99}[regime])
    if regime == "oracle":
        return h_true.copy(), 1.0, 1.0

    all_ids = np.arange(cfg.n_samples)
    prior_rows = []
    overlaps = []
    if regime in {"corrupt25", "corrupt50"}:
        n_replace = 1 if regime == "corrupt25" else 2
        known_fraction = 1.0 - n_replace / cfg.batch_size
        for row in h_true:
            true_members = row_members(row)
            keep = rng.choice(true_members, size=cfg.batch_size - n_replace, replace=False)
            pool = np.setdiff1d(all_ids, keep, assume_unique=False)
            fill = rng.choice(pool, size=n_replace, replace=False)
            guessed = np.concatenate([keep, fill])
            prior_rows.append(row_from_members(guessed, cfg.n_samples))
            overlaps.append(len(set(true_members).intersection(set(guessed))) / cfg.batch_size)
        return np.vstack(prior_rows), float(np.mean(overlaps)), float(known_fraction)

    # Unknown within each observed epoch: the attacker knows row/epoch counts but
    # has no sample-level membership.  Fixed-incidence baselines receive a
    # random feasible partition for the observed rows of each epoch.
    prior = np.zeros_like(h_true)
    for epoch in sorted({epoch for epoch, _ in metadata_obs}):
        row_idx = [i for i, (e, _) in enumerate(metadata_obs) if e == epoch]
        perm = rng.permutation(cfg.n_samples)
        cursor = 0
        for i in row_idx:
            if cursor + cfg.batch_size > cfg.n_samples:
                perm = rng.permutation(cfg.n_samples)
                cursor = 0
            members = perm[cursor : cursor + cfg.batch_size]
            cursor += cfg.batch_size
            prior[i] = row_from_members(members, cfg.n_samples)
            true_members = row_members(h_true[i])
            overlaps.append(len(set(true_members).intersection(set(members))) / cfg.batch_size)
    return prior, float(np.mean(overlaps)), 0.0


def estimate_mean_snapshot(m_obs: np.ndarray) -> np.ndarray:
    # Mean of observed batch averages is attacker-observable and is unbiased when
    # the observed rows cover samples approximately evenly.
    return m_obs.mean(axis=0)


def chain_laplacian(n_samples: int) -> np.ndarray:
    weights = np.zeros((n_samples, n_samples), dtype=np.float64)
    for i in range(n_samples - 1):
        weights[i, i + 1] = 1.0
        weights[i + 1, i] = 1.0
    degree = np.diag(weights.sum(axis=1))
    return degree - weights


def solve_snapshot_map(
    h_obs: np.ndarray,
    m_obs: np.ndarray,
    mean_snapshot: np.ndarray,
    cfg: SimConfig,
    ridge_lambda: float = 0.0,
    graph_lambda: float = 0.0,
    laplacian: np.ndarray | None = None,
) -> np.ndarray:
    n = cfg.n_samples
    anchor = np.ones((n, 1), dtype=np.float64) / n
    lhs = h_obs.T @ h_obs + cfg.mean_anchor_weight * (anchor @ anchor.T)
    if ridge_lambda > 0.0:
        lhs = lhs + ridge_lambda * np.eye(n)
    if graph_lambda > 0.0 and laplacian is not None:
        lhs = lhs + graph_lambda * laplacian
    rhs = h_obs.T @ m_obs + cfg.mean_anchor_weight * anchor @ mean_snapshot[None, :]
    return np.linalg.solve(lhs + 1e-8 * np.eye(n), rhs)


def assignment_prior_power(regime: str) -> float:
    return {
        "oracle": 4.0,
        "corrupt25": 1.5,
        "corrupt50": 0.75,
        "unknown_epoch": 0.0,
    }[regime]


def sinkhorn_epoch_capacity(
    cost: np.ndarray,
    prior_h: np.ndarray,
    cfg: SimConfig,
    prior_power: float,
) -> np.ndarray:
    # A is a soft membership matrix with row sum B and per-epoch column capacity
    # at most one.  H = A / B is the soft averaging incidence.
    cost = cost - np.min(cost, axis=1, keepdims=True)
    scale = np.median(cost)
    scale = max(float(scale), 1e-8)
    kernel = np.exp(-cost / (cfg.sinkhorn_tau * scale))
    if prior_power > 0.0:
        prior_membership = np.clip(prior_h * cfg.batch_size, 0.0, 1.0)
        prior_weight = 0.05 + prior_membership
        kernel *= prior_weight**prior_power
    kernel = np.maximum(kernel, 1e-12)
    a = kernel.copy()
    for _ in range(cfg.sinkhorn_iters):
        row_sums = a.sum(axis=1, keepdims=True)
        a *= cfg.batch_size / np.maximum(row_sums, 1e-12)
        col_sums = a.sum(axis=0, keepdims=True)
        too_large = col_sums > 1.0
        if np.any(too_large):
            a[:, too_large.ravel()] *= 1.0 / np.maximum(col_sums[:, too_large.ravel()], 1e-12)
    row_sums = a.sum(axis=1, keepdims=True)
    a *= cfg.batch_size / np.maximum(row_sums, 1e-12)
    return np.clip(a, 0.0, 1.0) / cfg.batch_size


def update_soft_assignments(
    s_hat: np.ndarray,
    m_obs: np.ndarray,
    h_prior: np.ndarray,
    metadata_obs: list[tuple[int, int]],
    cfg: SimConfig,
    regime: str,
) -> np.ndarray:
    h_new = np.zeros_like(h_prior)
    prior_power = assignment_prior_power(regime)
    for epoch in sorted({epoch for epoch, _ in metadata_obs}):
        idx = [i for i, (e, _) in enumerate(metadata_obs) if e == epoch]
        if not idx:
            continue
        s_norm = np.sum(s_hat * s_hat, axis=1)[None, :]
        m_norm = np.sum(m_obs[idx] * m_obs[idx], axis=1)[:, None]
        cost = s_norm + m_norm - 2.0 * m_obs[idx] @ s_hat.T
        h_new[idx] = sinkhorn_epoch_capacity(cost, h_prior[idx], cfg, prior_power)
    return h_new


def hard_overlap(h_soft: np.ndarray, h_true: np.ndarray, cfg: SimConfig) -> float:
    overlaps = []
    for soft_row, true_row in zip(h_soft, h_true):
        pred = set(np.argsort(soft_row)[-cfg.batch_size :].tolist())
        true = set(row_members(true_row).tolist())
        overlaps.append(len(pred.intersection(true)) / cfg.batch_size)
    return float(np.mean(overlaps))


def soft_true_mass(h_soft: np.ndarray, h_true: np.ndarray) -> float:
    masses = []
    for soft_row, true_row in zip(h_soft, h_true):
        masses.append(float(np.sum(soft_row[row_members(true_row)])))
    return float(np.mean(masses))


def run_jasper(
    h_prior: np.ndarray,
    m_obs: np.ndarray,
    metadata_obs: list[tuple[int, int]],
    cfg: SimConfig,
    regime: str,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    mean_snapshot = estimate_mean_snapshot(m_obs)
    lap = chain_laplacian(cfg.n_samples)
    h_soft = h_prior.copy()
    residual_trace = []
    s_hat = solve_snapshot_map(
        h_soft,
        m_obs,
        mean_snapshot,
        cfg,
        ridge_lambda=cfg.ridge_lambda,
        graph_lambda=cfg.jasper_graph_lambda,
        laplacian=lap,
    )
    for _ in range(cfg.jasper_iters):
        h_candidate = update_soft_assignments(s_hat, m_obs, h_prior, metadata_obs, cfg, regime)
        h_soft = (1.0 - cfg.jasper_damping) * h_soft + cfg.jasper_damping * h_candidate
        s_hat = solve_snapshot_map(
            h_soft,
            m_obs,
            mean_snapshot,
            cfg,
            ridge_lambda=cfg.ridge_lambda,
            graph_lambda=cfg.jasper_graph_lambda,
            laplacian=lap,
        )
        residual_trace.append(float(np.mean((h_soft @ s_hat - m_obs) ** 2)))
    return s_hat, h_soft, residual_trace


def snapshot_metrics(
    s_hat: np.ndarray,
    s_true: np.ndarray,
    h_eval: np.ndarray,
    h_used: np.ndarray,
    m_obs: np.ndarray,
) -> dict:
    return {
        "snapshot_mse": float(np.mean((s_hat - s_true) ** 2)),
        "relative_fro_error": float(
            np.linalg.norm(s_hat - s_true, "fro") / (np.linalg.norm(s_true, "fro") + 1e-12)
        ),
        "true_observed_batch_mse": float(np.mean((h_eval @ s_hat - m_obs) ** 2)),
        "used_observed_batch_mse": float(np.mean((h_used @ s_hat - m_obs) ** 2)),
    }


def run_one(
    seed: int,
    protocol: str,
    budget: float | int,
    assignment_regime: str,
    cfg: SimConfig,
) -> dict:
    s_true = make_low_rank_snapshots(cfg, seed)
    h_full, metadata = make_incidence(cfg, seed)
    keep = select_rows(h_full, metadata, protocol, budget, seed)
    metadata_obs = [metadata[i] for i in keep]
    h_true_obs = h_full[keep]
    rng = np.random.default_rng(seed + 50_000 + len(keep))
    m_obs = h_true_obs @ s_true + cfg.noise_std * rng.normal(size=(len(keep), cfg.dim_g))
    h_prior, prior_overlap, known_fraction = corrupt_or_hide_incidence(
        h_true_obs, metadata_obs, cfg, assignment_regime, seed
    )
    mean_est = estimate_mean_snapshot(m_obs)
    lap = chain_laplacian(cfg.n_samples)

    method_outputs: dict[str, dict] = {}
    s_ls = solve_snapshot_map(h_prior, m_obs, mean_est, cfg)
    method_outputs["shard_fixed_incidence_ls"] = {
        **snapshot_metrics(s_ls, s_true, h_true_obs, h_prior, m_obs),
        "hard_assignment_overlap": hard_overlap(h_prior, h_true_obs, cfg),
        "soft_true_member_mass": soft_true_mass(h_prior, h_true_obs),
    }

    s_gard = solve_snapshot_map(
        h_prior,
        m_obs,
        mean_est,
        cfg,
        ridge_lambda=cfg.ridge_lambda,
        graph_lambda=cfg.gard_lambda,
        laplacian=lap,
    )
    method_outputs["gard_conditional_prior"] = {
        **snapshot_metrics(s_gard, s_true, h_true_obs, h_prior, m_obs),
        "hard_assignment_overlap": hard_overlap(h_prior, h_true_obs, cfg),
        "soft_true_member_mass": soft_true_mass(h_prior, h_true_obs),
        "graph_lambda": cfg.gard_lambda,
    }

    s_jasper, h_jasper, residual_trace = run_jasper(
        h_prior, m_obs, metadata_obs, cfg, assignment_regime
    )
    method_outputs["jasper_joint_soft_incidence"] = {
        **snapshot_metrics(s_jasper, s_true, h_true_obs, h_jasper, m_obs),
        "hard_assignment_overlap": hard_overlap(h_jasper, h_true_obs, cfg),
        "soft_true_member_mass": soft_true_mass(h_jasper, h_true_obs),
        "residual_trace": residual_trace,
        "graph_lambda": cfg.jasper_graph_lambda,
        "sinkhorn_tau": cfg.sinkhorn_tau,
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
        "assignment_prior_member_overlap": prior_overlap,
        "assignment_known_fraction": known_fraction,
        "observed_batch_gradients": int(len(keep)),
        "full_batch_gradients": int(h_full.shape[0]),
        "observed_epochs": sorted({int(metadata[i][0]) for i in keep}),
        "observed_rows_by_epoch": {
            str(epoch): int(sum(1 for e, _ in metadata_obs if e == epoch))
            for epoch in range(cfg.n_epochs)
        },
        "methods": method_outputs,
    }


def aggregate_runs(runs: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str, str], list[tuple[dict, dict]]] = {}
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
        true_resid = np.array([item[1]["true_observed_batch_mse"] for item in items], dtype=np.float64)
        used_resid = np.array([item[1]["used_observed_batch_mse"] for item in items], dtype=np.float64)
        hard_overlap = np.array([item[1]["hard_assignment_overlap"] for item in items], dtype=np.float64)
        soft_mass = np.array([item[1]["soft_true_member_mass"] for item in items], dtype=np.float64)
        aggregate.append(
            {
                "protocol": protocol,
                "budget_label": budget_label,
                "assignment_regime": assignment,
                "method": method,
                "observed_batch_gradients_mean": float(rows.mean()),
                "snapshot_mse_mean": float(mse.mean()),
                "snapshot_mse_std": float(mse.std(ddof=0)),
                "true_observed_batch_mse_mean": float(true_resid.mean()),
                "used_observed_batch_mse_mean": float(used_resid.mean()),
                "hard_assignment_overlap_mean": float(hard_overlap.mean()),
                "soft_true_member_mass_mean": float(soft_mass.mean()),
                "n_seeds": len(items),
            }
        )
    return aggregate


def compute_threat_reduction(aggregate: list[dict], cfg: SimConfig) -> list[dict]:
    rows = []
    methods = [
        "shard_fixed_incidence_ls",
        "gard_conditional_prior",
        "jasper_joint_soft_incidence",
    ]
    for protocol in ("random_fraction", "prefix_epochs"):
        for assignment in ASSIGNMENT_REGIMES:
            for method in methods:
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
                            "budget_label": None,
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
                            1.0
                            - best["observed_batch_gradients_mean"]
                            / (cfg.n_epochs * cfg.n_samples / cfg.batch_size)
                        ),
                    }
                )
    return rows


def write_plots_if_available(aggregate: list[dict]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return write_svg_plots(aggregate)

    written = []
    methods = [
        ("shard_fixed_incidence_ls", "SHARD fixed LS"),
        ("gard_conditional_prior", "GARD conditional"),
        ("jasper_joint_soft_incidence", "JASPER joint soft"),
    ]
    for assignment in ("oracle", "corrupt25", "corrupt50", "unknown_epoch"):
        plt.figure(figsize=(7.5, 4.8))
        for method, label in methods:
            rows = [
                row
                for row in aggregate
                if row["protocol"] == "random_fraction"
                and row["assignment_regime"] == assignment
                and row["method"] == method
            ]
            rows.sort(key=lambda row: float(row["budget_label"]), reverse=True)
            xs = [float(row["budget_label"]) for row in rows]
            ys = [row["snapshot_mse_mean"] for row in rows]
            plt.plot(xs, ys, marker="o", label=label)
        plt.yscale("log")
        plt.gca().invert_xaxis()
        plt.axhline(0.10, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
        plt.xlabel("Observed batch-average fraction")
        plt.ylabel("Snapshot MSE")
        plt.title(f"Round 06 Stage-2 Recovery: {assignment}")
        plt.grid(True, which="both", alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        path = ARTIFACTS / f"round06_mse_{assignment}.png"
        plt.savefig(path, dpi=180)
        plt.close()
        written.append(str(path))

    plt.figure(figsize=(7.5, 4.8))
    assignments = list(ASSIGNMENT_REGIMES)
    x = np.arange(len(assignments))
    width = 0.25
    for offset, (method, label) in enumerate(methods):
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
        plt.bar(x + (offset - 1) * width, ys, width=width, label=label)
    plt.yscale("log")
    plt.axhline(0.10, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    plt.xticks(x, assignments)
    plt.ylabel("Snapshot MSE at 40% rows")
    plt.title("Round 06 Assignment Stress")
    plt.grid(True, axis="y", which="both", alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    path = ARTIFACTS / "round06_assignment_stress.png"
    plt.savefig(path, dpi=180)
    plt.close()
    written.append(str(path))
    return written


def _svg_line_plot(
    series: list[tuple[str, list[tuple[float, float]], str]],
    title: str,
    path: Path,
    y_log: bool = True,
) -> None:
    width, height = 780, 500
    left, right, top, bottom = 70, 25, 45, 65
    plot_w = width - left - right
    plot_h = height - top - bottom
    all_x = [x for _, points, _ in series for x, _ in points]
    all_y = [y for _, points, _ in series for _, y in points if y > 0]
    x_min, x_max = min(all_x), max(all_x)
    if y_log:
        y_vals = [math.log10(max(y, 1e-6)) for y in all_y]
    else:
        y_vals = all_y
    y_min, y_max = min(y_vals), max(y_vals)
    y_min = min(y_min, math.log10(0.08) if y_log else 0.08)
    y_max = max(y_max, math.log10(2.0) if y_log else 2.0)

    def px(x: float) -> float:
        # Higher observation fraction appears on the left, matching the PNG plots.
        return left + (x_max - x) / max(x_max - x_min, 1e-12) * plot_w

    def py(y: float) -> float:
        val = math.log10(max(y, 1e-6)) if y_log else y
        return top + (y_max - val) / max(y_max - y_min, 1e-12) * plot_h

    target_y = py(0.10)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="25" text-anchor="middle" font-family="sans-serif" font-size="18">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{target_y:.1f}" x2="{left + plot_w}" y2="{target_y:.1f}" stroke="#333" stroke-dasharray="5,5" opacity="0.6"/>',
        f'<text x="{left + plot_w - 5}" y="{target_y - 6:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">MSE=0.10</text>',
    ]
    for tick in sorted(set(all_x), reverse=True):
        x = px(tick)
        parts.append(f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 5}" stroke="#333"/>')
        parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 22}" text-anchor="middle" font-family="sans-serif" font-size="11">{tick:.2f}</text>')
    for y_tick in (0.1, 0.3, 1.0, 3.0, 10.0):
        if math.log10(y_tick) < y_min or math.log10(y_tick) > y_max:
            continue
        y = py(y_tick)
        parts.append(f'<line x1="{left - 5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#333"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{y_tick:g}</text>')
    parts.append(f'<text x="{left + plot_w/2}" y="{height - 18}" text-anchor="middle" font-family="sans-serif" font-size="13">Observed batch-average fraction</text>')
    parts.append(f'<text x="18" y="{top + plot_h/2}" transform="rotate(-90 18 {top + plot_h/2})" text-anchor="middle" font-family="sans-serif" font-size="13">Snapshot MSE</text>')
    legend_y = top + 12
    for label, points, color in series:
        coords = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in points)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.3" points="{coords}"/>')
        for x, y in points:
            parts.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.5" fill="{color}"/>')
        parts.append(f'<line x1="{left + plot_w - 170}" y1="{legend_y}" x2="{left + plot_w - 145}" y2="{legend_y}" stroke="{color}" stroke-width="2.3"/>')
        parts.append(f'<text x="{left + plot_w - 138}" y="{legend_y + 4}" font-family="sans-serif" font-size="11">{label}</text>')
        legend_y += 17
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_svg_plots(aggregate: list[dict]) -> list[str]:
    colors = {
        "shard_fixed_incidence_ls": "#4C78A8",
        "gard_conditional_prior": "#F58518",
        "jasper_joint_soft_incidence": "#54A24B",
    }
    labels = {
        "shard_fixed_incidence_ls": "SHARD fixed LS",
        "gard_conditional_prior": "GARD conditional",
        "jasper_joint_soft_incidence": "JASPER joint soft",
    }
    methods = list(labels)
    written = []
    for assignment in ASSIGNMENT_REGIMES:
        series = []
        for method in methods:
            rows = [
                row
                for row in aggregate
                if row["protocol"] == "random_fraction"
                and row["assignment_regime"] == assignment
                and row["method"] == method
            ]
            rows.sort(key=lambda row: float(row["budget_label"]), reverse=True)
            points = [
                (float(row["budget_label"]), float(row["snapshot_mse_mean"]))
                for row in rows
            ]
            series.append((labels[method], points, colors[method]))
        path = ARTIFACTS / f"round06_mse_{assignment}.svg"
        _svg_line_plot(series, f"Round 06 Stage-2 Recovery: {assignment}", path)
        written.append(str(path))
    return written


def summarize_headlines(aggregate: list[dict], threat: list[dict]) -> dict:
    def metric(protocol: str, budget: str, assignment: str, method: str, field: str) -> float:
        row = next(
            row
            for row in aggregate
            if row["protocol"] == protocol
            and row["budget_label"] == budget
            and row["assignment_regime"] == assignment
            and row["method"] == method
        )
        return float(row[field])

    methods = [
        "shard_fixed_incidence_ls",
        "gard_conditional_prior",
        "jasper_joint_soft_incidence",
    ]
    stress_40 = {
        assignment: {
            method: {
                "snapshot_mse": metric("random_fraction", "0.40", assignment, method, "snapshot_mse_mean"),
                "hard_assignment_overlap": metric(
                    "random_fraction", "0.40", assignment, method, "hard_assignment_overlap_mean"
                ),
            }
            for method in methods
        }
        for assignment in ASSIGNMENT_REGIMES
    }
    return {
        "random_fraction_0.40_assignment_stress": stress_40,
        "target_mse": 0.10,
        "target_reachability": threat,
    }


def main() -> None:
    logger = setup_logging()
    cfg = SimConfig()
    scenarios: list[tuple[str, float | int]] = [
        *[("random_fraction", fraction) for fraction in cfg.random_fractions],
        *[("prefix_epochs", epochs) for epochs in cfg.prefix_epochs],
    ]
    logger.info("Round 06 config: %s", cfg)
    logger.info("Seeds=%s", SEEDS)
    logger.info("Assignment regimes=%s", ASSIGNMENT_REGIMES)
    logger.info("Scenarios=%s", scenarios)
    t0 = time.perf_counter()
    all_runs = []
    for protocol, budget in scenarios:
        for assignment in ASSIGNMENT_REGIMES:
            logger.info("=== protocol=%s budget=%s assignment=%s ===", protocol, budget, assignment)
            for seed in SEEDS:
                run = run_one(seed, protocol, budget, assignment, cfg)
                all_runs.append(run)
                methods = run["methods"]
                logger.info(
                    "seed=%d rows=%d/%d prior_overlap=%.2f "
                    "ls_mse=%.4g gard_mse=%.4g jasper_mse=%.4g "
                    "jasper_overlap=%.3f jasper_true_resid=%.4g",
                    seed,
                    run["observed_batch_gradients"],
                    run["full_batch_gradients"],
                    run["assignment_prior_member_overlap"],
                    methods["shard_fixed_incidence_ls"]["snapshot_mse"],
                    methods["gard_conditional_prior"]["snapshot_mse"],
                    methods["jasper_joint_soft_incidence"]["snapshot_mse"],
                    methods["jasper_joint_soft_incidence"]["hard_assignment_overlap"],
                    methods["jasper_joint_soft_incidence"]["true_observed_batch_mse"],
                )
    aggregate = aggregate_runs(all_runs)
    threat = compute_threat_reduction(aggregate, cfg)
    plots = write_plots_if_available(aggregate)
    results = {
        "benchmark": "round06_jasper_assignment_first_stage2",
        "method": "JASPER: Joint Assignment Sinkhorn Projection and Estimation Recovery",
        "objective": (
            "min_{S,A} ||(A/B)S-M||_F^2 + lambda_g Tr(S^T L S) + lambda_r ||S||_F^2 "
            "+ entropic/KL assignment terms, subject to row batch-size and per-epoch sample-capacity constraints"
        ),
        "config": asdict(cfg),
        "seeds": list(SEEDS),
        "assignment_regimes": list(ASSIGNMENT_REGIMES),
        "scenarios": [{"protocol": protocol, "budget": budget} for protocol, budget in scenarios],
        "aggregate": aggregate,
        "threat_reduction": threat,
        "headline": summarize_headlines(aggregate, threat),
        "per_seed": all_runs,
        "plots": plots,
        "runtime_sec": time.perf_counter() - t0,
        "honesty_notes": [
            "Methods do not receive true snapshots or true mean anchors; the mean anchor is estimated from observed batch averages.",
            "Unknown-within-epoch remains label-symmetric and poorly identified in this benchmark.",
            "JASPER uses graph smoothing only as a weak regularizer after soft assignment updates.",
            "The synthetic benchmark is Stage 2 only; Stage 1 gradient extraction and Stage 3 input inversion are not modeled.",
        ],
    }
    out = ARTIFACTS / "round06_metrics.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out)
    if plots:
        for path in plots:
            logger.info("Wrote %s", path)
    else:
        logger.info("Matplotlib unavailable or failed; no plots written")
    logger.info("Runtime %.2fs", results["runtime_sec"])


if __name__ == "__main__":
    main()
