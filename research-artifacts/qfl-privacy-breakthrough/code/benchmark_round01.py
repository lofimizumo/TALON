#!/usr/bin/env python3
"""Round-01 QFL privacy breakthrough — multi-lane survey benchmark.

Lanes implemented (>=2 required):
  1. GARD-SPARSE — Stage-2 with fewer observed batch rows (parent Round-05 extension)
  2. QFL-SHIELD — client subspace mask + snapshot noise before gradient publish
  3. LEAK-CERT — tier-conditioned analytic MSE lower bounds vs LASA-QTERM attacks

Survey-only (not fully implemented): ASSIGN-LOCK assignment-hiding (Round 02).

Uses vendor/shard_sim via code/_paths.py; imports qterm_attack from sibling run.
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


def hungarian_snapshot_mse(recovered: np.ndarray, truth: np.ndarray) -> float:
    from scipy.optimize import linear_sum_assignment

    r_sq = np.sum(recovered**2, axis=1, keepdims=True)
    t_sq = np.sum(truth**2, axis=1, keepdims=True)
    dist = r_sq + t_sq.T - 2.0 * recovered @ truth.T
    np.maximum(dist, 0.0, out=dist)
    row, col = linear_sum_assignment(dist)
    return float(np.mean((recovered[row] - truth[col]) ** 2))


# ---------------------------------------------------------------------------
# Stage-2 sparse-row simulator (GARD-SPARSE), adapted from parent Round 05
# ---------------------------------------------------------------------------


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
    graph_lambda_grid: tuple[float, ...] = (0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
    random_fractions: tuple[float, ...] = (1.0, 0.6, 0.4, 0.25, 0.15)
    target_snapshot_mse: float = 0.10


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


def select_rows(
    h_full: np.ndarray,
    metadata: list[tuple[int, int]],
    fraction: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed + 20_000 + int(round(fraction * 1000)))
    n_rows = h_full.shape[0]
    n_keep = max(1, int(round(fraction * n_rows)))
    return np.sort(rng.choice(n_rows, size=n_keep, replace=False))


def chain_laplacian(n_samples: int) -> np.ndarray:
    weights = np.zeros((n_samples, n_samples), dtype=np.float64)
    for i in range(n_samples - 1):
        weights[i, i + 1] = 1.0
        weights[i + 1, i] = 1.0
    degrees = np.sum(weights, axis=1)
    return np.diag(degrees) - weights


def solve_map(
    h_obs: np.ndarray,
    m_obs: np.ndarray,
    mean_snapshot: np.ndarray,
    cfg: Stage2Config,
    graph_lambda: float = 0.0,
    laplacian: np.ndarray | None = None,
) -> np.ndarray:
    n_samples = cfg.n_samples
    anchor = np.ones((n_samples, 1), dtype=np.float64) / n_samples
    lhs = h_obs.T @ h_obs + cfg.mean_anchor_weight * (anchor @ anchor.T)
    if graph_lambda > 0.0 and laplacian is not None:
        lhs = lhs + graph_lambda * laplacian
    rhs = h_obs.T @ m_obs + cfg.mean_anchor_weight * anchor @ mean_snapshot[None, :]
    return np.linalg.solve(lhs + 1e-8 * np.eye(n_samples), rhs)


def validation_mse(s_hat: np.ndarray, h_val: np.ndarray, m_val: np.ndarray) -> float:
    if h_val.shape[0] == 0:
        return 0.0
    return float(np.mean((h_val @ s_hat - m_val) ** 2))


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
    rank_augmented: int,
) -> tuple[np.ndarray, float]:
    lambda_grid = (0.0,) if rank_augmented >= cfg.n_samples else cfg.graph_lambda_grid
    best_lambda = lambda_grid[0]
    best_score = float("inf")
    for lam in lambda_grid:
        s_candidate = solve_map(
            h_train, m_train, mean_snapshot, cfg, graph_lambda=lam, laplacian=lap
        )
        score = validation_mse(s_candidate, h_val, m_val)
        if score < best_score:
            best_score = score
            best_lambda = lam
    return (
        solve_map(h_all, m_all, mean_snapshot, cfg, graph_lambda=best_lambda, laplacian=lap),
        float(best_lambda),
    )


def snapshot_mse(s_hat: np.ndarray, s_true: np.ndarray) -> float:
    return float(np.mean((s_hat - s_true) ** 2))


def run_stage2_sparse(seed: int, fraction: float, cfg: Stage2Config) -> dict:
    s_true = make_low_rank_snapshots(cfg, seed)
    mean_snapshot = s_true.mean(axis=0)
    h_full, metadata = make_incidence(cfg, seed)
    keep = select_rows(h_full, metadata, fraction, seed)
    h_true = h_full[keep]
    rng = np.random.default_rng(seed + 50_000)
    m_obs = h_true @ s_true + cfg.noise_std * rng.normal(size=(len(keep), cfg.dim_g))
    train_idx, val_idx = split_train_val(len(keep), cfg, seed)
    h_train, m_train = h_true[train_idx], m_obs[train_idx]
    h_val, m_val = h_true[val_idx], m_obs[val_idx]
    rank_aug = int(
        np.linalg.matrix_rank(
            np.vstack([h_true, np.ones((1, cfg.n_samples), dtype=np.float64) / cfg.n_samples])
        )
    )
    lap = chain_laplacian(cfg.n_samples)
    s_ls = solve_map(h_true, m_obs, mean_snapshot, cfg)
    s_gard, lam = select_graph_map(
        h_train, m_train, h_val, m_val, h_true, m_obs, mean_snapshot, lap, cfg, rank_aug
    )
    return {
        "fraction": fraction,
        "observed_rows": int(len(keep)),
        "full_rows": int(h_full.shape[0]),
        "shard_ls_mse": snapshot_mse(s_ls, s_true),
        "gard_sparse_mse": snapshot_mse(s_gard, s_true),
        "selected_graph_lambda": lam,
        "rank_augmented": rank_aug,
    }


def aggregate_stage2(runs: list[dict], cfg: Stage2Config) -> dict:
    full_rows = runs[0]["full_rows"]
    out = []
    for fraction in cfg.random_fractions:
        subset = [r for r in runs if abs(r["fraction"] - fraction) < 1e-9]
        if not subset:
            continue
        obs_mean = float(np.mean([r["observed_rows"] for r in subset]))
        ls_mse = float(np.mean([r["shard_ls_mse"] for r in subset]))
        gard_mse = float(np.mean([r["gard_sparse_mse"] for r in subset]))
        out.append(
            {
                "fraction": fraction,
                "observed_rows_mean": obs_mean,
                "shard_ls_mse_mean": ls_mse,
                "gard_sparse_mse_mean": gard_mse,
                "improvement_factor_ls_over_gard": ls_mse / max(gard_mse, 1e-12),
            }
        )
    # Threat reduction at target MSE
    target = cfg.target_snapshot_mse
    shard_ok = [r for r in runs if r["shard_ls_mse"] <= target]
    gard_ok = [r for r in runs if r["gard_sparse_mse"] <= target]
    shard_min_rows = min((r["observed_rows"] for r in shard_ok), default=None)
    gard_min_rows = min((r["observed_rows"] for r in gard_ok), default=None)
    reduction = None
    if gard_min_rows is not None:
        reduction = 1.0 - gard_min_rows / full_rows
    return {
        "aggregate_by_fraction": out,
        "target_mse": target,
        "shard_min_rows_at_target": shard_min_rows,
        "gard_min_rows_at_target": gard_min_rows,
        "row_reduction_vs_full_gard": reduction,
        "breakthrough_A_25pct": reduction is not None and reduction >= 0.25,
    }


# ---------------------------------------------------------------------------
# QFL-SHIELD defense + LASA-QTERM attack (SurrogateQFL track)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QflConfig:
    n_samples: int = 32
    batch_size: int = 4
    n_epochs: int = 10
    dim_g: int = 32
    true_rank: int = 4
    noise_level: float = 0.01
    partial_rows: int = 7
    shield_rank: int = 2
    shield_noise_std: float = 0.15
    utility_metric: str = "gradient_mse"


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
    cfg: QflConfig,
    *,
    rank: int | None = None,
    noise_std: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """Client-side: low-rank subspace projection + optional isotropic noise."""
    r = cfg.shield_rank if rank is None else rank
    centered = true_snapshots - true_snapshots.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    r = min(r, len(s))
    low = (u[:, :r] * s[:r]) @ vt[:r]
    published = true_snapshots.mean(axis=0, keepdims=True) + low
    if noise_std > 0:
        rng = np.random.default_rng(42)
        published = published + noise_std * rng.normal(size=published.shape)
    energy_kept = float(np.sum(s[:r] ** 2) / (np.sum(s**2) + 1e-12))
    return published.astype(np.float64), {"rank": r, "noise_std": noise_std, "energy_kept": energy_kept}


def gradient_utility_loss(
    true_snaps: np.ndarray,
    published_snaps: np.ndarray,
    surrogate: SurrogateQFL,
    seed: int,
) -> float:
    """Mean batch-gradient MSE under same assignments (undefended vs defended)."""
    losses = []
    for e in range(3):
        batches = make_epoch_batches(true_snaps.shape[0], 4, seed + e)
        for batch in batches:
            g_true = simulate_gradients(true_snaps, [batch], surrogate)[1][0]
            g_pub = simulate_gradients(published_snaps, [batch], surrogate)[1][0]
            losses.append(float(np.mean((g_true - g_pub) ** 2)))
    return float(np.mean(losses))


def run_qfl_track(seed: int, cfg: QflConfig) -> dict:
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

    coeff_matrices: list[np.ndarray] = []
    batch_gradients: list[list[np.ndarray]] = []
    terminal_gradients: list[np.ndarray] = []

    for e in range(cfg.n_epochs):
        batches = make_epoch_batches(cfg.n_samples, cfg.batch_size, seed + e)
        a_e, grads_e = simulate_gradients(true_snapshots, batches, surrogate)
        coeff_matrices.append(a_e)
        batch_gradients.append(grads_e)
        terminal_gradients.append(np.mean(grads_e, axis=0))

    e_bar = attacker.level1_mean_recovery(coeff_matrices, batch_gradients)
    shard_s = attacker.level2_disaggregate(
        e_bar, coeff_matrices, batch_gradients, true_snapshots
    )
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
    r_t1p = qterm.recover(e_bar, coeff_matrices, batch_gradients=batch_gradients)
    t1p_mse = hungarian_snapshot_mse(r_t1p.snapshots, true_snapshots)

    # Defended publish path
    defended_snaps, shield_meta = apply_shield(
        true_snapshots,
        cfg,
        rank=cfg.shield_rank,
        noise_std=cfg.shield_noise_std,
    )
    coeff_d: list[np.ndarray] = []
    batch_gradients_d: list[list[np.ndarray]] = []
    for e in range(cfg.n_epochs):
        batches = make_epoch_batches(cfg.n_samples, cfg.batch_size, seed + e)
        a_e, grads_e = simulate_gradients(defended_snaps, batches, surrogate)
        coeff_d.append(a_e)
        batch_gradients_d.append(grads_e)
    e_bar_d = attacker.level1_mean_recovery(coeff_d, batch_gradients_d)
    r_def = qterm.recover(e_bar_d, coeff_d, batch_gradients=batch_gradients_d)
    defended_attack_mse = hungarian_snapshot_mse(r_def.snapshots, true_snapshots)
    util_loss = gradient_utility_loss(true_snapshots, defended_snaps, surrogate, seed)

    attack_reduction = 1.0 - defended_attack_mse / max(t1p_mse, 1e-12)

    return {
        "seed": seed,
        "shard_stage2_mse": shard_mse,
        "lasa_qterm_t1p_mse": t1p_mse,
        "defended_qterm_t1p_mse": defended_attack_mse,
        "attack_mse_reduction_vs_t1p": attack_reduction,
        "utility_gradient_mse": util_loss,
        "utility_loss_fraction": util_loss / max(
            float(np.mean(true_snapshots**2)), 1e-12
        ),
        "shield": shield_meta,
        "observed_terminal_rows": r_t1p.observed_terminal_gradient_rows,
    }


# ---------------------------------------------------------------------------
# LEAK-CERT — tier-conditioned bounds vs empirical attack MSE
# ---------------------------------------------------------------------------


def cert_naive_mean_only(e_bar: np.ndarray, true_snapshots: np.ndarray) -> float:
    """Passive broadcast MSE (T1 honest lower bound for spread)."""
    n = true_snapshots.shape[0]
    broadcast = np.tile(e_bar, (n, 1))
    return float(np.mean((broadcast - true_snapshots) ** 2))


def cert_rank_dof(
    n_obs_rows: int,
    n_samples: int,
    dim_g: int,
    population_var: float,
) -> float:
    """Residual variance when rank(obs) < N: cannot resolve all N individuals."""
    dof_gap = max(0, n_samples - n_obs_rows)
    if dof_gap <= 0:
        return 0.0
    return float(population_var * dof_gap / n_samples)


def run_leak_cert(seed: int, cfg: QflConfig) -> dict:
    true_snapshots = make_smooth_snapshots(cfg, seed)
    pop_var = float(np.var(true_snapshots))
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
        random_seed=seed,
    )
    coeff_matrices: list[np.ndarray] = []
    batch_gradients: list[list[np.ndarray]] = []
    terminal_gradients: list[np.ndarray] = []
    for e in range(cfg.n_epochs):
        batches = make_epoch_batches(cfg.n_samples, cfg.batch_size, seed + e)
        a_e, grads_e = simulate_gradients(true_snapshots, batches, surrogate)
        coeff_matrices.append(a_e)
        batch_gradients.append(grads_e)
        terminal_gradients.append(np.mean(grads_e, axis=0))

    e_bar = attacker.level1_mean_recovery(coeff_matrices, batch_gradients)

    qterm_t1 = QtermAttack(QtermConfig(tier=QtermTier.T1, n_samples=cfg.n_samples, random_seed=seed))
    r1 = qterm_t1.recover(e_bar, coeff_matrices, terminal_gradients=terminal_gradients)
    mse_t1 = hungarian_snapshot_mse(r1.snapshots, true_snapshots)

    qterm_t1p = QtermAttack(
        QtermConfig(
            tier=QtermTier.T1P,
            n_samples=cfg.n_samples,
            batch_size=cfg.batch_size,
            partial_rows_per_epoch=cfg.partial_rows,
            random_seed=seed,
        )
    )
    r1p = qterm_t1p.recover(e_bar, coeff_matrices, batch_gradients=batch_gradients)
    mse_t1p = hungarian_snapshot_mse(r1p.snapshots, true_snapshots)

    naive = cert_naive_mean_only(e_bar, true_snapshots)
    dof_bound = cert_rank_dof(
        r1p.observed_terminal_gradient_rows,
        cfg.n_samples,
        cfg.dim_g,
        pop_var,
    )
    tight = max(naive, dof_bound)

    return {
        "seed": seed,
        "empirical_t1_mse": mse_t1,
        "empirical_t1p_mse": mse_t1p,
        "cert_naive": naive,
        "cert_dof": dof_bound,
        "cert_tight": tight,
        "cert_vs_t1p_ratio": tight / max(mse_t1p, 1e-12),
        "cert_tighter_than_naive_2x": tight >= 2.0 * naive * 0.99,  # dof adds on top
        "n_obs_t1p": r1p.observed_terminal_gradient_rows,
    }


def setup_logging() -> logging.Logger:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    logging.getLogger("shard_sim.attacker").setLevel(logging.WARNING)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOGS / "experiment_round01.log", mode="w"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return logging.getLogger("round01")


def main() -> None:
    logger = setup_logging()
    t0 = time.perf_counter()

    stage2_cfg = Stage2Config()
    stage2_seeds = [3, 7, 11, 19, 23]
    stage2_runs = []
    for frac in stage2_cfg.random_fractions:
        for seed in stage2_seeds:
            run = run_stage2_sparse(seed, frac, stage2_cfg)
            run["seed"] = seed
            stage2_runs.append(run)
            logger.info(
                "GARD-SPARSE seed=%d frac=%.2f rows=%d ls=%.4f gard=%.4f",
                seed,
                frac,
                run["observed_rows"],
                run["shard_ls_mse"],
                run["gard_sparse_mse"],
            )
    stage2_summary = aggregate_stage2(stage2_runs, stage2_cfg)

    qfl_cfg = QflConfig()
    qfl_seeds = [3, 7, 11, 19, 23]
    qfl_runs = [run_qfl_track(s, qfl_cfg) for s in qfl_seeds]
    for r in qfl_runs:
        logger.info(
            "SHIELD seed=%d t1p=%.4f defended=%.4f util=%.4f reduction=%.1f%%",
            r["seed"],
            r["lasa_qterm_t1p_mse"],
            r["defended_qterm_t1p_mse"],
            r["utility_gradient_mse"],
            100 * r["attack_mse_reduction_vs_t1p"],
        )

    cert_runs = [run_leak_cert(s, qfl_cfg) for s in qfl_seeds]
    cert_agg = {
        "empirical_t1p_mse_mean": float(np.mean([r["empirical_t1p_mse"] for r in cert_runs])),
        "cert_tight_mean": float(np.mean([r["cert_tight"] for r in cert_runs])),
        "cert_naive_mean": float(np.mean([r["cert_naive"] for r in cert_runs])),
        "tight_over_naive_mean": float(
            np.mean([r["cert_tight"] / max(r["cert_naive"], 1e-12) for r in cert_runs])
        ),
        "cert_vs_t1p_ratio_mean": float(np.mean([r["cert_vs_t1p_ratio"] for r in cert_runs])),
    }

    qfl_agg = {
        "shard_mse_mean": float(np.mean([r["shard_stage2_mse"] for r in qfl_runs])),
        "t1p_mse_mean": float(np.mean([r["lasa_qterm_t1p_mse"] for r in qfl_runs])),
        "defended_mse_mean": float(np.mean([r["defended_qterm_t1p_mse"] for r in qfl_runs])),
        "attack_reduction_mean": float(
            np.mean([r["attack_mse_reduction_vs_t1p"] for r in qfl_runs])
        ),
        "utility_loss_fraction_mean": float(
            np.mean([r["utility_loss_fraction"] for r in qfl_runs])
        ),
        "breakthrough_B_50pct_attack_reduction": float(
            np.mean([r["attack_mse_reduction_vs_t1p"] for r in qfl_runs])
        )
        >= 0.50
        and float(np.mean([r["utility_loss_fraction"] for r in qfl_runs])) <= 0.10,
    }

    results = {
        "benchmark": "round01_privacy_breakthrough_survey",
        "lanes_implemented": ["GARD-SPARSE", "QFL-SHIELD", "LEAK-CERT"],
        "lanes_survey_only": ["ASSIGN-LOCK", "PROBE-DETECT"],
        "config_breakthrough": {
            "A_row_reduction_25pct": stage2_summary["breakthrough_A_25pct"],
            "B_defense_50pct_at_10pct_utility": qfl_agg["breakthrough_B_50pct_attack_reduction"],
            "C_cert_2x_tighter_than_naive": cert_agg["tight_over_naive_mean"] >= 2.0,
        },
        "stage2": {
            "config": asdict(stage2_cfg),
            "seeds": stage2_seeds,
            "per_run": stage2_runs,
            "summary": stage2_summary,
        },
        "qfl_shield": {
            "config": asdict(qfl_cfg),
            "per_seed": qfl_runs,
            "aggregate": qfl_agg,
        },
        "leak_cert": {
            "per_seed": cert_runs,
            "aggregate": cert_agg,
        },
        "primary_lane_recommendation": None,  # filled below
        "runtime_sec": time.perf_counter() - t0,
        "honesty_notes": [
            "GARD-SPARSE uses oracle assignment (parent Round-05 favorable setting).",
            "QFL-SHIELD rank/noise are not yet validation-tuned; single configuration sweep.",
            "LEAK-CERT dof bound is a heuristic, not a formal DP guarantee.",
            "ASSIGN-LOCK deferred to Round 02 pending secure-aggregation stub.",
        ],
    }

    # Recommend primary lane from breakthrough proximity
    scores = {
        "GARD-SPARSE": 1.0 if stage2_summary["breakthrough_A_25pct"] else 0.35,
        "QFL-SHIELD": float(np.clip(qfl_agg["attack_reduction_mean"], 0, 1)),
        "LEAK-CERT": min(1.0, cert_agg["tight_over_naive_mean"] / 2.0) * 0.5,
    }
    primary = max(scores, key=scores.get)
    results["primary_lane_recommendation"] = {
        "lane": primary,
        "scores": scores,
        "rationale": (
            "Highest near-term measurable signal vs config acceptance criteria."
        ),
    }

    out = ARTIFACTS / "round01_metrics.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out)
    logger.info("Primary lane: %s", primary)
    logger.info("Runtime %.2fs", results["runtime_sec"])


if __name__ == "__main__":
    main()
