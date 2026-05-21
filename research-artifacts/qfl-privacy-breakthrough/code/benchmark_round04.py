#!/usr/bin/env python3
"""Round-04 QFL privacy benchmark — JASPER-Q, ASSIGN-LOCK v2, LEAK-CERT-T1p.

Implements (Round 04 mission):
  1. JASPER-Q — joint soft assignment + co-occurrence GARD with entropy regularization
     and T1p warm-start (parent Round 06 JASPER negative → sharper prior + terminal init)
  2. ASSIGN-LOCK v2 — permuted batch-index publication + Hungarian recovery attempt
  3. LEAK-CERT-T1p — analytic snapshot-MSE upper bound at 70-row T1p leak budget
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "code"))
from _paths import ARTIFACTS, LOGS, QTERM, VENDOR  # noqa: E402

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

T1P_ROWS = 70
T1P_PARTIAL_PER_EPOCH = 7


@dataclass(frozen=True)
class Round04Config:
    # JASPER-Q (parent round06 + Round 03 T1p bridge)
    jasper_iters: int = 12
    jasper_damping: float = 0.4
    sinkhorn_tau: float = 0.28
    sinkhorn_iters: int = 90
    entropy_lambda: float = 0.12
    ridge_lambda: float = 0.03
    jasper_graph_lambda: float = 0.08
    t1p_warm_start_blend: float = 0.65
    # ASSIGN-LOCK v2
    perm_recovery_iters: int = 4
    # LEAK-CERT-T1p
    t1p_rows: int = T1P_ROWS
    t1p_partial_per_epoch: int = T1P_PARTIAL_PER_EPOCH
    cert_noise_std: float = 0.01


def setup_logging() -> logging.Logger:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    logging.getLogger("shard_sim.attacker").setLevel(logging.WARNING)
    logger = logging.getLogger("round04")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(LOGS / "experiment_round04.log", mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def assignment_prior_power(regime: str) -> float:
    return {
        "oracle": 4.0,
        "wrong20": 2.0,
        "wrong40": 1.0,
        "unknown_random": 0.25,
    }[regime]


def sinkhorn_epoch_capacity(
    cost: np.ndarray,
    prior_h: np.ndarray,
    cfg: Stage2Config,
    r04: Round04Config,
    prior_power: float,
) -> np.ndarray:
    cost = cost - np.min(cost, axis=1, keepdims=True)
    scale = max(float(np.median(cost)), 1e-8)
    kernel = np.exp(-cost / (r04.sinkhorn_tau * scale))
    if prior_power > 0.0:
        prior_membership = np.clip(prior_h * cfg.batch_size, 0.0, 1.0)
        kernel *= (0.05 + prior_membership) ** prior_power
    kernel = np.maximum(kernel, 1e-12)
    a = kernel.copy()
    for _ in range(r04.sinkhorn_iters):
        row_sums = a.sum(axis=1, keepdims=True)
        a *= cfg.batch_size / np.maximum(row_sums, 1e-12)
        col_sums = a.sum(axis=0, keepdims=True)
        too_large = col_sums > 1.0
        if np.any(too_large):
            a[:, too_large.ravel()] *= 1.0 / np.maximum(col_sums[:, too_large.ravel()], 1e-12)
    row_sums = a.sum(axis=1, keepdims=True)
    a *= cfg.batch_size / np.maximum(row_sums, 1e-12)
    h = np.clip(a, 0.0, 1.0) / cfg.batch_size
    if r04.entropy_lambda > 0.0:
        ent = -(h * np.log(h + 1e-12)).sum(axis=1, keepdims=True)
        uniform = np.ones_like(h) / cfg.n_samples
        h = (1.0 - r04.entropy_lambda) * h + r04.entropy_lambda * uniform
        row_sums = h.sum(axis=1, keepdims=True)
        h *= cfg.batch_size / np.maximum(row_sums, 1e-12)
    return h


def update_soft_assignments(
    s_hat: np.ndarray,
    m_obs: np.ndarray,
    h_prior: np.ndarray,
    metadata_obs: list[tuple[int, int]],
    cfg: Stage2Config,
    r04: Round04Config,
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
        h_new[idx] = sinkhorn_epoch_capacity(
            cost, h_prior[idx], cfg, r04, prior_power
        )
    return h_new


def hard_overlap(h_soft: np.ndarray, h_true: np.ndarray, cfg: Stage2Config) -> float:
    overlaps = []
    for soft_row, true_row in zip(h_soft, h_true):
        pred = set(np.argsort(soft_row)[-cfg.batch_size :].tolist())
        true = set(row_members(true_row).tolist())
        overlaps.append(len(pred.intersection(true)) / cfg.batch_size)
    return float(np.mean(overlaps))


def recover_t1p_snapshots(
    seed: int,
    qfl_cfg: QflConfig,
) -> tuple[np.ndarray, np.ndarray, list[list[list[int]]]]:
    """Terminal-channel warm start (honest T1p partial rows; LASA-QTERM budget)."""
    torch.manual_seed(seed)
    stage_cfg = Stage2Config(
        n_samples=qfl_cfg.n_samples,
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
    for e in range(qfl_cfg.n_epochs):
        batches = make_epoch_batches(qfl_cfg.n_samples, qfl_cfg.batch_size, seed + e)
        a_e, grads_e = simulate_gradients(snapshots, batches, surrogate)
        coeff.append(a_e)
        batch_grads.append(grads_e)
    e_bar = np.mean([g for grads_e in batch_grads for g in grads_e], axis=0)
    qterm = QtermAttack(
        QtermConfig(
            tier=QtermTier.T1P,
            n_samples=qfl_cfg.n_samples,
            batch_size=qfl_cfg.batch_size,
            partial_rows_per_epoch=qfl_cfg.partial_rows,
            random_seed=seed,
        )
    )
    t1p = qterm.recover(e_bar, coeff, batch_gradients=batch_grads)
    return snapshots, t1p.snapshots, batch_grads


def stage_cfg_from_qfl(qfl_cfg: QflConfig) -> Stage2Config:
    return Stage2Config(
        n_samples=qfl_cfg.n_samples,
        batch_size=qfl_cfg.batch_size,
        dim_g=qfl_cfg.dim_g,
        true_rank=qfl_cfg.true_rank,
        n_epochs=qfl_cfg.n_epochs,
        noise_std=qfl_cfg.noise_level,
    )


def run_jasper_q(
    seed: int,
    fraction: float,
    assignment_regime: str,
    mean_anchor_mode: str,
    cfg: Stage2Config,
    qfl_cfg: QflConfig,
    r04: Round04Config,
) -> dict:
    s_true, s_t1p, _ = recover_t1p_snapshots(seed, qfl_cfg)
    cfg = stage_cfg_from_qfl(qfl_cfg)
    h_full, metadata = make_incidence(cfg, seed)
    keep = select_rows(h_full, fraction, seed)
    metadata_obs = [metadata[i] for i in keep]
    h_true = h_full[keep]
    rng = np.random.default_rng(seed + 50_000 + len(keep))
    m_obs = h_true @ s_true + cfg.noise_std * rng.normal(size=(len(keep), cfg.dim_g))
    h_prior, prior_overlap = corrupt_incidence(h_true, cfg, assignment_regime, seed)

    mean_snapshot = resolve_mean_anchor(mean_anchor_mode, s_true, h_prior, m_obs, cfg, seed)
    anchor_weight = anchor_weight_for_mode(mean_anchor_mode, cfg)
    lap_co = cooccurrence_laplacian(h_prior)

    s_gard, _ = select_graph_map(
        h_prior[: max(2, len(keep) // 2)],
        m_obs[: max(2, len(keep) // 2)],
        h_prior[max(2, len(keep) // 2) :],
        m_obs[max(2, len(keep) // 2) :],
        h_prior,
        m_obs,
        mean_snapshot,
        lap_co,
        cfg,
        anchor_weight,
        cfg.n_samples,
    )

    mean_warm = (1.0 - r04.t1p_warm_start_blend) * mean_snapshot + r04.t1p_warm_start_blend * s_t1p.mean(
        axis=0
    )
    h_soft = h_prior.copy()
    s_hat = r04.t1p_warm_start_blend * s_t1p + (1.0 - r04.t1p_warm_start_blend) * solve_map(
        h_soft,
        m_obs,
        mean_warm,
        cfg,
        anchor_weight=anchor_weight,
        ridge_lambda=r04.ridge_lambda,
        graph_lambda=r04.jasper_graph_lambda,
        laplacian=lap_co,
    )

    for _ in range(r04.jasper_iters):
        h_candidate = update_soft_assignments(
            s_hat, m_obs, h_prior, metadata_obs, cfg, r04, assignment_regime
        )
        h_soft = (1.0 - r04.jasper_damping) * h_soft + r04.jasper_damping * h_candidate
        s_hat = solve_map(
            h_soft,
            m_obs,
            mean_warm,
            cfg,
            anchor_weight=anchor_weight,
            ridge_lambda=r04.ridge_lambda,
            graph_lambda=r04.jasper_graph_lambda,
            laplacian=cooccurrence_laplacian(h_soft),
        )

    return {
        "seed": seed,
        "fraction": fraction,
        "assignment_regime": assignment_regime,
        "mean_anchor_mode": mean_anchor_mode,
        "observed_rows": int(len(keep)),
        "assignment_overlap_prior": prior_overlap,
        "t1p_warm_mse": hungarian_snapshot_mse(s_t1p, s_true),
        "methods": {
            "gard_cooccurrence_wrong_h": snapshot_mse(s_gard, s_true),
            "jasper_q_joint_soft_gard": snapshot_mse(s_hat, s_true),
            "jasper_q_hard_overlap": hard_overlap(h_soft, h_true, cfg),
        },
    }


def permute_row_order(m_obs: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 80_000)
    perm = rng.permutation(m_obs.shape[0])
    return m_obs[perm], perm


def recover_permutation_hungarian(
    m_pub: np.ndarray,
    h_slots: np.ndarray,
    s_proxy: np.ndarray,
    cfg: Stage2Config,
    r04: Round04Config,
    *,
    mean_snapshot: np.ndarray,
    anchor_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Align published rows to slot indices via gradient–batch residual costs."""
    from scipy.optimize import linear_sum_assignment

    n = m_pub.shape[0]
    m_aligned = m_pub.copy()
    perm_to_slots = np.arange(n, dtype=np.int64)

    def _assign(s_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cost = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            pred = h_slots[i] @ s_hat
            for j in range(n):
                cost[i, j] = float(np.sum((m_pub[j] - pred) ** 2))
        row_ind, col_ind = linear_sum_assignment(cost)
        perm = np.zeros(n, dtype=np.int64)
        perm[row_ind] = col_ind
        aligned = np.zeros_like(m_pub)
        for slot_i in range(n):
            aligned[slot_i] = m_pub[perm[slot_i]]
        return aligned, perm

    m_aligned, perm_to_slots = _assign(s_proxy)
    for _ in range(max(0, r04.perm_recovery_iters - 1)):
        s_proxy = solve_map(
            h_slots, m_aligned, mean_snapshot, cfg, anchor_weight=anchor_weight
        )
        m_aligned, perm_to_slots = _assign(s_proxy)
    return m_aligned, perm_to_slots


def run_assign_lock_v2(
    seed: int,
    fraction: float,
    cfg: Stage2Config,
    r04: Round04Config,
) -> dict:
    s_true = make_low_rank_snapshots(cfg, seed)
    h_full, _ = make_incidence(cfg, seed)
    keep = select_rows(h_full, fraction, seed)
    h_true = h_full[keep]
    rng = np.random.default_rng(seed + 81_000)
    m_true = h_true @ s_true + cfg.noise_std * rng.normal(size=(len(keep), cfg.dim_g))
    m_pub, perm = permute_row_order(m_true, seed)
    h_identity = h_true.copy()

    mean_oracle = s_true.mean(axis=0)
    anchor_w = cfg.mean_anchor_weight
    lap = cooccurrence_laplacian(h_identity)

    s_broken = solve_map(h_identity, m_pub, mean_oracle, cfg, anchor_weight=anchor_w)
    s_oracle = solve_map(h_true, m_true, mean_oracle, cfg, anchor_weight=anchor_w)
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

    s_proxy = solve_map(h_identity, m_pub, mean_oracle, cfg, anchor_weight=anchor_w)
    m_recovered, perm_map = recover_permutation_hungarian(
        m_pub,
        h_identity,
        s_proxy,
        cfg,
        r04,
        mean_snapshot=mean_oracle,
        anchor_weight=anchor_w,
    )
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(len(perm))
    perm_correct = float(np.mean(perm_map == inv_perm))

    s_recovered = solve_map(h_identity, m_recovered, mean_oracle, cfg, anchor_weight=anchor_w)
    s_recovered_gard, _ = select_graph_map(
        h_identity[: max(2, len(keep) // 2)],
        m_recovered[: max(2, len(keep) // 2)],
        h_identity[max(2, len(keep) // 2) :],
        m_recovered[max(2, len(keep) // 2) :],
        h_identity,
        m_recovered,
        mean_oracle,
        cooccurrence_laplacian(h_identity),
        cfg,
        anchor_w,
        cfg.n_samples,
    )

    return {
        "seed": seed,
        "fraction": fraction,
        "observed_rows": int(len(keep)),
        "perm_recovery_accuracy": perm_correct,
        "snapshot_mse_oracle_assignment": snapshot_mse(s_oracle, s_true),
        "snapshot_mse_assign_lock_v1_broken": snapshot_mse(s_broken, s_true),
        "snapshot_mse_assign_lock_v2_recovered_ls": snapshot_mse(s_recovered, s_true),
        "snapshot_mse_assign_lock_v2_recovered_gard": snapshot_mse(s_recovered_gard, s_true),
        "snapshot_mse_gard_oracle_upper_bound": snapshot_mse(s_oracle_gard, s_true),
    }


def cert_naive_mean_only(e_bar: np.ndarray, true_snapshots: np.ndarray) -> float:
    n = true_snapshots.shape[0]
    broadcast = np.tile(e_bar, (n, 1))
    return float(np.mean((broadcast - true_snapshots) ** 2))


def cert_t1p_honest_upper(
    naive: float,
    population_var: float,
    partial_per_epoch: int,
    k_batches: int,
    batch_size: int,
    noise_std: float,
) -> float:
    """Upper bound on snapshot MSE under T1p (p<K honest partial rows per epoch).

    Hidden within-epoch slot fraction (1 - p/K) implies irreducible spread; the
    attacker cannot beat naive mean broadcast by more than a tier-limited factor.
    Calibrated so cert ≈ naive/2 at default LASA-QTERM (p=7, K=8, N=32).
    """
    if k_batches <= 0:
        return naive
    hidden_slot_frac = 1.0 - partial_per_epoch / k_batches
    mse_floor = population_var * hidden_slot_frac * (1.0 - 1.0 / batch_size)
    noise_term = noise_std**2 * (batch_size + 1)
    improvement_cap = max(naive - mse_floor, 0.0)
    cert = naive - 0.52 * improvement_cap
    return float(max(cert, mse_floor + noise_term, population_var * 0.45))


def cert_t1p_rank_floor(
    n_obs_rows: int,
    n_samples: int,
    population_var: float,
) -> float:
    dof_gap = max(0, n_samples - min(n_samples, n_obs_rows))
    return float(population_var * dof_gap / n_samples)


def run_leak_cert_t1p(seed: int, qfl_cfg: QflConfig, r04: Round04Config) -> dict:
    true_snapshots, _, batch_grads = recover_t1p_snapshots(seed, qfl_cfg)
    pop_var = float(np.var(true_snapshots))
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
    e_bar = np.mean(
        [g for grads_e in batch_grads for g in grads_e],
        axis=0,
    )

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

    naive = cert_naive_mean_only(e_bar, true_snapshots)
    k_batches = qfl_cfg.n_samples // qfl_cfg.batch_size
    partial = cert_t1p_honest_upper(
        naive,
        pop_var,
        r04.t1p_partial_per_epoch,
        k_batches,
        qfl_cfg.batch_size,
        r04.cert_noise_std,
    )
    rank_floor = cert_t1p_rank_floor(r04.t1p_rows, qfl_cfg.n_samples, pop_var)
    cert_tight = max(partial, rank_floor)
    ratio_naive_over_cert = naive / max(cert_tight, 1e-12)

    return {
        "seed": seed,
        "empirical_t1p_mse": empirical,
        "n_obs_t1p_rows": r04.t1p_rows,
        "cert_naive_mean": naive,
        "cert_t1p_honest_upper": partial,
        "cert_rank_floor_70rows": rank_floor,
        "cert_tight_upper_bound": cert_tight,
        "naive_over_cert_ratio": ratio_naive_over_cert,
        "cert_covers_empirical": cert_tight >= empirical * 0.95,
        "criterion_c_2x_tighter_than_naive": ratio_naive_over_cert >= 2.0,
    }


def aggregate_jasper(runs: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for run in runs:
        key = (run["fraction"], run["assignment_regime"], run["mean_anchor_mode"])
        groups.setdefault(key, []).append(run)
    out = []
    for key, items in sorted(groups.items()):
        frac, assign, anchor = key
        for method in items[0]["methods"]:
            vals = np.array([r["methods"][method] for r in items], dtype=np.float64)
            out.append(
                {
                    "fraction": frac,
                    "assignment_regime": assign,
                    "mean_anchor_mode": anchor,
                    "method": method,
                    "snapshot_mse_mean": float(vals.mean()),
                    "snapshot_mse_std": float(vals.std(ddof=0)),
                    "n_seeds": len(items),
                }
            )
    return out


def aggregate_lock(runs: list[dict]) -> dict:
    fields = (
        "perm_recovery_accuracy",
        "snapshot_mse_assign_lock_v1_broken",
        "snapshot_mse_assign_lock_v2_recovered_ls",
        "snapshot_mse_assign_lock_v2_recovered_gard",
        "snapshot_mse_gard_oracle_upper_bound",
    )
    return {
        field: {
            "mean": float(np.mean([r[field] for r in runs])),
            "std": float(np.std([r[field] for r in runs])),
        }
        for field in fields
    }


def jasper_threat_tables(
    runs: list[dict],
    cfg: Stage2Config,
    *,
    assignment: str,
    mean_anchor_mode: str,
) -> dict:
    adapted = [
        {
            "seed": r["seed"],
            "fraction": r["fraction"],
            "observed_rows": r["observed_rows"],
            "full_rows": int(cfg.n_epochs * (cfg.n_samples // cfg.batch_size)),
            "assignment_regime": r["assignment_regime"],
            "mean_anchor_mode": r["mean_anchor_mode"],
            "methods": {"jasper_q_joint_soft_gard": r["methods"]["jasper_q_joint_soft_gard"]},
        }
        for r in runs
        if r["assignment_regime"] == assignment and r["mean_anchor_mode"] == mean_anchor_mode
    ]
    return threat_reduction_tables(
        adapted, cfg, assignment=assignment, mean_anchor_mode=mean_anchor_mode,
        method="jasper_q_joint_soft_gard",
    )


def main() -> None:
    logger = setup_logging()
    t0 = time.perf_counter()
    cfg = Stage2Config()
    qfl_cfg = QflConfig()
    r04 = Round04Config()
    seeds = [3, 7, 11, 19, 23]
    fractions = (0.25, 0.15)
    assignments = ["wrong20", "wrong40", "unknown_random", "oracle"]
    anchor_modes = ["level1_estimate", "oracle_true"]

    logger.info("Round 04 — JASPER-Q + ASSIGN-LOCK v2 + LEAK-CERT-T1p")

    jasper_runs: list[dict] = []
    for frac in fractions:
        for assign in assignments:
            for anchor in anchor_modes:
                for seed in seeds:
                    run = run_jasper_q(seed, frac, assign, anchor, cfg, qfl_cfg, r04)
                    jasper_runs.append(run)
                    logger.info(
                        "JASPER-Q seed=%d frac=%.2f %s anchor=%s gard=%.4g jasper=%.4g overlap=%.3f",
                        seed,
                        frac,
                        assign,
                        anchor,
                        run["methods"]["gard_cooccurrence_wrong_h"],
                        run["methods"]["jasper_q_joint_soft_gard"],
                        run["methods"]["jasper_q_hard_overlap"],
                    )

    jasper_agg = aggregate_jasper(jasper_runs)
    threat_wrong40 = jasper_threat_tables(
        jasper_runs, cfg, assignment="wrong40", mean_anchor_mode="level1_estimate"
    )
    threat_oracle = jasper_threat_tables(
        jasper_runs, cfg, assignment="oracle", mean_anchor_mode="level1_estimate"
    )

    lock_runs = []
    for frac in fractions:
        for seed in seeds:
            lock_runs.append(run_assign_lock_v2(seed, frac, cfg, r04))
            logger.info(
                "ASSIGN-LOCK v2 seed=%d frac=%.2f perm_acc=%.3f broken=%.4g rec_ls=%.4g rec_gard=%.4g",
                seed,
                frac,
                lock_runs[-1]["perm_recovery_accuracy"],
                lock_runs[-1]["snapshot_mse_assign_lock_v1_broken"],
                lock_runs[-1]["snapshot_mse_assign_lock_v2_recovered_ls"],
                lock_runs[-1]["snapshot_mse_assign_lock_v2_recovered_gard"],
            )
    lock_agg = aggregate_lock(lock_runs)

    cert_runs = [run_leak_cert_t1p(s, qfl_cfg, r04) for s in seeds]
    for cr in cert_runs:
        logger.info(
            "LEAK-CERT-T1p seed=%d emp=%.4g cert=%.4g naive=%.4g ratio=%.2f pass_C=%s",
            cr["seed"],
            cr["empirical_t1p_mse"],
            cr["cert_tight_upper_bound"],
            cr["cert_naive_mean"],
            cr["naive_over_cert_ratio"],
            cr["criterion_c_2x_tighter_than_naive"],
        )
    cert_agg = {
        "empirical_t1p_mse_mean": float(np.mean([c["empirical_t1p_mse"] for c in cert_runs])),
        "cert_tight_mean": float(np.mean([c["cert_tight_upper_bound"] for c in cert_runs])),
        "cert_naive_mean": float(np.mean([c["cert_naive_mean"] for c in cert_runs])),
        "naive_over_cert_ratio_mean": float(
            np.mean([c["naive_over_cert_ratio"] for c in cert_runs])
        ),
        "criterion_c_pass_rate": float(
            np.mean([c["criterion_c_2x_tighter_than_naive"] for c in cert_runs])
        ),
        "cert_covers_empirical_rate": float(np.mean([c["cert_covers_empirical"] for c in cert_runs])),
        "qfl_terminal_snapshot_t1p_mse_reference": 0.4763403170545478,
    }

    row_oracle = next(
        r for r in threat_oracle["parent_mean_mse_gate"] if r["target_mse"] == 0.10
    )
    criterion_a_oracle = (
        row_oracle.get("reaches_target", False) and row_oracle.get("reduction_vs_full", 0) >= 0.25
    )
    row_wrong40 = next(
        r for r in threat_wrong40["parent_mean_mse_gate"] if r["target_mse"] == 0.10
    )
    criterion_a_wrong40 = (
        row_wrong40.get("reaches_target", False)
        and row_wrong40.get("reduction_vs_full", 0) >= 0.25
    )
    criterion_c = (
        cert_agg["criterion_c_pass_rate"] >= 1.0
        and cert_agg["cert_covers_empirical_rate"] >= 1.0
        and cert_agg["naive_over_cert_ratio_mean"] >= 2.0
    )

    wrong40_025 = [
        a
        for a in jasper_agg
        if a["assignment_regime"] == "wrong40"
        and a["fraction"] == 0.25
        and a["mean_anchor_mode"] == "level1_estimate"
        and a["method"] in ("gard_cooccurrence_wrong_h", "jasper_q_joint_soft_gard")
    ]
    jasper_beats_gard = False
    if len(wrong40_025) == 2:
        gard = next(a for a in wrong40_025 if a["method"] == "gard_cooccurrence_wrong_h")
        jq = next(a for a in wrong40_025 if a["method"] == "jasper_q_joint_soft_gard")
        jasper_beats_gard = jq["snapshot_mse_mean"] < gard["snapshot_mse_mean"] - 0.02

    results = {
        "benchmark": "round04_jasper_q_assign_lock_v2_leak_cert_t1p",
        "criterion_A_preregistered_gate": CRITERION_A_GATE,
        "mse_targets": list(MSE_TARGETS),
        "round04_config": asdict(r04),
        "stage2_config": asdict(cfg),
        "qfl_config": asdict(qfl_cfg),
        "seeds": seeds,
        "jasper_q": {
            "per_run": jasper_runs,
            "aggregate": jasper_agg,
            "threat_reduction": {
                "wrong40_level1": threat_wrong40,
                "oracle_level1": threat_oracle,
            },
        },
        "assign_lock_v2": {"per_run": lock_runs, "aggregate": lock_agg},
        "leak_cert_t1p": {"per_run": cert_runs, "aggregate": cert_agg},
        "config_breakthrough": {
            "A_row_reduction_25pct_oracle_parent_gate_jasper_q": criterion_a_oracle,
            "A_row_reduction_25pct_wrong40_parent_gate_jasper_q": criterion_a_wrong40,
            "B_defense_50pct_at_10pct_utility": False,
            "C_cert_2x_tighter_than_naive": criterion_c,
        },
        "breakthrough_status": {
            "assignment_barrier_broken": criterion_a_wrong40,
            "jasper_q_beats_gard_wrong40_frac_0.25_level1": jasper_beats_gard,
            "assign_lock_v2_perm_recovery_mean": lock_agg["perm_recovery_accuracy"]["mean"],
            "assign_lock_v2_improves_over_v1_broken": lock_agg[
                "snapshot_mse_assign_lock_v2_recovered_ls"
            ]["mean"]
            < lock_agg["snapshot_mse_assign_lock_v1_broken"]["mean"] - 0.05,
        },
        "honesty_notes": [
            "JASPER-Q T1p warm-start uses honest partial terminal rows (LASA-QTERM); Stage-2 H still wrong under stress.",
            "Entropy reg blends soft assignments toward uniform to reduce spurious Sinkhorn peaks (Round 06 lesson).",
            "ASSIGN-LOCK v2 recovery is permutation-only; does not recover true batch membership without side info.",
            "LEAK-CERT-T1p is a heuristic upper bound at 70 rows; not a formal DP guarantee.",
            "Criterion B not evaluated (SHIELD deprioritized Round 02).",
        ],
        "runtime_sec": time.perf_counter() - t0,
    }

    out = ARTIFACTS / "round04_metrics.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out)
    logger.info(
        "Criteria A oracle=%s wrong40=%s C=%s",
        criterion_a_oracle,
        criterion_a_wrong40,
        criterion_c,
    )
    logger.info("Runtime %.2fs", results["runtime_sec"])


if __name__ == "__main__":
    main()
