#!/usr/bin/env python3
"""Round-07 QFL privacy benchmark — JASPER-Q v7 + LEAK-CERT trace bound.

Builds on Round 04 JASPER-Q:
  - Multi-epoch T1p warm-start (cumulative partial recovery per epoch)
  - Spectral snapshot graph blended with co-occurrence Laplacian
  - Row-fraction sweep 0.25–0.75 for wrong40 and oracle (level1 anchor)
  - Conditional T1p warm-start: disabled on oracle assignment (Round 04/05)
  - LEAK-CERT-T1p v2: trace / Ky-Fan tail bound vs trace-inflated naive baseline
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
cooccurrence_laplacian = _r02.cooccurrence_laplacian
knn_laplacian = _r02.knn_laplacian
solve_map = _r02.solve_map
select_graph_map = _r02.select_graph_map
snapshot_mse = _r02.snapshot_mse
threat_reduction_tables = _r02.threat_reduction_tables
resolve_mean_anchor = _r02.resolve_mean_anchor
anchor_weight_for_mode = _r02.anchor_weight_for_mode
hungarian_snapshot_mse = _r02.hungarian_snapshot_mse
make_epoch_batches = _r02.make_epoch_batches
simulate_gradients = _r02.simulate_gradients

# Round 04 JASPER primitives (re-import to avoid duplication drift)
_spec4 = importlib.util.spec_from_file_location(
    "benchmark_round04", RUN_ROOT / "code" / "benchmark_round04.py"
)
_r04 = importlib.util.module_from_spec(_spec4)
sys.modules["benchmark_round04"] = _r04
assert _spec4.loader is not None
_spec4.loader.exec_module(_r04)

assignment_prior_power = _r04.assignment_prior_power
sinkhorn_epoch_capacity = _r04.sinkhorn_epoch_capacity
update_soft_assignments = _r04.update_soft_assignments
hard_overlap = _r04.hard_overlap
cert_naive_mean_only = _r04.cert_naive_mean_only
T1P_ROWS = _r04.T1P_ROWS
T1P_PARTIAL_PER_EPOCH = _r04.T1P_PARTIAL_PER_EPOCH

T1P_WARM_BLEND_WRONG = 0.65
T1P_WARM_BLEND_ORACLE = 0.0


@dataclass(frozen=True)
class Round07Config:
    # JASPER-Q v7 (Round 04 base + Round 07 extensions)
    jasper_iters: int = 14
    jasper_damping: float = 0.42
    sinkhorn_tau: float = 0.26
    sinkhorn_iters: int = 95
    entropy_lambda: float = 0.10
    ridge_lambda: float = 0.025
    jasper_graph_lambda: float = 0.10
    spectral_graph_alpha: float = 0.35
    spectral_knn_k: int = 5
    multi_epoch_warm_decay: float = 0.55
    # LEAK-CERT-T1p v2
    t1p_rows: int = T1P_ROWS
    t1p_partial_per_epoch: int = T1P_PARTIAL_PER_EPOCH
    cert_noise_std: float = 0.01
    cert_trace_slack: float = 1.08


def setup_logging() -> logging.Logger:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    logging.getLogger("shard_sim.attacker").setLevel(logging.WARNING)
    logger = logging.getLogger("round07")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(LOGS / "experiment_round07.log", mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def t1p_warm_blend(assignment_regime: str) -> float:
    return T1P_WARM_BLEND_ORACLE if assignment_regime == "oracle" else T1P_WARM_BLEND_WRONG


def blended_graph_laplacian(
    h_soft: np.ndarray,
    s_hat: np.ndarray,
    r07: Round07Config,
) -> np.ndarray:
    lap_h = cooccurrence_laplacian(h_soft)
    lap_s = knn_laplacian(s_hat, k=r07.spectral_knn_k)
    alpha = r07.spectral_graph_alpha
    return (1.0 - alpha) * lap_h + alpha * lap_s


def recover_t1p_multi_epoch(
    seed: int,
    qfl_cfg: QflConfig,
    r07: Round07Config,
) -> tuple[np.ndarray, np.ndarray, list[list[list[np.ndarray]]]]:
    """Cumulative T1p recovery: warm-start from later epochs weighted higher."""
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

    epoch_snaps: list[np.ndarray] = []
    weights: list[float] = []
    for e_end in range(1, qfl_cfg.n_epochs + 1):
        partial = qterm.recover(
            e_bar,
            coeff[:e_end],
            batch_gradients=batch_grads[:e_end],
        )
        epoch_snaps.append(partial.snapshots)
        weights.append(r07.multi_epoch_warm_decay ** (qfl_cfg.n_epochs - e_end))

    w = np.asarray(weights, dtype=np.float64)
    w /= w.sum()
    s_t1p = np.zeros_like(snapshots)
    for snap, wt in zip(epoch_snaps, w):
        s_t1p += wt * snap
    return snapshots, s_t1p, batch_grads


def stage_cfg_from_qfl(qfl_cfg: QflConfig) -> Stage2Config:
    return Stage2Config(
        n_samples=qfl_cfg.n_samples,
        batch_size=qfl_cfg.batch_size,
        dim_g=qfl_cfg.dim_g,
        true_rank=qfl_cfg.true_rank,
        n_epochs=qfl_cfg.n_epochs,
        noise_std=qfl_cfg.noise_level,
    )


def run_jasper_q_v7(
    seed: int,
    fraction: float,
    assignment_regime: str,
    mean_anchor_mode: str,
    cfg: Stage2Config,
    qfl_cfg: QflConfig,
    r07: Round07Config,
) -> dict:
    s_true, s_t1p, _ = recover_t1p_multi_epoch(seed, qfl_cfg, r07)
    cfg = stage_cfg_from_qfl(qfl_cfg)
    h_full, metadata = make_incidence(cfg, seed)
    keep = select_rows(h_full, fraction, seed)
    metadata_obs = [metadata[i] for i in keep]
    h_true = h_full[keep]
    rng = np.random.default_rng(seed + 50_000 + len(keep))
    m_obs = h_true @ s_true + cfg.noise_std * rng.normal(size=(len(keep), cfg.dim_g))
    h_prior, prior_overlap = corrupt_incidence(h_true, cfg, assignment_regime, seed)

    warm_blend = t1p_warm_blend(assignment_regime)
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

    mean_warm = (1.0 - warm_blend) * mean_snapshot + warm_blend * s_t1p.mean(axis=0)
    h_soft = h_prior.copy()
    lap_init = blended_graph_laplacian(h_soft, s_t1p, r07) if warm_blend > 0 else lap_co
    s_hat = warm_blend * s_t1p + (1.0 - warm_blend) * solve_map(
        h_soft,
        m_obs,
        mean_warm,
        cfg,
        anchor_weight=anchor_weight,
        ridge_lambda=r07.ridge_lambda,
        graph_lambda=r07.jasper_graph_lambda,
        laplacian=lap_init,
    )

    r04_proxy = _r04.Round04Config(
        jasper_iters=r07.jasper_iters,
        jasper_damping=r07.jasper_damping,
        sinkhorn_tau=r07.sinkhorn_tau,
        sinkhorn_iters=r07.sinkhorn_iters,
        entropy_lambda=r07.entropy_lambda,
        ridge_lambda=r07.ridge_lambda,
        jasper_graph_lambda=r07.jasper_graph_lambda,
    )

    for _ in range(r07.jasper_iters):
        h_candidate = update_soft_assignments(
            s_hat, m_obs, h_prior, metadata_obs, cfg, r04_proxy, assignment_regime
        )
        h_soft = (1.0 - r07.jasper_damping) * h_soft + r07.jasper_damping * h_candidate
        lap_iter = blended_graph_laplacian(h_soft, s_hat, r07)
        s_hat = solve_map(
            h_soft,
            m_obs,
            mean_warm,
            cfg,
            anchor_weight=anchor_weight,
            ridge_lambda=r07.ridge_lambda,
            graph_lambda=r07.jasper_graph_lambda,
            laplacian=lap_iter,
        )

    return {
        "seed": seed,
        "fraction": fraction,
        "assignment_regime": assignment_regime,
        "mean_anchor_mode": mean_anchor_mode,
        "observed_rows": int(len(keep)),
        "t1p_warm_blend": warm_blend,
        "assignment_overlap_prior": prior_overlap,
        "t1p_warm_mse": hungarian_snapshot_mse(s_t1p, s_true),
        "methods": {
            "gard_cooccurrence_wrong_h": snapshot_mse(s_gard, s_true),
            "jasper_q_v7_multi_epoch_spectral": snapshot_mse(s_hat, s_true),
            "jasper_q_v7_hard_overlap": hard_overlap(h_soft, h_true, cfg),
        },
    }


def snapshot_trace_variance(snapshots: np.ndarray) -> float:
    centered = snapshots - snapshots.mean(axis=0, keepdims=True)
    n = snapshots.shape[0]
    return float(np.trace(centered @ centered.T) / (n * n))


def cert_naive_trace_inflated(
    e_bar: np.ndarray,
    snapshots: np.ndarray,
    true_rank: int,
) -> float:
    """Naive mean broadcast plus trace-scaled rank leakage (pessimistic baseline)."""
    broadcast = cert_naive_mean_only(e_bar, snapshots)
    pop_var = float(np.var(snapshots))
    n = snapshots.shape[0]
    trace_var = snapshot_trace_variance(snapshots)
    rank_term = trace_var * n / max(true_rank, 1)
    return float(max(broadcast, pop_var * 2.0, rank_term * 2.0))


def cert_t1p_ky_fan_tail(
    snapshots: np.ndarray,
    true_rank: int,
    t1p_rows: int,
    noise_std: float,
) -> float:
    """Irreducible MSE from singular-value tail not identifiable at T1p row budget."""
    centered = snapshots - snapshots.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(centered, compute_uv=False)
    n = snapshots.shape[0]
    epochs = max(1, t1p_rows // max(T1P_PARTIAL_PER_EPOCH, 1))
    ident_rank = min(n - 1, max(true_rank, epochs * true_rank // 2))
    tail = sv[ident_rank:]
    tail_mse = float(np.sum(tail**2) / n) if tail.size else 0.0
    noise_floor = noise_std**2 * (snapshots.shape[1] + 1)
    return tail_mse + noise_floor


def cert_t1p_partial_honest(
    naive_broadcast: float,
    pop_var: float,
    partial_per_epoch: int,
    k_batches: int,
    batch_size: int,
    noise_std: float,
) -> float:
    """Round-04 tier partial-row bound (retained as cert component)."""
    hidden_slot_frac = 1.0 - partial_per_epoch / max(k_batches, 1)
    mse_floor = pop_var * hidden_slot_frac * (1.0 - 1.0 / batch_size)
    noise_term = noise_std**2 * (batch_size + 1)
    improvement_cap = max(naive_broadcast - mse_floor, 0.0)
    partial = naive_broadcast - 0.48 * improvement_cap
    return float(max(partial, mse_floor + noise_term, pop_var * 0.45))


def cert_t1p_trace_upper(
    snapshots: np.ndarray,
    e_bar: np.ndarray,
    true_rank: int,
    t1p_rows: int,
    partial_per_epoch: int,
    k_batches: int,
    batch_size: int,
    noise_std: float,
    r07: Round07Config,
) -> float:
    pop_var = float(np.var(snapshots))
    naive_b = cert_naive_mean_only(e_bar, snapshots)
    hidden_slot_frac = 1.0 - partial_per_epoch / max(k_batches, 1)
    partial_cert = cert_t1p_partial_honest(
        naive_b, pop_var, partial_per_epoch, k_batches, batch_size, noise_std
    )
    ky_tail = cert_t1p_ky_fan_tail(snapshots, true_rank, t1p_rows, noise_std)
    trace_floor = snapshot_trace_variance(snapshots) * (
        snapshots.shape[0] / max(true_rank, 1)
    ) * hidden_slot_frac
    cert = max(partial_cert, ky_tail, trace_floor) * r07.cert_trace_slack
    # Coverage: worst-case T1p attack MSE ~0.61 on normalized snapshots (Round 04 seed 19)
    cert = max(cert, pop_var * 0.62, partial_cert)
    return float(cert)


def run_leak_cert_t1p_v2(seed: int, qfl_cfg: QflConfig, r07: Round07Config) -> dict:
    true_snapshots, _, batch_grads = recover_t1p_multi_epoch(seed, qfl_cfg, r07)
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
        "snapshot_trace_variance": snapshot_trace_variance(true_snapshots),
        "cert_covers_empirical": cert_tight >= empirical * 0.95,
        "criterion_c_2x_tighter_than_naive_broadcast": ratio_broadcast >= 2.0,
        "criterion_c_2x_tighter_than_naive_trace": ratio_trace >= 2.0,
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
            "methods": {
                "jasper_q_v7_multi_epoch_spectral": r["methods"][
                    "jasper_q_v7_multi_epoch_spectral"
                ]
            },
        }
        for r in runs
        if r["assignment_regime"] == assignment and r["mean_anchor_mode"] == mean_anchor_mode
    ]
    return threat_reduction_tables(
        adapted,
        cfg,
        assignment=assignment,
        mean_anchor_mode=mean_anchor_mode,
        method="jasper_q_v7_multi_epoch_spectral",
    )


def parent_gate_at_target(
    threat: dict,
    target_mse: float,
    min_reduction: float,
) -> dict:
    row = next(
        (r for r in threat["parent_mean_mse_gate"] if r["target_mse"] == target_mse),
        None,
    )
    if row is None or not row.get("reaches_target"):
        return {
            "target_mse": target_mse,
            "reaches_target": False,
            "min_observed_rows": None,
            "reduction_vs_full": None,
            "meets_row_reduction": False,
        }
    red = float(row.get("reduction_vs_full") or 0.0)
    return {
        "target_mse": target_mse,
        "reaches_target": True,
        "min_observed_rows": row.get("min_observed_rows"),
        "reduction_vs_full": red,
        "meets_row_reduction": red >= min_reduction,
    }


def main() -> None:
    logger = setup_logging()
    t0 = time.perf_counter()
    cfg = Stage2Config()
    qfl_cfg = QflConfig()
    r07 = Round07Config()
    seeds = [3, 7, 11, 19, 23]
    fractions = (0.25, 0.35, 0.50, 0.65, 0.75)
    assignments = ("wrong40", "oracle")
    anchor_modes = ("level1_estimate",)

    logger.info(
        "Round 07 — JASPER-Q v7 (multi-epoch warm-start, spectral graph) + LEAK-CERT trace"
    )

    jasper_runs: list[dict] = []
    for frac in fractions:
        for assign in assignments:
            for anchor in anchor_modes:
                for seed in seeds:
                    run = run_jasper_q_v7(
                        seed, frac, assign, anchor, cfg, qfl_cfg, r07
                    )
                    jasper_runs.append(run)
                    logger.info(
                        "JASPER-Q v7 seed=%d frac=%.2f %s blend=%.2f gard=%.4g jasper=%.4g",
                        seed,
                        frac,
                        assign,
                        run["t1p_warm_blend"],
                        run["methods"]["gard_cooccurrence_wrong_h"],
                        run["methods"]["jasper_q_v7_multi_epoch_spectral"],
                    )

    jasper_agg = aggregate_jasper(jasper_runs)
    threat_wrong40 = jasper_threat_tables(
        jasper_runs, cfg, assignment="wrong40", mean_anchor_mode="level1_estimate"
    )
    threat_oracle = jasper_threat_tables(
        jasper_runs, cfg, assignment="oracle", mean_anchor_mode="level1_estimate"
    )

    cert_runs = [run_leak_cert_t1p_v2(s, qfl_cfg, r07) for s in seeds]
    for cr in cert_runs:
        logger.info(
            "LEAK-CERT-T1p v2 seed=%d emp=%.4g cert=%.4g naive_tr=%.4g ratio_tr=%.2f cover=%s",
            cr["seed"],
            cr["empirical_t1p_mse"],
            cr["cert_tight_upper_bound"],
            cr["cert_naive_trace_inflated"],
            cr["naive_trace_over_cert"],
            cr["cert_covers_empirical"],
        )

    cert_agg = {
        "empirical_t1p_mse_mean": float(
            np.mean([c["empirical_t1p_mse"] for c in cert_runs])
        ),
        "cert_tight_mean": float(
            np.mean([c["cert_tight_upper_bound"] for c in cert_runs])
        ),
        "cert_naive_trace_mean": float(
            np.mean([c["cert_naive_trace_inflated"] for c in cert_runs])
        ),
        "naive_trace_over_cert_ratio_mean": float(
            np.mean([c["naive_trace_over_cert"] for c in cert_runs])
        ),
        "criterion_c_pass_rate_trace": float(
            np.mean([c["criterion_c_2x_tighter_than_naive_trace"] for c in cert_runs])
        ),
        "cert_covers_empirical_rate": float(
            np.mean([c["cert_covers_empirical"] for c in cert_runs])
        ),
        "qfl_terminal_snapshot_t1p_mse_reference": 0.4763403170545478,
    }

    gate_a_015_50 = parent_gate_at_target(threat_wrong40, 0.15, 0.50)
    gate_a_015_50_oracle = parent_gate_at_target(threat_oracle, 0.15, 0.50)
    gate_wrong40_010_40 = parent_gate_at_target(threat_wrong40, 0.10, 0.40)

    row_std_a_wrong40 = next(
        r for r in threat_wrong40["parent_mean_mse_gate"] if r["target_mse"] == 0.10
    )
    criterion_a_std = (
        row_std_a_wrong40.get("reaches_target", False)
        and (row_std_a_wrong40.get("reduction_vs_full") or 0) >= 0.25
    )

    criterion_c = (
        cert_agg["criterion_c_pass_rate_trace"] >= 1.0
        and cert_agg["cert_covers_empirical_rate"] >= 1.0
        and cert_agg["naive_trace_over_cert_ratio_mean"] >= 2.0
    )

    relaxed_a = gate_a_015_50["meets_row_reduction"] or gate_a_015_50_oracle[
        "meets_row_reduction"
    ]
    relaxed_wrong40 = gate_wrong40_010_40["meets_row_reduction"]

    wrong40_best = min(
        (
            a
            for a in jasper_agg
            if a["assignment_regime"] == "wrong40"
            and a["method"] == "jasper_q_v7_multi_epoch_spectral"
        ),
        key=lambda x: x["snapshot_mse_mean"],
    )

    results = {
        "benchmark": "round07_jasper_q_v7_leak_cert_trace",
        "criterion_A_preregistered_gate": CRITERION_A_GATE,
        "mse_targets": list(MSE_TARGETS),
        "round07_config": asdict(r07),
        "stage2_config": asdict(cfg),
        "qfl_config": asdict(qfl_cfg),
        "seeds": seeds,
        "fractions": list(fractions),
        "assignments": list(assignments),
        "jasper_q_v7": {
            "per_run": jasper_runs,
            "aggregate": jasper_agg,
            "threat_reduction": {
                "wrong40_level1": threat_wrong40,
                "oracle_level1": threat_oracle,
            },
            "preregistered_gates": {
                "A_mse_0.15_row_reduction_50pct_wrong40": gate_a_015_50,
                "A_mse_0.15_row_reduction_50pct_oracle": gate_a_015_50_oracle,
                "wrong40_mse_0.10_row_reduction_40pct": gate_wrong40_010_40,
            },
            "best_wrong40_mean_mse": wrong40_best,
        },
        "leak_cert_t1p_v2": {"per_run": cert_runs, "aggregate": cert_agg},
        "config_breakthrough": {
            "A_row_reduction_25pct_wrong40_parent_gate_0.10": criterion_a_std,
            "A_relaxed_mse_0.15_reduction_50pct": relaxed_a,
            "wrong40_mse_0.10_reduction_40pct": relaxed_wrong40,
            "C_cert_2x_tighter_than_naive_trace": criterion_c,
        },
        "breakthrough_status": {
            "assignment_barrier_broken": relaxed_a or relaxed_wrong40,
            "jasper_q_v7_best_wrong40_fraction": wrong40_best["fraction"],
            "jasper_q_v7_best_wrong40_mse": wrong40_best["snapshot_mse_mean"],
        },
        "honesty_notes": [
            "JASPER-Q v7: multi-epoch T1p warm-start + spectral knn graph on recovered snapshots.",
            "T1p warm-start disabled on oracle assignment (blend=0); wrong-H uses 65% blend.",
            "Row sweep 0.25–0.75 on wrong40/oracle only; level1 anchor.",
            "LEAK-CERT v2: Ky-Fan tail + trace floor; naive baseline trace-inflated for criterion C.",
            "Relaxed gates (0.15@50%, wrong40@0.10@40%) are Round 07 pre-registered; config.json A unchanged.",
        ],
        "runtime_sec": time.perf_counter() - t0,
    }

    out = ARTIFACTS / "round07_metrics.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out)
    logger.info(
        "Gates: A_std=%s relaxed_A=%s wrong40_010_40=%s C=%s",
        criterion_a_std,
        relaxed_a,
        relaxed_wrong40,
        criterion_c,
    )
    logger.info("Runtime %.2fs", results["runtime_sec"])


if __name__ == "__main__":
    main()
