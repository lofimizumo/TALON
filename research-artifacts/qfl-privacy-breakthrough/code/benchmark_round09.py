#!/usr/bin/env python3
"""Round-09 QFL privacy — high-risk PROBE-RAND breakthrough.

Server randomizes probe matrix A^(e) each round (published to clients only).
Hypothesis: cross-epoch co-occurrence structure breaks for SHARD/JASPER attackers
that decode gradients with a stale or pooled A, while clients still train.

Compares JASPER-Q v7 @ wrong40 / 50% rows vs static-A baseline (Round 07 parity).
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

_spec02 = importlib.util.spec_from_file_location(
    "benchmark_round02", RUN_ROOT / "code" / "benchmark_round02.py"
)
_r02 = importlib.util.module_from_spec(_spec02)
sys.modules["benchmark_round02"] = _r02
assert _spec02.loader is not None
_spec02.loader.exec_module(_r02)

_spec07 = importlib.util.spec_from_file_location(
    "benchmark_round07", RUN_ROOT / "code" / "benchmark_round07.py"
)
_r07 = importlib.util.module_from_spec(_spec07)
sys.modules["benchmark_round07"] = _r07
assert _spec07.loader is not None
_spec07.loader.exec_module(_r07)

sys.path.insert(0, str(VENDOR.parent))
sys.path.insert(0, str(QTERM / "code"))

from shard_sim.attacker import ShardAttacker  # noqa: E402
from shard_sim.surrogate_model import SurrogateQFL  # noqa: E402
from qterm_attack import QtermAttack, QtermConfig, QtermTier  # noqa: E402
from terminal_attacks import _solve_batch_means  # noqa: E402

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
resolve_mean_anchor = _r02.resolve_mean_anchor
anchor_weight_for_mode = _r02.anchor_weight_for_mode
hungarian_snapshot_mse = _r02.hungarian_snapshot_mse
make_epoch_batches = _r02.make_epoch_batches

run_jasper_q_v7 = _r07.run_jasper_q_v7
Round07Config = _r07.Round07Config
blended_graph_laplacian = _r07.blended_graph_laplacian
t1p_warm_blend = _r07.t1p_warm_blend
update_soft_assignments = _r07._r04.update_soft_assignments
hard_overlap = _r07._r04.hard_overlap


@dataclass(frozen=True)
class Round09Config:
    breakthrough_lane: str = "PROBE-RAND"
    fraction: float = 0.50
    assignment_regime: str = "wrong40"
    mean_anchor_mode: str = "level1_estimate"
    seeds: tuple[int, ...] = (3, 7, 11, 19, 23)


def setup_logging() -> logging.Logger:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    logging.getLogger("shard_sim.attacker").setLevel(logging.WARNING)
    logger = logging.getLogger("round09")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(LOGS / "experiment_round09.log", mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def random_probe_matrix(dim_g: int, n_params: int, seed: int) -> np.ndarray:
    """Well-conditioned random probe A^(e) (server-chosen, per round)."""
    rng = np.random.default_rng(seed + 77_000)
    raw = rng.standard_normal((n_params, dim_g))
    q, _ = np.linalg.qr(raw.T)
    a = q.T[:n_params]
    # Scale so typical batch-mean snapshots map to O(1) gradients.
    a *= 1.0 / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    return a.astype(np.float64)


def simulate_probe_rand_fl(
    snapshots: np.ndarray,
    stage: Stage2Config,
    seed: int,
    *,
    randomize_probe: bool,
    static_a: np.ndarray | None = None,
) -> tuple[list[np.ndarray], list[list[np.ndarray]], list[list[np.ndarray]], np.ndarray]:
    """Returns (coeff_per_epoch, batch_gradients, true_batch_means, m_true_rows).

    Batch order matches ``make_incidence`` rows (SHARD Stage-2 alignment).
    """
    torch.manual_seed(seed)
    surrogate = SurrogateQFL(
        input_dim=64,
        dim_g=stage.dim_g,
        n_params=stage.dim_g,
        noise_level=stage.noise_std,
        seed=seed,
    )
    h_full, metadata = make_incidence(stage, seed)
    coeff: list[np.ndarray] = []
    grads: list[list[np.ndarray]] = []
    true_means: list[list[np.ndarray]] = []
    m_rows: list[np.ndarray] = []
    a_static = static_a
    if a_static is None and not randomize_probe:
        a_static = surrogate.generate_coefficient_matrix()

    epoch_to_idx: dict[int, list[int]] = {}
    for i, (ep, _) in enumerate(metadata):
        epoch_to_idx.setdefault(ep, []).append(i)

    for e in range(stage.n_epochs):
        if randomize_probe:
            a_e = random_probe_matrix(stage.dim_g, stage.dim_g, seed + 1000 * e)
        else:
            a_e = a_static if a_static is not None else surrogate.generate_coefficient_matrix()
        coeff.append(a_e)
        grads_e: list[np.ndarray] = []
        means_e: list[np.ndarray] = []
        for row_idx in epoch_to_idx[e]:
            members = _r02.row_members(h_full[row_idx])
            mean_snap = snapshots[members].mean(axis=0)
            g = a_e @ mean_snap
            if surrogate.noise_level > 0:
                g = g + surrogate._np_rng.normal(0, surrogate.noise_level, size=g.shape)
            grads_e.append(g)
            means_e.append(mean_snap.astype(np.float64))
            m_rows.append(mean_snap)
        grads.append(grads_e)
        true_means.append(means_e)
    return coeff, grads, true_means, np.vstack(m_rows)


def decode_mode_batch_means(
    coeff: list[np.ndarray],
    grads: list[list[np.ndarray]],
    dim_g: int,
    mode: str,
) -> np.ndarray:
    """Stack row-wise batch-mean snapshots for SHARD incidence rows (dim_g)."""
    rows: list[np.ndarray] = []
    a_pooled: np.ndarray | None = None
    if mode == "pooled_lstsq":
        # Attacker assumes one fixed A across rounds (mean of published round mats).
        a_pooled = np.mean(coeff, axis=0)

    for e, (a_e, grads_e) in enumerate(zip(coeff, grads)):
        if mode == "oracle_per_epoch":
            decoded = _solve_batch_means(a_e, grads_e, dim_g)
        elif mode == "stale_first_epoch":
            decoded = _solve_batch_means(coeff[0], grads_e, dim_g)
        elif mode == "pooled_lstsq":
            decoded = _solve_batch_means(a_pooled, grads_e, dim_g)
        elif mode == "raw_gradient":
            # Attacker treats D-dim gradients as pseudo snapshots (broken coordinates).
            decoded = np.stack(grads_e, axis=0)
            if decoded.shape[1] != dim_g:
                decoded = decoded[:, :dim_g]
        else:
            raise ValueError(mode)
        rows.extend(decoded)
    return np.vstack(rows)


def client_training_mse(
    coeff: list[np.ndarray],
    true_means: list[list[np.ndarray]],
    grads: list[list[np.ndarray]],
) -> float:
    """Utility: client has correct A^(e); gradient prediction error."""
    errs = []
    for a_e, means_e, grads_e in zip(coeff, true_means, grads):
        for mean_snap, g_obs in zip(means_e, grads_e):
            g_pred = a_e @ mean_snap
            errs.append(float(np.mean((g_pred - g_obs) ** 2)))
    return float(np.mean(errs))


def level1_with_assumed_coeff(
    assumed_coeff: list[np.ndarray],
    batch_gradients: list[list[np.ndarray]],
    dim_g: int,
) -> np.ndarray:
    attacker = ShardAttacker(n_samples=32, batch_size=4, dim_g=dim_g)
    return attacker.level1_mean_recovery(assumed_coeff, batch_gradients)


def recover_t1p_warm_probe(
    seed: int,
    qfl_cfg: QflConfig,
    coeff: list[np.ndarray],
    batch_grads: list[list[np.ndarray]],
    r07: Round07Config,
) -> np.ndarray:
    """T1p warm-start under probe-rand coeff (honest partial rows, real A^(e))."""
    e_bar = level1_with_assumed_coeff(coeff, batch_grads, qfl_cfg.dim_g)
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
    s_t1p = np.zeros_like(epoch_snaps[0])
    for snap, wt in zip(epoch_snaps, w):
        s_t1p += wt * snap
    return s_t1p


def stage_from_qfl(qfl_cfg: QflConfig) -> Stage2Config:
    return Stage2Config(
        n_samples=qfl_cfg.n_samples,
        batch_size=qfl_cfg.batch_size,
        dim_g=qfl_cfg.dim_g,
        true_rank=qfl_cfg.true_rank,
        n_epochs=qfl_cfg.n_epochs,
        noise_std=qfl_cfg.noise_level,
    )


def run_jasper_on_custom_m_obs(
    seed: int,
    fraction: float,
    assignment_regime: str,
    mean_anchor_mode: str,
    qfl_cfg: QflConfig,
    r07: Round07Config,
    *,
    s_true: np.ndarray,
    m_rows: np.ndarray,
    s_t1p: np.ndarray,
) -> dict[str, float]:
    cfg = stage_from_qfl(qfl_cfg)
    h_full, metadata = make_incidence(cfg, seed)
    keep = select_rows(h_full, fraction, seed)
    metadata_obs = [metadata[i] for i in keep]
    m_obs = m_rows[keep]
    h_prior, _ = corrupt_incidence(h_full[keep], cfg, assignment_regime, seed)
    # Align h_prior row count with selected rows
    h_prior = h_prior[: len(keep)]

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

    r04_proxy = _r07._r04.Round04Config(
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
        "gard_mse": snapshot_mse(s_gard, s_true),
        "jasper_mse": snapshot_mse(s_hat, s_true),
        "hard_overlap": hard_overlap(h_soft, h_full[keep], cfg),
    }


def encoding_residual_mse(
    h_assumed: np.ndarray,
    m_obs: np.ndarray,
    s_true: np.ndarray,
) -> float:
    """MSE of m_obs vs H @ S under wrong incidence (co-occurrence encoding stress)."""
    s_ls, *_ = np.linalg.lstsq(h_assumed, m_obs, rcond=None)
    pred = h_assumed @ s_ls
    return float(np.mean((pred - m_obs) ** 2))


def run_probe_rand_seed(
    seed: int,
    qfl_cfg: QflConfig,
    r07: Round07Config,
    r09: Round09Config,
) -> dict:
    stage = stage_from_qfl(qfl_cfg)
    s_true = make_low_rank_snapshots(stage, seed)
    cfg_r07 = Stage2Config()

    # Round-07 parity baseline (snapshot-space SHARD rows, static implicit A).
    baseline = run_jasper_q_v7(
        seed,
        r09.fraction,
        r09.assignment_regime,
        r09.mean_anchor_mode,
        cfg_r07,
        qfl_cfg,
        r07,
    )

    coeff_static, grads_static, means_static, m_static_rows = simulate_probe_rand_fl(
        s_true, stage, seed, randomize_probe=False
    )
    coeff_rand, grads_rand, means_rand, m_rand_true_rows = simulate_probe_rand_fl(
        s_true, stage, seed, randomize_probe=True
    )
    m_oracle_check = decode_mode_batch_means(
        coeff_rand, grads_rand, qfl_cfg.dim_g, "oracle_per_epoch"
    )
    oracle_row_mse = float(np.mean((m_oracle_check - m_rand_true_rows) ** 2))

    client_mse_static = client_training_mse(coeff_static, means_static, grads_static)
    client_mse_rand = client_training_mse(coeff_rand, means_rand, grads_rand)

    # T1p warm-start uses true per-epoch A (attacker sees gradients, not A — warm uses coeff_rand).
    s_t1p_rand = recover_t1p_warm_probe(seed, qfl_cfg, coeff_rand, grads_rand, r07)

    decode_modes = (
        "oracle_per_epoch",
        "stale_first_epoch",
        "pooled_lstsq",
        "raw_gradient",
    )
    probe_runs: dict[str, dict] = {}
    h_full, _ = make_incidence(stage, seed)
    keep = select_rows(h_full, r09.fraction, seed)
    h_prior, overlap = corrupt_incidence(h_full[keep], stage, r09.assignment_regime, seed)

    for mode in decode_modes:
        m_rows = decode_mode_batch_means(coeff_rand, grads_rand, qfl_cfg.dim_g, mode)
        metrics = run_jasper_on_custom_m_obs(
            seed,
            r09.fraction,
            r09.assignment_regime,
            r09.mean_anchor_mode,
            qfl_cfg,
            r07,
            s_true=s_true,
            m_rows=m_rows,
            s_t1p=s_t1p_rand,
        )
        # Level-1 anchor under wrong static-A assumption (no per-round A published to attacker).
        assumed = [coeff_rand[0]] * len(coeff_rand)
        e_bar_wrong = level1_with_assumed_coeff(assumed, grads_rand, qfl_cfg.dim_g)
        e_bar_true = level1_with_assumed_coeff(coeff_rand, grads_rand, qfl_cfg.dim_g)
        probe_runs[mode] = {
            **metrics,
            "encoding_residual_mse": encoding_residual_mse(h_prior, m_rows[keep], s_true),
            "level1_mse_vs_true_ebar": float(np.mean((e_bar_wrong - s_true.mean(axis=0)) ** 2)),
            "level1_oracle_coeff_mse": float(np.mean((e_bar_true - s_true.mean(axis=0)) ** 2)),
        }

    # Snapshot-space upper bound: true batch means, same rows as SHARD (no A mismatch).
    m_snapshot_rows = h_full @ s_true
    snapshot_oracle = run_jasper_on_custom_m_obs(
        seed,
        r09.fraction,
        r09.assignment_regime,
        r09.mean_anchor_mode,
        qfl_cfg,
        r07,
        s_true=s_true,
        m_rows=m_snapshot_rows,
        s_t1p=s_t1p_rand,
    )

    return {
        "seed": seed,
        "assignment_overlap": overlap,
        "oracle_decode_row_mse": oracle_row_mse,
        "client_grad_mse_static_a": client_mse_static,
        "client_grad_mse_probe_rand": client_mse_rand,
        "baseline_round07_jasper_mse": baseline["methods"]["jasper_q_v7_multi_epoch_spectral"],
        "baseline_round07_gard_mse": baseline["methods"]["gard_cooccurrence_wrong_h"],
        "probe_rand_snapshot_rows_jasper_mse": snapshot_oracle["jasper_mse"],
        "probe_rand_jasper_by_decode": probe_runs,
    }


def aggregate_runs(runs: list[dict]) -> dict:
    def mean_key(path: tuple[str, ...]) -> float:
        vals = []
        for r in runs:
            cur: object = r
            for k in path:
                cur = cur[k]  # type: ignore[index]
            vals.append(float(cur))
        return float(np.mean(vals))

    decode_modes = runs[0]["probe_rand_jasper_by_decode"].keys()
    agg_decode = {}
    for mode in decode_modes:
        agg_decode[mode] = {
            "jasper_mse_mean": mean_key(("probe_rand_jasper_by_decode", mode, "jasper_mse")),
            "gard_mse_mean": mean_key(("probe_rand_jasper_by_decode", mode, "gard_mse")),
            "encoding_residual_mse_mean": mean_key(
                ("probe_rand_jasper_by_decode", mode, "encoding_residual_mse")
            ),
        }

    baseline_mean = mean_key(("baseline_round07_jasper_mse",))
    stale_mean = agg_decode["stale_first_epoch"]["jasper_mse_mean"]
    oracle_mean = agg_decode["oracle_per_epoch"]["jasper_mse_mean"]
    snapshot_rows_mean = mean_key(("probe_rand_snapshot_rows_jasper_mse",))

    return {
        "n_seeds": len(runs),
        "baseline_jasper_mse_mean": baseline_mean,
        "probe_rand_snapshot_rows_jasper_mean": snapshot_rows_mean,
        "client_grad_mse_probe_rand_mean": mean_key(("client_grad_mse_probe_rand",)),
        "client_grad_mse_static_a_mean": mean_key(("client_grad_mse_static_a",)),
        "probe_rand_decode": agg_decode,
        "jasper_recovers_under_wrong40": {
            "baseline_at_50pct": baseline_mean,
            "probe_rand_stale_decode": stale_mean,
            "probe_rand_oracle_decode_cheating": oracle_mean,
            "stale_worse_than_baseline": stale_mean > baseline_mean * 1.05,
            "oracle_near_baseline": abs(oracle_mean - baseline_mean) < 0.08,
            "defense_breaks_cooccurrence_without_a": stale_mean > 1.2,
        },
    }


def main() -> None:
    logger = setup_logging()
    t0 = time.perf_counter()
    r09 = Round09Config()
    cfg = Stage2Config()
    qfl_cfg = QflConfig()
    r07 = Round07Config()

    logger.info(
        "Round 09 — %s: per-round random A^(e), JASPER @ %s %.0f%% rows",
        r09.breakthrough_lane,
        r09.assignment_regime,
        r09.fraction * 100,
    )

    runs = [run_probe_rand_seed(s, qfl_cfg, r07, r09) for s in r09.seeds]
    for r in runs:
        stale = r["probe_rand_jasper_by_decode"]["stale_first_epoch"]["jasper_mse"]
        logger.info(
            "seed=%d baseline_jasper=%.4g stale_decode_jasper=%.4g client_mse=%.4g",
            r["seed"],
            r["baseline_round07_jasper_mse"],
            stale,
            r["client_grad_mse_probe_rand"],
        )

    agg = aggregate_runs(runs)
    verdict = agg["jasper_recovers_under_wrong40"]

    results = {
        "benchmark": "round09_probe_rand_breakthrough",
        "breakthrough_lane": r09.breakthrough_lane,
        "threat_model": {
            "server_publishes": "A^(e) per round to clients only (not to SHARD attacker)",
            "attacker_observes": "batch gradients, wrong secure-aggregation incidence H",
            "attacker_does_not_observe": "per-round A^(e) (must guess stale/pooled A to decode)",
        },
        "round09_config": asdict(r09),
        "round07_config": asdict(r07),
        "per_seed": runs,
        "aggregate": agg,
        "verdict": {
            "probe_rand_breaks_jasper_without_per_epoch_a": verdict[
                "defense_breaks_cooccurrence_without_a"
            ],
            "jasper_still_recovers_if_attacker_knows_a_per_epoch": verdict[
                "oracle_near_baseline"
            ],
            "clients_train_with_published_a": agg["client_grad_mse_probe_rand_mean"] < 1e-3,
            "honest_failure": not verdict["defense_breaks_cooccurrence_without_a"]
            and verdict["baseline_at_50pct"] < 1.0,
        },
        "honesty_notes": [
            "PROBE-RAND does not hide A from a server-side attacker who logs probes; "
            "threat is SHARD/JASPER path with wrong H and no per-round A publication.",
            "oracle_per_epoch decode is an upper bound (cheating); stale_first_epoch "
            "models legacy single-A assumption.",
            "SUBSPACE-MIX deferred (single high-risk lane per Round 09).",
        ],
        "runtime_sec": time.perf_counter() - t0,
    }

    out = ARTIFACTS / "round09_metrics.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out)
    logger.info("Verdict: %s", results["verdict"])
    logger.info("Runtime %.2fs", results["runtime_sec"])


if __name__ == "__main__":
    main()
