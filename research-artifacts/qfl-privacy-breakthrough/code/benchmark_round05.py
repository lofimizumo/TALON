#!/usr/bin/env python3
"""Round-05 QFL privacy benchmark — Snapshot-DP defense (criterion B).

Clients add Gaussian or Laplace noise to batch-mean snapshots before gradient
publish. Measures attacker snapshot MSE vs utility:

  - SHARD Stage-2 (co-occurrence GARD, wrong40, level1 anchor)
  - JASPER-Q (wrong40, level1, fraction 0.25)
  - LASA-QTERM T1p (QFL surrogate track)

Grid over (mechanism, epsilon, sigma). Criterion B: >=50% attack MSE increase
(attack worse) at <=10% normalized utility loss on at least one grid cell.

Secondary: wrong40 @ MSE 0.15 with >=50% row reduction (relaxed row gate).
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

_spec04 = importlib.util.spec_from_file_location(
    "benchmark_round04", RUN_ROOT / "code" / "benchmark_round04.py"
)
_r04 = importlib.util.module_from_spec(_spec04)
sys.modules["benchmark_round04"] = _r04
assert _spec04.loader is not None
_spec04.loader.exec_module(_r04)

sys.path.insert(0, str(VENDOR.parent))
sys.path.insert(0, str(QTERM / "code"))

from shard_sim.attacker import ShardAttacker  # noqa: E402
from shard_sim.surrogate_model import SurrogateQFL  # noqa: E402
from qterm_attack import QtermAttack, QtermConfig, QtermTier  # noqa: E402

MSE_TARGETS = _r02.MSE_TARGETS
CRITERION_A_GATE = _r02.CRITERION_A_GATE
Stage2Config = _r02.Stage2Config
QflConfig = _r02.QflConfig
Round04Config = _r04.Round04Config

make_low_rank_snapshots = _r02.make_low_rank_snapshots
make_incidence = _r02.make_incidence
select_rows = _r02.select_rows
corrupt_incidence = _r02.corrupt_incidence
cooccurrence_laplacian = _r02.cooccurrence_laplacian
solve_map = _r02.solve_map
select_graph_map = _r02.select_graph_map
snapshot_mse = _r02.snapshot_mse
threat_reduction_tables = _r02.threat_reduction_tables
row_members = _r02.row_members
resolve_mean_anchor = _r02.resolve_mean_anchor
anchor_weight_for_mode = _r02.anchor_weight_for_mode
hungarian_snapshot_mse = _r02.hungarian_snapshot_mse
make_epoch_batches = _r02.make_epoch_batches
simulate_gradients = _r02.simulate_gradients
run_stage2_one = _r02.run_stage2_one
normalized_utility_loss = _r02.normalized_utility_loss

run_jasper_q = _r04.run_jasper_q
update_soft_assignments = _r04.update_soft_assignments
recover_t1p_snapshots = _r04.recover_t1p_snapshots
stage_cfg_from_qfl = _r04.stage_cfg_from_qfl


@dataclass(frozen=True)
class Round05Config:
    mechanisms: tuple[str, ...] = ("gaussian", "laplace")
    epsilon_grid: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    sigma_grid: tuple[float, ...] = (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)
    batch_mean_l2_sensitivity: float = 2.0
    max_utility_loss_fraction: float = 0.10
    attack_mse_increase_target: float = 0.50
    stage2_assignment: str = "wrong40"
    stage2_fraction: float = 0.25
    stage2_anchor: str = "level1_estimate"
    stage2_method: str = "gard_cooccurrence_graph"
    jasper_fraction: float = 0.25
    secondary_mse_target: float = 0.15
    secondary_row_reduction: float = 0.50


def setup_logging() -> logging.Logger:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    logging.getLogger("shard_sim.attacker").setLevel(logging.WARNING)
    logger = logging.getLogger("round05")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(LOGS / "experiment_round05.log", mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def dp_noise_scale(
    epsilon: float,
    sigma: float,
    sensitivity: float,
    dim_g: int,
) -> float:
    if sigma > 0.0:
        return sigma
    return sensitivity / max(epsilon, 1e-6)


def draw_dp_noise(
    rng: np.random.Generator,
    mechanism: str,
    scale: float,
    dim_g: int,
) -> np.ndarray:
    if mechanism == "gaussian":
        return rng.normal(0.0, scale, size=(dim_g,))
    if mechanism == "laplace":
        return rng.laplace(0.0, scale / np.sqrt(dim_g), size=(dim_g,))
    raise ValueError(f"unknown mechanism {mechanism}")


def batch_mean_noise_cache(
    s_true: np.ndarray,
    h_rows: np.ndarray,
    cfg: Stage2Config,
    mechanism: str,
    epsilon: float,
    sigma: float,
    seed: int,
    r05: Round05Config,
) -> dict[tuple[int, ...], np.ndarray]:
    rng = np.random.default_rng(seed + 120_000)
    scale = dp_noise_scale(
        epsilon,
        sigma,
        r05.batch_mean_l2_sensitivity / np.sqrt(cfg.dim_g),
        cfg.dim_g,
    )
    cache: dict[tuple[int, ...], np.ndarray] = {}
    for row in h_rows:
        members = row_members(row)
        key = tuple(sorted(members.tolist()))
        if key not in cache:
            batch_mean = s_true[members].mean(axis=0)
            cache[key] = batch_mean + draw_dp_noise(rng, mechanism, scale, cfg.dim_g)
    return cache


def build_m_obs_from_cache(
    h_rows: np.ndarray,
    cache: dict[tuple[int, ...], np.ndarray],
    cfg: Stage2Config,
    seed: int,
    *,
    channel_noise: bool,
) -> np.ndarray:
    rng = np.random.default_rng(seed + 121_000)
    rows = []
    for row in h_rows:
        key = tuple(sorted(row_members(row).tolist()))
        obs = cache[key].copy()
        if channel_noise:
            obs = obs + cfg.noise_std * rng.normal(size=obs.shape)
        rows.append(obs)
    return np.vstack(rows)


def run_stage2_with_m_obs(
    seed: int,
    fraction: float,
    assignment_regime: str,
    mean_anchor_mode: str,
    cfg: Stage2Config,
    m_obs: np.ndarray,
    *,
    method: str,
) -> float:
    s_true = make_low_rank_snapshots(cfg, seed)
    h_full, _ = make_incidence(cfg, seed)
    keep = select_rows(h_full, fraction, seed)
    h_true = h_full[keep]
    h_assumed, _ = corrupt_incidence(h_true, cfg, assignment_regime, seed)
    mean_snapshot = resolve_mean_anchor(mean_anchor_mode, s_true, h_assumed, m_obs, cfg, seed)
    anchor_weight = anchor_weight_for_mode(mean_anchor_mode, cfg)
    train_idx, val_idx = _r02.split_train_val(len(keep), cfg, seed)
    rank_aug = int(
        np.linalg.matrix_rank(
            np.vstack([h_assumed, np.ones((1, cfg.n_samples), dtype=np.float64) / cfg.n_samples])
        )
    )
    lap = cooccurrence_laplacian(h_assumed)
    s_g, _ = select_graph_map(
        h_assumed[train_idx],
        m_obs[train_idx],
        h_assumed[val_idx],
        m_obs[val_idx],
        h_assumed,
        m_obs,
        mean_snapshot,
        lap,
        cfg,
        anchor_weight,
        rank_aug,
    )
    if method == "shard_style_ls":
        s_g = solve_map(h_assumed, m_obs, mean_snapshot, cfg, anchor_weight=anchor_weight)
    return snapshot_mse(s_g, s_true)


def run_jasper_q_with_m_obs(
    seed: int,
    fraction: float,
    assignment_regime: str,
    mean_anchor_mode: str,
    cfg: Stage2Config,
    qfl_cfg: QflConfig,
    r04: Round04Config,
    m_obs: np.ndarray,
) -> float:
    s_true, s_t1p, _ = recover_t1p_snapshots(seed, qfl_cfg)
    cfg = stage_cfg_from_qfl(qfl_cfg)
    h_full, metadata = make_incidence(cfg, seed)
    keep = select_rows(h_full, fraction, seed)
    metadata_obs = [metadata[i] for i in keep]
    h_true = h_full[keep]
    h_prior, _ = corrupt_incidence(h_true, cfg, assignment_regime, seed)
    mean_snapshot = resolve_mean_anchor(mean_anchor_mode, s_true, h_prior, m_obs, cfg, seed)
    anchor_weight = anchor_weight_for_mode(mean_anchor_mode, cfg)
    lap_co = cooccurrence_laplacian(h_prior)
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
    return snapshot_mse(s_hat, s_true)


def simulate_gradients_snapshot_dp(
    snapshots: np.ndarray,
    batches: list[list[int]],
    surrogate: SurrogateQFL,
    mechanism: str,
    epsilon: float,
    sigma: float,
    seed: int,
    r05: Round05Config,
) -> tuple[np.ndarray, list[list[np.ndarray]]]:
    rng = np.random.default_rng(seed + 130_000)
    scale = dp_noise_scale(
        epsilon,
        sigma,
        r05.batch_mean_l2_sensitivity / np.sqrt(surrogate.dim_g),
        surrogate.dim_g,
    )
    a_epoch = surrogate.generate_coefficient_matrix()
    grads: list[np.ndarray] = []
    for batch_indices in batches:
        mean_snap = snapshots[batch_indices].mean(axis=0)
        noisy_mean = mean_snap + draw_dp_noise(rng, mechanism, scale, surrogate.dim_g)
        g = a_epoch @ noisy_mean
        if surrogate.noise_level > 0:
            g = g + surrogate._np_rng.normal(0, surrogate.noise_level, size=g.shape)
        grads.append(g)
    return a_epoch, grads


def utility_snapshot_dp(
    true_snapshots: np.ndarray,
    surrogate: SurrogateQFL,
    mechanism: str,
    epsilon: float,
    sigma: float,
    seed: int,
    r05: Round05Config,
) -> tuple[float, float]:
    losses = []
    base_losses = []
    for e in range(3):
        batches = make_epoch_batches(true_snapshots.shape[0], 4, seed + e)
        for batch in batches:
            g_true = simulate_gradients(true_snapshots, [batch], surrogate)[1][0]
            g_pub = simulate_gradients_snapshot_dp(
                true_snapshots, [batch], surrogate, mechanism, epsilon, sigma, seed + e, r05
            )[1][0]
            losses.append(float(np.mean((g_true - g_pub) ** 2)))
            base_losses.append(float(np.mean(g_true**2)))
    raw = float(np.mean(losses))
    return raw, raw / max(float(np.mean(base_losses)), 1e-12)


def run_t1p_attack_mse(
    seed: int,
    qfl_cfg: QflConfig,
    mechanism: str | None,
    epsilon: float,
    sigma: float,
    r05: Round05Config,
) -> float:
    torch.manual_seed(seed)
    true_snapshots = make_low_rank_snapshots(
        Stage2Config(
            n_samples=qfl_cfg.n_samples,
            dim_g=qfl_cfg.dim_g,
            true_rank=qfl_cfg.true_rank,
            n_epochs=qfl_cfg.n_epochs,
        ),
        seed,
    )
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
        if mechanism is None:
            a_e, grads_e = simulate_gradients(true_snapshots, batches, surrogate)
        else:
            a_e, grads_e = simulate_gradients_snapshot_dp(
                true_snapshots, batches, surrogate, mechanism, epsilon, sigma, seed + e, r05
            )
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
    rec = qterm.recover(e_bar, coeff, batch_gradients=batch_grads)
    return hungarian_snapshot_mse(rec.snapshots, true_snapshots)


def run_shard_stage2_attack_mse(
    seed: int,
    cfg: Stage2Config,
    mechanism: str | None,
    epsilon: float,
    sigma: float,
    r05: Round05Config,
) -> float:
    s_true = make_low_rank_snapshots(cfg, seed)
    h_full, _ = make_incidence(cfg, seed)
    keep = select_rows(h_full, r05.stage2_fraction, seed)
    h_true = h_full[keep]
    rng = np.random.default_rng(seed + 50_000 + len(keep))
    m_undef = h_true @ s_true + cfg.noise_std * rng.normal(size=(len(keep), cfg.dim_g))
    if mechanism is None:
        m_obs = m_undef
    else:
        cache = batch_mean_noise_cache(
            s_true, h_true, cfg, mechanism, epsilon, sigma, seed, r05
        )
        m_obs = build_m_obs_from_cache(h_true, cache, cfg, seed, channel_noise=True)
    return run_stage2_with_m_obs(
        seed,
        r05.stage2_fraction,
        r05.stage2_assignment,
        r05.stage2_anchor,
        cfg,
        m_obs,
        method=r05.stage2_method,
    )


def run_jasper_attack_mse(
    seed: int,
    qfl_cfg: QflConfig,
    r04: Round04Config,
    r05: Round05Config,
    mechanism: str | None,
    epsilon: float,
    sigma: float,
) -> float:
    cfg = stage_cfg_from_qfl(qfl_cfg)
    s_true, _, _ = recover_t1p_snapshots(seed, qfl_cfg)
    h_full, _ = make_incidence(cfg, seed)
    keep = select_rows(h_full, r05.jasper_fraction, seed)
    h_true = h_full[keep]
    rng = np.random.default_rng(seed + 50_000 + len(keep))
    if mechanism is None:
        m_obs = h_true @ s_true + cfg.noise_std * rng.normal(size=(len(keep), cfg.dim_g))
    else:
        cache = batch_mean_noise_cache(s_true, h_true, cfg, mechanism, epsilon, sigma, seed, r05)
        m_obs = build_m_obs_from_cache(h_true, cache, cfg, seed, channel_noise=True)
    return run_jasper_q_with_m_obs(
        seed,
        r05.jasper_fraction,
        r05.stage2_assignment,
        r05.stage2_anchor,
        cfg,
        qfl_cfg,
        r04,
        m_obs,
    )


def attack_increase_fraction(def_mse: float, undef_mse: float) -> float:
    return (def_mse - undef_mse) / max(undef_mse, 1e-12)


def _cell_config_summary(cell: dict) -> dict:
    return {
        "mechanism": cell["mechanism"],
        "epsilon": cell["epsilon"],
        "sigma": cell["sigma"],
        "seed_example": cell["seed"],
        "utility_loss_fraction_normalized": cell["utility_loss_fraction_normalized"],
        "t1p_attack_increase": cell["t1p_attack_increase"],
        "shard_attack_increase": cell["shard_attack_increase"],
        "jasper_attack_increase": cell["jasper_attack_increase"],
        "meets_utility_constraint": cell["meets_utility_constraint"],
        "meets_B_any_attack": cell["meets_B_any_attack"],
    }


def run_grid_cell(
    seed: int,
    mechanism: str,
    epsilon: float,
    sigma: float,
    stage2_cfg: Stage2Config,
    qfl_cfg: QflConfig,
    r04: Round04Config,
    r05: Round05Config,
    baselines: dict,
) -> dict:
    util_raw, util_frac = utility_snapshot_dp(
        make_low_rank_snapshots(
            Stage2Config(
                n_samples=qfl_cfg.n_samples,
                dim_g=qfl_cfg.dim_g,
                true_rank=qfl_cfg.true_rank,
                n_epochs=qfl_cfg.n_epochs,
            ),
            seed,
        ),
        SurrogateQFL(
            input_dim=64,
            dim_g=qfl_cfg.dim_g,
            n_params=qfl_cfg.dim_g,
            noise_level=qfl_cfg.noise_level,
            seed=seed,
        ),
        mechanism,
        epsilon,
        sigma,
        seed,
        r05,
    )
    t1p = run_t1p_attack_mse(seed, qfl_cfg, mechanism, epsilon, sigma, r05)
    shard = run_shard_stage2_attack_mse(seed, stage2_cfg, mechanism, epsilon, sigma, r05)
    jasper = run_jasper_attack_mse(seed, qfl_cfg, r04, r05, mechanism, epsilon, sigma)
    return {
        "seed": seed,
        "mechanism": mechanism,
        "epsilon": epsilon,
        "sigma": sigma,
        "utility_gradient_mse": util_raw,
        "utility_loss_fraction_normalized": util_frac,
        "meets_utility_constraint": util_frac <= r05.max_utility_loss_fraction,
        "t1p_mse_undefended": baselines["t1p"][seed],
        "t1p_mse_defended": t1p,
        "t1p_attack_increase": attack_increase_fraction(t1p, baselines["t1p"][seed]),
        "shard_stage2_mse_undefended": baselines["shard"][seed],
        "shard_stage2_mse_defended": shard,
        "shard_attack_increase": attack_increase_fraction(shard, baselines["shard"][seed]),
        "jasper_q_mse_undefended": baselines["jasper"][seed],
        "jasper_q_mse_defended": jasper,
        "jasper_attack_increase": attack_increase_fraction(jasper, baselines["jasper"][seed]),
        "meets_B_t1p": (
            util_frac <= r05.max_utility_loss_fraction
            and attack_increase_fraction(t1p, baselines["t1p"][seed]) >= r05.attack_mse_increase_target
        ),
        "meets_B_shard": (
            util_frac <= r05.max_utility_loss_fraction
            and attack_increase_fraction(shard, baselines["shard"][seed])
            >= r05.attack_mse_increase_target
        ),
        "meets_B_jasper": (
            util_frac <= r05.max_utility_loss_fraction
            and attack_increase_fraction(jasper, baselines["jasper"][seed])
            >= r05.attack_mse_increase_target
        ),
        "meets_B_any_attack": False,
    }


def aggregate_grid(cells: list[dict], r05: Round05Config) -> dict:
    for c in cells:
        c["meets_B_any_attack"] = c["meets_B_t1p"] or c["meets_B_shard"] or c["meets_B_jasper"]

    feasible = [c for c in cells if c["meets_utility_constraint"]]
    b_pass = [c for c in cells if c["meets_B_any_attack"]]

    def _mean(key: str, subset: list[dict]) -> float:
        return float(np.mean([c[key] for c in subset])) if subset else float("nan")

    best_b = None
    if b_pass:
        best_b = max(
            b_pass,
            key=lambda c: max(
                c["t1p_attack_increase"],
                c["shard_attack_increase"],
                c["jasper_attack_increase"],
            ),
        )

    by_config: dict[tuple, list[dict]] = {}
    for c in cells:
        key = (c["mechanism"], c["epsilon"], c["sigma"])
        by_config.setdefault(key, []).append(c)

    config_summary = []
    for key, items in sorted(by_config.items()):
        mech, eps, sig = key
        config_summary.append(
            {
                "mechanism": mech,
                "epsilon": eps,
                "sigma": sig,
                "n_seeds": len(items),
                "mean_utility_loss_fraction": _mean("utility_loss_fraction_normalized", items),
                "mean_t1p_attack_increase": _mean("t1p_attack_increase", items),
                "mean_shard_attack_increase": _mean("shard_attack_increase", items),
                "mean_jasper_attack_increase": _mean("jasper_attack_increase", items),
                "feasible_utility_rate": float(np.mean([x["meets_utility_constraint"] for x in items])),
                "B_pass_rate_any_attack": float(np.mean([x["meets_B_any_attack"] for x in items])),
            }
        )

    def _attack_score(c: dict) -> float:
        return max(
            c["t1p_attack_increase"],
            c["shard_attack_increase"],
            c["jasper_attack_increase"],
        )

    closest_to_B = min(
        cells,
        key=lambda c: (
            max(0.0, c["utility_loss_fraction_normalized"] - r05.max_utility_loss_fraction),
            -_attack_score(c),
        ),
    )
    best_t1p_attack = max(cells, key=lambda c: c["t1p_attack_increase"])

    return {
        "n_grid_cells": len(cells),
        "n_feasible_utility": len(feasible),
        "n_B_pass_cells": len(b_pass),
        "best_B_pass_cell": best_b,
        "closest_to_B_cell": closest_to_B,
        "best_t1p_attack_increase_cell": best_t1p_attack,
        "config_summary": config_summary,
        "breakthrough_B_50pct_mse_increase_at_10pct_utility": len(b_pass) > 0,
    }


def jasper_threat_defended(
    seeds: list[int],
    qfl_cfg: QflConfig,
    r04: Round04Config,
    r05: Round05Config,
    mechanism: str,
    epsilon: float,
    sigma: float,
) -> dict:
    cfg = stage_cfg_from_qfl(qfl_cfg)
    runs = []
    for seed in seeds:
        for frac in (0.25, 0.15):
            s_true, _, _ = recover_t1p_snapshots(seed, qfl_cfg)
            h_full, _ = make_incidence(cfg, seed)
            keep = select_rows(h_full, frac, seed)
            h_true = h_full[keep]
            cache = batch_mean_noise_cache(s_true, h_true, cfg, mechanism, epsilon, sigma, seed, r05)
            m_obs = build_m_obs_from_cache(h_true, cache, cfg, seed, channel_noise=True)
            mse = run_jasper_q_with_m_obs(
                seed,
                frac,
                "wrong40",
                "level1_estimate",
                cfg,
                qfl_cfg,
                r04,
                m_obs,
            )
            runs.append(
                {
                    "seed": seed,
                    "fraction": frac,
                    "observed_rows": int(len(keep)),
                    "full_rows": int(h_full.shape[0]),
                    "assignment_regime": "wrong40",
                    "mean_anchor_mode": "level1_estimate",
                    "methods": {"jasper_q_joint_soft_gard": mse},
                }
            )
    return threat_reduction_tables(
        runs,
        cfg,
        assignment="wrong40",
        mean_anchor_mode="level1_estimate",
        method="jasper_q_joint_soft_gard",
    )


def main() -> None:
    logger = setup_logging()
    t0 = time.perf_counter()
    stage2_cfg = Stage2Config()
    qfl_cfg = QflConfig()
    r04 = Round04Config()
    r05 = Round05Config()
    seeds = [3, 7, 11, 19, 23]

    logger.info("Round 05 — Snapshot-DP defense (Gaussian/Laplace on batch means)")

    baselines = {"t1p": {}, "shard": {}, "jasper": {}}
    for seed in seeds:
        baselines["t1p"][seed] = run_t1p_attack_mse(seed, qfl_cfg, None, 0.0, 0.0, r05)
        baselines["shard"][seed] = run_shard_stage2_attack_mse(seed, stage2_cfg, None, 0.0, 0.0, r05)
        baselines["jasper"][seed] = run_jasper_attack_mse(seed, qfl_cfg, r04, r05, None, 0.0, 0.0)
        logger.info(
            "Undefended seed=%d t1p=%.4g shard=%.4g jasper=%.4g",
            seed,
            baselines["t1p"][seed],
            baselines["shard"][seed],
            baselines["jasper"][seed],
        )

    cells: list[dict] = []
    for mechanism in r05.mechanisms:
        for epsilon in r05.epsilon_grid:
            for sigma in r05.sigma_grid:
                for seed in seeds:
                    cell = run_grid_cell(
                        seed,
                        mechanism,
                        epsilon,
                        sigma,
                        stage2_cfg,
                        qfl_cfg,
                        r04,
                        r05,
                        baselines,
                    )
                    cells.append(cell)
                agg = [
                    c
                    for c in cells
                    if c["mechanism"] == mechanism
                    and c["epsilon"] == epsilon
                    and c["sigma"] == sigma
                ]
                logger.info(
                    "Grid %s eps=%.2f sigma=%.3f util_mean=%.3f t1p_inc=%.2f shard_inc=%.2f jasper_inc=%.2f B_any=%s",
                    mechanism,
                    epsilon,
                    sigma,
                    float(np.mean([c["utility_loss_fraction_normalized"] for c in agg])),
                    float(np.mean([c["t1p_attack_increase"] for c in agg])),
                    float(np.mean([c["shard_attack_increase"] for c in agg])),
                    float(np.mean([c["jasper_attack_increase"] for c in agg])),
                    any(c["meets_B_any_attack"] for c in agg),
                )

    grid_summary = aggregate_grid(cells, r05)

    secondary_pass = False
    secondary_detail: dict = {}
    best = grid_summary.get("best_B_pass_cell")
    if best:
        threat = jasper_threat_defended(
            seeds,
            qfl_cfg,
            r04,
            r05,
            best["mechanism"],
            best["epsilon"],
            best["sigma"],
        )
        row = next(
            (
                r
                for r in threat["parent_mean_mse_gate"]
                if r["target_mse"] == r05.secondary_mse_target
            ),
            None,
        )
        if row and row.get("reaches_target") and row.get("reduction_vs_full", 0) >= r05.secondary_row_reduction:
            secondary_pass = True
        secondary_detail = {
            "mechanism": best["mechanism"],
            "epsilon": best["epsilon"],
            "sigma": best["sigma"],
            "parent_gate_row": row,
            "target_mse": r05.secondary_mse_target,
            "min_row_reduction": r05.secondary_row_reduction,
        }

    results = {
        "benchmark": "round05_snapshot_dp_defense",
        "criterion_B_definition": ">=50% attack MSE increase vs undefended at <=10% normalized utility",
        "criterion_A_preregistered_gate": CRITERION_A_GATE,
        "mse_targets": list(MSE_TARGETS),
        "round05_config": asdict(r05),
        "stage2_config": asdict(stage2_cfg),
        "qfl_config": asdict(qfl_cfg),
        "seeds": seeds,
        "undefended_baselines": {
            "per_seed": baselines,
            "mean": {
                "t1p_mse": float(np.mean(list(baselines["t1p"].values()))),
                "shard_stage2_mse": float(np.mean(list(baselines["shard"].values()))),
                "jasper_q_mse": float(np.mean(list(baselines["jasper"].values()))),
            },
        },
        "snapshot_dp_grid": {
            "per_cell": cells,
            "summary": grid_summary,
        },
        "config_breakthrough": {
            "B_defense_50pct_mse_increase_at_10pct_utility": grid_summary[
                "breakthrough_B_50pct_mse_increase_at_10pct_utility"
            ],
            "secondary_wrong40_mse015_row_reduction_50pct": secondary_pass,
        },
        "best_defense_config": _cell_config_summary(best)
        if best
        else _cell_config_summary(grid_summary["closest_to_B_cell"]),
        "best_B_pass_config": _cell_config_summary(best) if best else None,
        "closest_to_B_config": _cell_config_summary(grid_summary["closest_to_B_cell"]),
        "best_t1p_attack_config": _cell_config_summary(
            grid_summary["best_t1p_attack_increase_cell"]
        ),
        "secondary_relaxed_gate": secondary_detail,
        "honesty_notes": [
            "Snapshot-DP noises batch means before gradient publish; not formal DP-SGD.",
            "Stage-2 attacker still uses wrong40 incidence + level1 anchor (co-occurrence GARD).",
            "JASPER-Q uses T1p warm-start (Round 04); defense perturbs observed batch gradients only.",
            "Utility is normalized batch-gradient MSE vs undefended (3-epoch proxy, Round 02).",
            "Criterion B uses attack MSE *increase* (higher MSE = worse attack), aligned with config.json.",
        ],
        "runtime_sec": time.perf_counter() - t0,
    }

    out = ARTIFACTS / "round05_metrics.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out)
    logger.info(
        "Criterion B pass=%s best=%s secondary=%s",
        grid_summary["breakthrough_B_50pct_mse_increase_at_10pct_utility"],
        results["best_defense_config"],
        secondary_pass,
    )
    logger.info("Runtime %.2fs", results["runtime_sec"])


if __name__ == "__main__":
    main()
