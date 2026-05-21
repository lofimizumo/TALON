#!/usr/bin/env python3
"""Round-06 QFL privacy benchmark — Path 2 milestone + assignment barrier (Path 3).

Primary (Path 2): wrong40 + level1_estimate @ 50% observed rows (30/60).
Compare GARD co-occurrence vs JASPER-Q (T1p warm-start off on oracle assignment).
Pre-registered: mean snapshot MSE <= 0.15 on >= 4/5 seeds (best attacker).

Path 3: cites paper/assignment_barrier_theorem.md (ABT-1).

Cross-cutting: T1p audit uses ShardAttacker.level1_mean_recovery (Round 02 parity).
Snapshot-DP is not evaluated (killed Round 05).
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
split_train_val = _r02.split_train_val
cooccurrence_laplacian = _r02.cooccurrence_laplacian
solve_map = _r02.solve_map
select_graph_map = _r02.select_graph_map
snapshot_mse = _r02.snapshot_mse
row_members = _r02.row_members
resolve_mean_anchor = _r02.resolve_mean_anchor
anchor_weight_for_mode = _r02.anchor_weight_for_mode
hungarian_snapshot_mse = _r02.hungarian_snapshot_mse
make_epoch_batches = _r02.make_epoch_batches
simulate_gradients = _r02.simulate_gradients

run_jasper_q = _r04.run_jasper_q
update_soft_assignments = _r04.update_soft_assignments
recover_t1p_snapshots = _r04.recover_t1p_snapshots
stage_cfg_from_qfl = _r04.stage_cfg_from_qfl
hard_overlap = _r04.hard_overlap

THEOREM_ID = "ABT-1"
THEOREM_PATH = RUN_ROOT / "paper" / "assignment_barrier_theorem.md"


@dataclass(frozen=True)
class Round06Config:
    primary_fraction: float = 0.50
    primary_assignment: str = "wrong40"
    primary_anchor: str = "level1_estimate"
    mse_target: float = 0.15
    min_seeds_pass: int = 4
    snapshot_dp_killed: bool = True
    disable_t1p_warm_on_oracle: bool = True


def setup_logging() -> logging.Logger:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    logging.getLogger("shard_sim.attacker").setLevel(logging.WARNING)
    logger = logging.getLogger("round06")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(LOGS / "experiment_round06.log", mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def t1p_warm_blend(assignment_regime: str, r04: Round04Config, r06: Round06Config) -> float:
    if r06.disable_t1p_warm_on_oracle and assignment_regime == "oracle":
        return 0.0
    return r04.t1p_warm_start_blend


def observed_batch_residual_mse(
    m_obs: np.ndarray,
    h_used: np.ndarray,
    s_hat: np.ndarray,
) -> float:
    """Mean squared residual of published batch means under used incidence H."""
    preds = h_used @ s_hat
    return float(np.mean((m_obs - preds) ** 2))


def run_gard_cooccurrence(
    seed: int,
    fraction: float,
    assignment_regime: str,
    mean_anchor_mode: str,
    cfg: Stage2Config,
) -> dict:
    s_true = make_low_rank_snapshots(cfg, seed)
    h_full, _ = make_incidence(cfg, seed)
    keep = select_rows(h_full, fraction, seed)
    h_true = h_full[keep]
    rng = np.random.default_rng(seed + 50_000 + len(keep))
    m_obs = h_true @ s_true + cfg.noise_std * rng.normal(size=(len(keep), cfg.dim_g))
    h_prior, overlap = corrupt_incidence(h_true, cfg, assignment_regime, seed)
    mean_snapshot = resolve_mean_anchor(mean_anchor_mode, s_true, h_prior, m_obs, cfg, seed)
    anchor_weight = anchor_weight_for_mode(mean_anchor_mode, cfg)
    train_idx, val_idx = split_train_val(len(keep), cfg, seed)
    rank_aug = int(
        np.linalg.matrix_rank(
            np.vstack([h_prior, np.ones((1, cfg.n_samples), dtype=np.float64) / cfg.n_samples])
        )
    )
    lap = cooccurrence_laplacian(h_prior)
    s_gard, _ = select_graph_map(
        h_prior[train_idx],
        m_obs[train_idx],
        h_prior[val_idx],
        m_obs[val_idx],
        h_prior,
        m_obs,
        mean_snapshot,
        lap,
        cfg,
        anchor_weight,
        rank_aug,
    )
    rank_true = int(
        np.linalg.matrix_rank(
            np.vstack([h_true, np.ones((1, cfg.n_samples), dtype=np.float64) / cfg.n_samples])
        )
    )
    return {
        "seed": seed,
        "fraction": fraction,
        "assignment_regime": assignment_regime,
        "mean_anchor_mode": mean_anchor_mode,
        "observed_rows": int(len(keep)),
        "full_rows": int(h_full.shape[0]),
        "assignment_overlap_prior": overlap,
        "incidence_rank_used": rank_aug,
        "incidence_rank_true": rank_true,
        "snapshot_mse": snapshot_mse(s_gard, s_true),
        "used_vs_true_residual": {
            "residual_mse_used_h": observed_batch_residual_mse(m_obs, h_prior, s_gard),
            "residual_mse_true_h": observed_batch_residual_mse(m_obs, h_true, s_gard),
            "ratio_true_over_used": None,
        },
    }


def run_jasper_q_r06(
    seed: int,
    fraction: float,
    assignment_regime: str,
    mean_anchor_mode: str,
    cfg: Stage2Config,
    qfl_cfg: QflConfig,
    r04: Round04Config,
    r06: Round06Config,
) -> dict:
    s_true, s_t1p, _ = recover_t1p_snapshots(seed, qfl_cfg)
    cfg = stage_cfg_from_qfl(qfl_cfg)
    h_full, metadata = make_incidence(cfg, seed)
    keep = select_rows(h_full, fraction, seed)
    metadata_obs = [metadata[i] for i in keep]
    h_true = h_full[keep]
    rng = np.random.default_rng(seed + 50_000 + len(keep))
    m_obs = h_true @ s_true + cfg.noise_std * rng.normal(size=(len(keep), cfg.dim_g))
    h_prior, overlap = corrupt_incidence(h_true, cfg, assignment_regime, seed)
    blend = t1p_warm_blend(assignment_regime, r04, r06)
    mean_snapshot = resolve_mean_anchor(mean_anchor_mode, s_true, h_prior, m_obs, cfg, seed)
    anchor_weight = anchor_weight_for_mode(mean_anchor_mode, cfg)
    lap_co = cooccurrence_laplacian(h_prior)
    mean_warm = (1.0 - blend) * mean_snapshot + blend * s_t1p.mean(axis=0)
    h_soft = h_prior.copy()
    s_hat = blend * s_t1p + (1.0 - blend) * solve_map(
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
    res_used = observed_batch_residual_mse(m_obs, h_prior, s_hat)
    res_true = observed_batch_residual_mse(m_obs, h_true, s_hat)
    return {
        "seed": seed,
        "fraction": fraction,
        "assignment_regime": assignment_regime,
        "mean_anchor_mode": mean_anchor_mode,
        "observed_rows": int(len(keep)),
        "full_rows": int(h_full.shape[0]),
        "assignment_overlap_prior": overlap,
        "t1p_warm_blend": blend,
        "t1p_warm_mse": hungarian_snapshot_mse(s_t1p, s_true),
        "snapshot_mse": snapshot_mse(s_hat, s_true),
        "jasper_q_hard_overlap": hard_overlap(h_soft, h_true, cfg),
        "used_vs_true_residual": {
            "residual_mse_used_h": res_used,
            "residual_mse_true_h": res_true,
            "ratio_true_over_used": res_true / max(res_used, 1e-12),
        },
    }


def run_t1p_attack_mse_level1(
    seed: int,
    qfl_cfg: QflConfig,
) -> dict:
    """T1p with Round-02 ShardAttacker.level1_mean_recovery (not flat batch mean)."""
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
    attacker = ShardAttacker(
        dim_g=qfl_cfg.dim_g,
        n_samples=qfl_cfg.n_samples,
        batch_size=qfl_cfg.batch_size,
        max_iter=200,
        tol=1e-8,
        random_seed=seed,
    )
    coeff: list[np.ndarray] = []
    batch_grads: list[list[np.ndarray]] = []
    for e in range(qfl_cfg.n_epochs):
        batches = make_epoch_batches(qfl_cfg.n_samples, qfl_cfg.batch_size, seed + e)
        a_e, grads_e = simulate_gradients(true_snapshots, batches, surrogate)
        coeff.append(a_e)
        batch_grads.append(grads_e)
    e_bar = attacker.level1_mean_recovery(coeff, batch_grads)
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
    mse = hungarian_snapshot_mse(rec.snapshots, true_snapshots)
    return {
        "seed": seed,
        "t1p_mse_level1_recovery": mse,
        "recovery_stack": "ShardAttacker.level1_mean_recovery + QtermAttack.T1P",
    }


def path2_pass_counts(
    runs: list[dict],
    *,
    mse_key: str,
    target: float,
) -> dict:
    per_seed = {r["seed"]: r[mse_key] for r in runs}
    passes = {s: m <= target for s, m in per_seed.items()}
    return {
        "per_seed_mse": per_seed,
        "per_seed_pass": passes,
        "n_pass": sum(passes.values()),
        "n_seeds": len(passes),
        "mean_mse": float(np.mean(list(per_seed.values()))),
        "passes_gate": sum(passes.values()) >= 4,
    }


def enrich_residual_ratios(run: dict) -> None:
    uv = run["used_vs_true_residual"]
    used = uv["residual_mse_used_h"]
    uv["ratio_true_over_used"] = uv["residual_mse_true_h"] / max(used, 1e-12)


def main() -> None:
    logger = setup_logging()
    t0 = time.perf_counter()
    cfg = Stage2Config()
    qfl_cfg = QflConfig()
    r04 = Round04Config()
    r06 = Round06Config()
    seeds = [3, 7, 11, 19, 23]
    theorem_exists = THEOREM_PATH.is_file()

    logger.info("Round 06 — Path 2 wrong40@50%% + theorem %s", THEOREM_ID)

    gard_runs = [
        run_gard_cooccurrence(
            seed,
            r06.primary_fraction,
            r06.primary_assignment,
            r06.primary_anchor,
            cfg,
        )
        for seed in seeds
    ]
    jasper_runs = [
        run_jasper_q_r06(
            seed,
            r06.primary_fraction,
            r06.primary_assignment,
            r06.primary_anchor,
            cfg,
            qfl_cfg,
            r04,
            r06,
        )
        for seed in seeds
    ]
    for run in gard_runs + jasper_runs:
        enrich_residual_ratios(run)

    gard_gate = path2_pass_counts(gard_runs, mse_key="snapshot_mse", target=r06.mse_target)
    jasper_gate = path2_pass_counts(jasper_runs, mse_key="snapshot_mse", target=r06.mse_target)
    best_attacker = min(gard_gate["mean_mse"], jasper_gate["mean_mse"])
    best_method = (
        "jasper_q_joint_soft_gard"
        if jasper_gate["mean_mse"] <= gard_gate["mean_mse"]
        else "gard_cooccurrence_wrong_h"
    )
    best_runs = jasper_runs if best_method.startswith("jasper") else gard_runs
    best_gate = path2_pass_counts(best_runs, mse_key="snapshot_mse", target=r06.mse_target)
    path2_pass = best_gate["passes_gate"]

    for gr, jr in zip(gard_runs, jasper_runs):
        logger.info(
            "seed=%d gard_mse=%.4g jasper_mse=%.4g used_res=%.4g true_res=%.4g ratio=%.2f",
            gr["seed"],
            gr["snapshot_mse"],
            jr["snapshot_mse"],
            jr["used_vs_true_residual"]["residual_mse_used_h"],
            jr["used_vs_true_residual"]["residual_mse_true_h"],
            jr["used_vs_true_residual"]["ratio_true_over_used"],
        )

    # Oracle ablation: JASPER with warm-start disabled (Round 04 demand)
    oracle_runs = [
        run_jasper_q_r06(seed, r06.primary_fraction, "oracle", r06.primary_anchor, cfg, qfl_cfg, r04, r06)
        for seed in seeds
    ]
    oracle_blend_ok = all(r["t1p_warm_blend"] == 0.0 for r in oracle_runs)

    t1p_audit = [run_t1p_attack_mse_level1(seed, qfl_cfg) for seed in seeds]
    t1p_mean = float(np.mean([t["t1p_mse_level1_recovery"] for t in t1p_audit]))

    # Fraction 0.25 reference for theorem falsification (ABT-1 predicts floor >= 0.15)
    ref_025 = [
        run_jasper_q_r06(seed, 0.25, r06.primary_assignment, r06.primary_anchor, cfg, qfl_cfg, r04, r06)
        for seed in seeds
    ]
    ref_025_mean = float(np.mean([r["snapshot_mse"] for r in ref_025]))

    results = {
        "benchmark": "round06_path2_wrong40_50pct_assignment_barrier",
        "theorem": {
            "id": THEOREM_ID,
            "document": str(THEOREM_PATH.relative_to(RUN_ROOT)),
            "document_present": theorem_exists,
            "falsifiable_prediction_mse_floor_at_25pct_rows": 0.15,
            "observed_jasper_mean_mse_at_25pct": ref_025_mean,
            "prediction_holds_at_25pct": ref_025_mean >= 0.15,
        },
        "round06_config": asdict(r06),
        "round04_config": asdict(r04),
        "stage2_config": asdict(cfg),
        "qfl_config": asdict(qfl_cfg),
        "seeds": seeds,
        "snapshot_dp_status": "killed_round05",
        "path2_primary": {
            "fraction": r06.primary_fraction,
            "observed_rows": gard_runs[0]["observed_rows"],
            "full_rows": gard_runs[0]["full_rows"],
            "assignment": r06.primary_assignment,
            "anchor": r06.primary_anchor,
            "mse_target": r06.mse_target,
            "min_seeds_pass": r06.min_seeds_pass,
            "gard_cooccurrence": {"per_run": gard_runs, "gate": gard_gate},
            "jasper_q": {"per_run": jasper_runs, "gate": jasper_gate},
            "best_attacker": {
                "method": best_method,
                "mean_mse": best_gate["mean_mse"],
                "gate": best_gate,
            },
        },
        "oracle_jasper_ablation": {
            "fraction": r06.primary_fraction,
            "t1p_warm_disabled": oracle_blend_ok,
            "per_run": oracle_runs,
            "mean_mse": float(np.mean([r["snapshot_mse"] for r in oracle_runs])),
        },
        "t1p_audit_level1_recovery": {
            "per_seed": t1p_audit,
            "mean_mse": t1p_mean,
            "round05_flat_mean_baseline_reference": 0.427,
        },
        "used_vs_true_aggregate": {
            "jasper_mean_residual_used_h": float(
                np.mean([r["used_vs_true_residual"]["residual_mse_used_h"] for r in jasper_runs])
            ),
            "jasper_mean_residual_true_h": float(
                np.mean([r["used_vs_true_residual"]["residual_mse_true_h"] for r in jasper_runs])
            ),
            "jasper_mean_ratio_true_over_used": float(
                np.mean([r["used_vs_true_residual"]["ratio_true_over_used"] for r in jasper_runs])
            ),
        },
        "config_breakthrough": {
            "A_wrong40_mse_0.15_at_50pct_rows": path2_pass,
            "A_row_reduction_25pct_wrong40_parent_gate": False,
            "B_defense_50pct_mse_increase_at_10pct_utility": False,
            "C_cert_2x_tighter_than_naive": False,
            "A_wrong20_mse_0.15_at_50pct_rows": False,
            "A_unknown_random_mse_0.15_at_50pct_rows": False,
        },
        "breakthrough_status": {
            "path2_relaxed_milestone_pass": path2_pass,
            "assignment_barrier_broken": path2_pass,
            "theorem_abt1_linked": theorem_exists,
            "jasper_beats_gard_at_50pct": jasper_gate["mean_mse"] < gard_gate["mean_mse"] - 0.02,
        },
        "honesty_notes": [
            "Snapshot-DP killed after Round 05; no DP grid in Round 06.",
            "Path 2 success uses min(GARD,JASPER) snapshot MSE @ wrong40, 50% rows, level1 anchor.",
            "T1p audit lane only; uses level1_mean_recovery per Round 05 supervisor fix.",
            "used_vs_true residual: low used-H residual does not imply low snapshot MSE under wrong40.",
            f"Theorem {THEOREM_ID} in paper/assignment_barrier_theorem.md — rank deficiency under wrong incidence.",
        ],
        "runtime_sec": time.perf_counter() - t0,
    }

    out = ARTIFACTS / "round06_metrics.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out)
    logger.info(
        "Path2 A_wrong40_mse_0.15_at_50pct_rows=%s gard_mean=%.4g jasper_mean=%.4g best=%s",
        path2_pass,
        gard_gate["mean_mse"],
        jasper_gate["mean_mse"],
        best_method,
    )
    logger.info("Theorem %s present=%s ref_025_mean=%.4g", THEOREM_ID, theorem_exists, ref_025_mean)
    logger.info("Runtime %.2fs", results["runtime_sec"])


if __name__ == "__main__":
    main()
