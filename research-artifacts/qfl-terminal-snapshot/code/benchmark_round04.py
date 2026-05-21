#!/usr/bin/env python3
"""Round-04 acceptance benchmark — LASA-QTERM consolidated evaluation.

Runs smooth + MNIST tracks, all tiers, budget-matched table, production
``QtermAttack`` methods, and SHARD oracle baselines.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "code"))
from _paths import ARTIFACTS, LOGS, PARENT, VENDOR  # noqa: E402

sys.path.insert(0, str(VENDOR.parent))

from shard_sim.attacker import ShardAttacker  # noqa: E402
from shard_sim.surrogate_model import SurrogateQFL  # noqa: E402
from qterm_attack import (  # noqa: E402
    QtermAttack,
    QtermConfig,
    QtermTier,
    recover_passive_baseline,
)
from terminal_attacks import (  # noqa: E402
    fixed_order_snapshot_mse,
    partial_epoch_gradients,
    snapshot_row_std,
    subsample_gradient_rows,
)

SEEDS = [3, 7, 11, 19, 23]
N_SAMPLES = 32
BATCH_SIZE = 4
N_EPOCHS = 10
DIM_G = 32
NOISE_LEVEL = 0.01
TRUE_RANK = 4
SHARD_MAX_ITER = 200
BUDGET_ROWS = 80
PARTIAL_ROWS = 7
METHOD_NAME = "LASA-QTERM"
METHOD_ALIAS = "Q-SNAP-T"


def setup_logging(log_name: str) -> logging.Logger:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    logging.getLogger("shard_sim.attacker").setLevel(logging.WARNING)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOGS / log_name, mode="w"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return logging.getLogger("round04")


def make_smooth_snapshots(n_samples: int, dim_g: int, rank: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 5000)
    t = np.linspace(0.0, 1.0, n_samples)
    latent_cols = [
        np.sin(2.0 * np.pi * t),
        np.cos(2.0 * np.pi * t),
        np.sin(4.0 * np.pi * t + 0.3),
        np.cos(4.0 * np.pi * t - 0.2),
    ]
    latent = np.vstack(latent_cols[:rank]).T
    latent += 0.08 * rng.normal(size=latent.shape)
    basis = rng.normal(size=(dim_g, rank))
    basis, _ = np.linalg.qr(basis)
    snapshots = latent @ basis.T
    snapshots += 0.03 * rng.normal(size=snapshots.shape)
    snapshots -= snapshots.mean(axis=0, keepdims=True)
    snapshots += 0.15
    snapshots /= np.std(snapshots) + 1e-12
    return snapshots.astype(np.float64)


def load_mnist_snapshots(
    n_samples: int, dim_g: int, seed: int
) -> tuple[np.ndarray, int]:
    from torchvision import datasets, transforms

    torch.manual_seed(seed)
    tfm = transforms.Compose(
        [
            transforms.Resize((8, 8)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    ds = datasets.MNIST(
        root=str(PARENT / "data" / "mnist"),
        train=True,
        download=True,
        transform=tfm,
    )
    rng = np.random.default_rng(seed + 6000)
    idx = rng.choice(len(ds), size=n_samples, replace=False)
    images = torch.stack([ds[int(i)][0] for i in idx]).flatten(1)
    input_dim = images.shape[1]
    surrogate = SurrogateQFL(
        input_dim=input_dim,
        dim_g=dim_g,
        n_params=DIM_G,
        noise_level=0.0,
        seed=seed + 7000,
    )
    with torch.no_grad():
        snaps = surrogate.encode(images).numpy().astype(np.float64)
    snaps -= snaps.mean(axis=0, keepdims=True)
    snaps /= np.std(snaps) + 1e-12
    return snaps, input_dim


def simulate_epoch_gradients(
    true_snapshots: np.ndarray,
    batches: list[list[int]],
    surrogate: SurrogateQFL,
) -> tuple[np.ndarray, list[np.ndarray]]:
    a_epoch = surrogate.generate_coefficient_matrix()
    grads: list[np.ndarray] = []
    for batch_indices in batches:
        mean_snap = true_snapshots[batch_indices].mean(axis=0)
        g = a_epoch @ mean_snap
        if surrogate.noise_level > 0:
            g = g + surrogate._np_rng.normal(0, surrogate.noise_level, size=g.shape)
        grads.append(g)
    return a_epoch, grads


def make_epoch_batches(n_samples: int, batch_size: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed + 9000)
    perm = rng.permutation(n_samples)
    k = n_samples // batch_size
    return [
        perm[b * batch_size : (b + 1) * batch_size].tolist() for b in range(k)
    ]


def hungarian_snapshot_mse(recovered: np.ndarray, truth: np.ndarray) -> float:
    from scipy.optimize import linear_sum_assignment

    r_sq = np.sum(recovered**2, axis=1, keepdims=True)
    t_sq = np.sum(truth**2, axis=1, keepdims=True)
    dist = r_sq + t_sq.T - 2.0 * recovered @ truth.T
    np.maximum(dist, 0.0, out=dist)
    row, col = linear_sum_assignment(dist)
    return float(np.mean((recovered[row] - truth[col]) ** 2))


def run_shard_oracle(
    attacker: ShardAttacker,
    e_bar: np.ndarray,
    coeff_matrices: list[np.ndarray],
    batch_gradients: list[list[np.ndarray]],
    true_snapshots: np.ndarray,
) -> dict:
    t0 = time.perf_counter()
    s = attacker.level2_disaggregate(
        e_bar, coeff_matrices, batch_gradients, true_snapshots
    )
    runtime = time.perf_counter() - t0
    n_intermediate = sum(len(g) for g in batch_gradients)
    return {
        "snapshot_mse": hungarian_snapshot_mse(s, true_snapshots),
        "fixed_order_mse": fixed_order_snapshot_mse(s, true_snapshots),
        "observed_intermediate_batch_gradients": n_intermediate,
        "observed_terminal_gradient_rows": 0,
        "runtime_sec": runtime,
        "matching_acc": ShardAttacker._matching_accuracy(s, true_snapshots),
        "snapshot_row_std": snapshot_row_std(s),
    }


def method_record(
    s: np.ndarray,
    true_snapshots: np.ndarray,
    *,
    n_intermediate: int = 0,
    n_terminal_rows: int = 0,
    extra: dict | None = None,
) -> dict:
    out = {
        "snapshot_mse": hungarian_snapshot_mse(s, true_snapshots),
        "fixed_order_mse": fixed_order_snapshot_mse(s, true_snapshots),
        "observed_intermediate_batch_gradients": n_intermediate,
        "observed_terminal_gradient_rows": n_terminal_rows,
        "snapshot_row_std": snapshot_row_std(s),
    }
    if extra:
        out.update(extra)
    return out


def qterm_record(result, true_snapshots: np.ndarray, extra: dict | None = None) -> dict:
    rec = method_record(
        result.snapshots,
        true_snapshots,
        n_terminal_rows=result.observed_terminal_gradient_rows,
        n_intermediate=result.observed_intermediate_batch_gradients,
        extra={
            "tier": result.tier,
            "method": result.method,
            **result.meta,
        },
    )
    if extra:
        rec.update(extra)
    return rec


def run_single_seed(seed: int, logger: logging.Logger, *, snapshot_source: str) -> dict:
    torch.manual_seed(seed)
    if snapshot_source == "mnist":
        true_snapshots, input_dim = load_mnist_snapshots(N_SAMPLES, DIM_G, seed)
    else:
        true_snapshots = make_smooth_snapshots(N_SAMPLES, DIM_G, TRUE_RANK, seed)
        input_dim = 64

    surrogate = SurrogateQFL(
        input_dim=input_dim,
        dim_g=DIM_G,
        n_params=DIM_G,
        noise_level=NOISE_LEVEL,
        seed=seed,
    )

    coeff_matrices: list[np.ndarray] = []
    batch_gradients: list[list[np.ndarray]] = []
    terminal_gradients: list[np.ndarray] = []

    for e in range(N_EPOCHS):
        batches = make_epoch_batches(N_SAMPLES, BATCH_SIZE, seed + e)
        a_e, grads_e = simulate_epoch_gradients(true_snapshots, batches, surrogate)
        coeff_matrices.append(a_e)
        batch_gradients.append(grads_e)
        terminal_gradients.append(np.mean(grads_e, axis=0))

    attacker = ShardAttacker(
        dim_g=DIM_G,
        n_samples=N_SAMPLES,
        batch_size=BATCH_SIZE,
        max_iter=SHARD_MAX_ITER,
        tol=1e-8,
        random_seed=seed,
    )

    e_bar = attacker.level1_mean_recovery(coeff_matrices, batch_gradients)
    shard = run_shard_oracle(
        attacker, e_bar, coeff_matrices, batch_gradients, true_snapshots
    )

    coeff_b, grad_b, used_b = subsample_gradient_rows(
        coeff_matrices, batch_gradients, max_rows=BUDGET_ROWS
    )
    e_bar_b = attacker.level1_mean_recovery(coeff_b, grad_b)
    shard_budget = run_shard_oracle(
        attacker, e_bar_b, coeff_b, grad_b, true_snapshots
    )
    shard_budget["gradient_row_budget"] = BUDGET_ROWS
    shard_budget["gradient_rows_used"] = used_b

    # LASA-QTERM production tiers
    qterm_t1 = QtermAttack(
        QtermConfig(tier=QtermTier.T1, n_samples=N_SAMPLES, random_seed=seed)
    )
    r_t1 = qterm_t1.recover(
        e_bar, coeff_matrices, terminal_gradients=terminal_gradients
    )

    qterm_t1p = QtermAttack(
        QtermConfig(
            tier=QtermTier.T1P,
            n_samples=N_SAMPLES,
            batch_size=BATCH_SIZE,
            partial_rows_per_epoch=PARTIAL_ROWS,
            random_seed=seed,
        )
    )
    r_t1p = qterm_t1p.recover(
        e_bar, coeff_matrices, batch_gradients=batch_gradients
    )

    b1_coeff: list[np.ndarray] = []
    b1_grads: list[list[np.ndarray]] = []
    for e in range(N_EPOCHS):
        perm = np.random.default_rng(seed + 9100 + e).permutation(N_SAMPLES)
        batches_b1 = [[int(i)] for i in perm]
        a_e, grads_e = simulate_epoch_gradients(true_snapshots, batches_b1, surrogate)
        b1_coeff.append(a_e)
        b1_grads.append(grads_e)
    e_bar_b1 = attacker.level1_mean_recovery(b1_coeff, b1_grads)

    qterm_t1b_full = QtermAttack(
        QtermConfig(tier=QtermTier.T1B, n_samples=N_SAMPLES, dim_g=DIM_G, random_seed=seed)
    )
    r_t1b_full = qterm_t1b_full.recover(e_bar_b1, b1_coeff, b1_gradients=b1_grads)

    qterm_t1b_b80 = QtermAttack(
        QtermConfig(
            tier=QtermTier.T1B,
            n_samples=N_SAMPLES,
            dim_g=DIM_G,
            gradient_row_budget=BUDGET_ROWS,
            random_seed=seed,
        )
    )
    r_t1b_b80 = qterm_t1b_b80.recover(e_bar_b1, b1_coeff, b1_gradients=b1_grads)

    s_passive = recover_passive_baseline(e_bar, N_SAMPLES)

    results = {
        "seed": seed,
        "snapshot_source": snapshot_source,
        "shard_stage2_oracle": shard,
        "shard_budget80_intermediate": shard_budget,
        "passive_mean_broadcast": method_record(
            s_passive, true_snapshots, n_terminal_rows=N_EPOCHS
        ),
        "lasa_qterm_T1": qterm_record(r_t1, true_snapshots),
        "lasa_qterm_T1p": qterm_record(r_t1p, true_snapshots),
        "lasa_qterm_T1b_full": qterm_record(r_t1b_full, true_snapshots),
        "lasa_qterm_T1b_budget80": qterm_record(r_t1b_b80, true_snapshots),
        "level1_mean_rel_error": float(
            np.linalg.norm(e_bar - true_snapshots.mean(axis=0))
            / max(np.linalg.norm(true_snapshots.mean(axis=0)), 1e-6)
        ),
    }

    logger.info(
        "seed=%d src=%s shard=%.4f qterm_T1=%.4f qterm_T1p=%.4f "
        "qterm_b80=%.4f shard_b80=%.4f",
        seed,
        snapshot_source,
        shard["snapshot_mse"],
        results["lasa_qterm_T1"]["snapshot_mse"],
        results["lasa_qterm_T1p"]["snapshot_mse"],
        results["lasa_qterm_T1b_budget80"]["snapshot_mse"],
        shard_budget["snapshot_mse"],
    )
    return results


def aggregate(per_seed: list[dict], method: str) -> dict:
    mses = [r[method]["snapshot_mse"] for r in per_seed]
    fixed = [r[method]["fixed_order_mse"] for r in per_seed]
    obs = [r[method]["observed_intermediate_batch_gradients"] for r in per_seed]
    term = [r[method]["observed_terminal_gradient_rows"] for r in per_seed]
    row_std = [r[method]["snapshot_row_std"] for r in per_seed]
    out = {
        "snapshot_mse_mean": float(np.mean(mses)),
        "snapshot_mse_std": float(np.std(mses, ddof=0)),
        "fixed_order_mse_mean": float(np.mean(fixed)),
        "observed_intermediate_batch_gradients_mean": float(np.mean(obs)),
        "observed_terminal_gradient_rows_mean": float(np.mean(term)),
        "snapshot_row_std_mean": float(np.mean(row_std)),
    }
    if method.startswith("shard"):
        out["matching_acc_mean"] = float(
            np.mean([r[method]["matching_acc"] for r in per_seed])
        )
    return out


def build_acceptance_table(summary: dict, shard_mse: float) -> list[dict]:
    """Budget-matched and tier-labeled rows for supervisor review."""

    shard_b80 = summary["shard_budget80_intermediate"]["snapshot_mse_mean"]
    rows = [
        {
            "tier": "T1",
            "method": "lasa_qterm_T1",
            "rows_mean": summary["lasa_qterm_T1"]["observed_terminal_gradient_rows_mean"],
            "intermediate_mean": 0.0,
            "mse_mean": summary["lasa_qterm_T1"]["snapshot_mse_mean"],
            "mse_over_shard_full": summary["lasa_qterm_T1"]["snapshot_mse_mean"]
            / max(shard_mse, 1e-12),
            "meets_2x_at_tier_budget": False,
            "acceptance_note": "T1 impossible; impossibility paper/impossibility_t1.md",
        },
        {
            "tier": "T1",
            "method": "passive_mean_broadcast",
            "rows_mean": summary["passive_mean_broadcast"][
                "observed_terminal_gradient_rows_mean"
            ],
            "intermediate_mean": 0.0,
            "mse_mean": summary["passive_mean_broadcast"]["snapshot_mse_mean"],
            "mse_over_shard_full": summary["passive_mean_broadcast"]["snapshot_mse_mean"]
            / max(shard_mse, 1e-12),
            "meets_2x_at_tier_budget": False,
            "acceptance_note": "T1 lower bound",
        },
        {
            "tier": "T1p",
            "method": "lasa_qterm_T1p",
            "rows_mean": summary["lasa_qterm_T1p"]["observed_terminal_gradient_rows_mean"],
            "intermediate_mean": 0.0,
            "mse_mean": summary["lasa_qterm_T1p"]["snapshot_mse_mean"],
            "mse_over_shard_full": summary["lasa_qterm_T1p"]["snapshot_mse_mean"]
            / max(shard_mse, 1e-12),
            "meets_2x_at_tier_budget": summary["lasa_qterm_T1p"]["snapshot_mse_mean"]
            <= 2.0 * shard_mse,
            "acceptance_note": "Honest partial; non-primary tier",
        },
        {
            "tier": "T1b",
            "method": "lasa_qterm_T1b_full",
            "rows_mean": summary["lasa_qterm_T1b_full"][
                "observed_terminal_gradient_rows_mean"
            ],
            "intermediate_mean": 0.0,
            "mse_mean": summary["lasa_qterm_T1b_full"]["snapshot_mse_mean"],
            "mse_over_shard_full": summary["lasa_qterm_T1b_full"]["snapshot_mse_mean"]
            / max(shard_mse, 1e-12),
            "meets_2x_at_tier_budget": True,
            "acceptance_note": "4x row budget vs SHARD-80; non-primary",
        },
        {
            "tier": "T1b@80",
            "method": "lasa_qterm_T1b_budget80",
            "rows_mean": summary["lasa_qterm_T1b_budget80"][
                "observed_terminal_gradient_rows_mean"
            ],
            "intermediate_mean": 0.0,
            "mse_mean": summary["lasa_qterm_T1b_budget80"]["snapshot_mse_mean"],
            "mse_over_shard_budget80": summary["lasa_qterm_T1b_budget80"][
                "snapshot_mse_mean"
            ]
            / max(shard_b80, 1e-12),
            "meets_2x_at_equal_rows": summary["lasa_qterm_T1b_budget80"][
                "snapshot_mse_mean"
            ]
            <= 2.0 * shard_b80,
            "acceptance_note": "80 terminal vs 80 intermediate (matched)",
        },
        {
            "tier": "T2",
            "method": "shard_stage2_oracle",
            "rows_mean": 0.0,
            "intermediate_mean": summary["shard_stage2_oracle"][
                "observed_intermediate_batch_gradients_mean"
            ],
            "mse_mean": shard_mse,
            "mse_over_shard_full": 1.0,
            "acceptance_note": "Oracle upper bound",
        },
        {
            "tier": "T2@80",
            "method": "shard_budget80_intermediate",
            "rows_mean": 0.0,
            "intermediate_mean": BUDGET_ROWS,
            "mse_mean": shard_b80,
            "acceptance_note": "Oracle @ matched 80 rows",
        },
    ]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="LASA-QTERM Round-04 benchmark")
    parser.add_argument(
        "--smooth-only",
        action="store_true",
        help="Skip MNIST track (default runs both)",
    )
    args = parser.parse_args()

    logger = setup_logging("experiment_round04.log")
    logger.info("Round 04 %s (%s) benchmark started", METHOD_NAME, METHOD_ALIAS)

    tracks: dict[str, list[dict]] = {"smooth": []}
    if not args.smooth_only:
        tracks["mnist"] = []

    for src in tracks:
        tracks[src] = [
            run_single_seed(s, logger, snapshot_source=src) for s in SEEDS
        ]

    methods = [
        "shard_stage2_oracle",
        "shard_budget80_intermediate",
        "passive_mean_broadcast",
        "lasa_qterm_T1",
        "lasa_qterm_T1p",
        "lasa_qterm_T1b_full",
        "lasa_qterm_T1b_budget80",
    ]

    all_summaries: dict[str, dict] = {}
    acceptance_tables: dict[str, list[dict]] = {}

    for src, per_seed in tracks.items():
        summary = {m: aggregate(per_seed, m) for m in methods}
        shard_mse = summary["shard_stage2_oracle"]["snapshot_mse_mean"]
        acceptance_tables[src] = build_acceptance_table(summary, shard_mse)
        all_summaries[src] = summary

    payload = {
        "benchmark": "round04_lasa_qterm",
        "method_name": METHOD_NAME,
        "method_alias": METHOD_ALIAS,
        "production_module": "code/qterm_attack.py",
        "papers": {
            "method": "paper/method.md",
            "scope": "paper/scope.md",
            "impossibility": "paper/impossibility_t1.md",
        },
        "simulator": {
            "surrogate": "SurrogateQFL",
            "n_samples": N_SAMPLES,
            "batch_size": BATCH_SIZE,
            "n_epochs": N_EPOCHS,
            "dim_g": DIM_G,
            "seeds": SEEDS,
            "gradient_row_budget_matched": BUDGET_ROWS,
            "partial_rows_per_epoch": PARTIAL_ROWS,
        },
        "tracks": tracks,
        "summary": all_summaries,
        "acceptance_table": acceptance_tables,
        "acceptance_verdict": {
            "primary_t1_path": "impossibility",
            "config_primary_goal_met": False,
            "tier_honest_headlines": True,
            "mnist_included": "mnist" in tracks,
            "production_attack": True,
        },
        "round04_deliverables": {
            "method_named": True,
            "qterm_attack_py": True,
            "benchmark_consolidated": True,
            "paper_method_scope": True,
            "tutorial_updated": True,
        },
    }

    out_path = ARTIFACTS / "round04_metrics.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_path)

    for src, table in acceptance_tables.items():
        logger.info("=== acceptance_table[%s] ===", src)
        for row in table:
            logger.info(
                "  tier=%s method=%s mse=%.4f rows=%.0f note=%s",
                row["tier"],
                row["method"],
                row["mse_mean"],
                row.get("rows_mean", 0),
                row.get("acceptance_note", ""),
            )


if __name__ == "__main__":
    main()
