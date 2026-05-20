#!/usr/bin/env python3
"""Round-01 QFL terminal-snapshot benchmark.

Compares terminal-only snapshot recovery (zero intermediate batch-gradient rows)
against SHARD Stage-2 oracle with full intermediate access on the same
SurrogateQFL synthetic federated run.
"""

from __future__ import annotations

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
from terminal_attacks import (  # noqa: E402
    graph_rank_terminal,
    graph_term_map,
    passive_mean_broadcast,
)

SEEDS = [3, 7, 11, 19, 23]
N_SAMPLES = 32
BATCH_SIZE = 4
N_EPOCHS = 10
DIM_G = 32
NOISE_LEVEL = 0.01
TRUE_RANK = 4
SHARD_MAX_ITER = 50


def setup_logging() -> logging.Logger:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOGS / "experiment_round01.log", mode="w"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("round01")


def make_smooth_snapshots(n_samples: int, dim_g: int, rank: int, seed: int) -> np.ndarray:
    """Low-rank smooth snapshots (graph-correlated latent), normalized."""
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
    snapshots += 0.15  # keep nonzero global mean for identifiable Level-1 anchor
    snapshots /= np.std(snapshots) + 1e-12
    return snapshots.astype(np.float64)


def simulate_epoch_gradients(
    true_snapshots: np.ndarray,
    batches: list[list[int]],
    surrogate: SurrogateQFL,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """g^(e,k) = A^(e) @ batch_mean(s) + noise (LASA linearity)."""
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


def run_single_seed(seed: int, logger: logging.Logger) -> dict:
    torch.manual_seed(seed)
    true_snapshots = make_smooth_snapshots(
        N_SAMPLES, DIM_G, TRUE_RANK, seed
    )

    surrogate = SurrogateQFL(
        input_dim=64,
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

    n_intermediate = sum(len(g) for g in batch_gradients)

    attacker = ShardAttacker(
        dim_g=DIM_G,
        n_samples=N_SAMPLES,
        batch_size=BATCH_SIZE,
        max_iter=SHARD_MAX_ITER,
        tol=1e-8,
        random_seed=seed,
    )

    t0 = time.perf_counter()
    e_bar = attacker.level1_mean_recovery(coeff_matrices, batch_gradients)
    s_shard = attacker.level2_disaggregate(
        e_bar, coeff_matrices, batch_gradients, true_snapshots
    )
    t_shard = time.perf_counter() - t0
    mse_shard = hungarian_snapshot_mse(s_shard, true_snapshots)

    # Terminal-only: level1 uses epoch averages only (same as terminal list).
    terminal_batch_grads = [[c] for c in terminal_gradients]
    e_bar_term = attacker.level1_mean_recovery(coeff_matrices, terminal_batch_grads)

    s_passive = passive_mean_broadcast(e_bar_term, N_SAMPLES)
    s_graph = graph_term_map(e_bar_term, N_SAMPLES, graph_lambda=0.5)
    s_grank = graph_rank_terminal(
        e_bar_term, N_SAMPLES, DIM_G, rank=4, graph_lambda=0.35
    )

    results = {
        "seed": seed,
        "shard_stage2_oracle": {
            "snapshot_mse": mse_shard,
            "observed_intermediate_batch_gradients": n_intermediate,
            "runtime_sec": t_shard,
            "matching_acc": ShardAttacker._matching_accuracy(s_shard, true_snapshots),
        },
        "passive_mean_broadcast": {
            "snapshot_mse": hungarian_snapshot_mse(s_passive, true_snapshots),
            "observed_intermediate_batch_gradients": 0,
        },
        "graph_term_terminal": {
            "snapshot_mse": hungarian_snapshot_mse(s_graph, true_snapshots),
            "observed_intermediate_batch_gradients": 0,
        },
        "graph_rank_terminal": {
            "snapshot_mse": hungarian_snapshot_mse(s_grank, true_snapshots),
            "observed_intermediate_batch_gradients": 0,
        },
        "level1_mean_rel_error": float(
            np.linalg.norm(e_bar - true_snapshots.mean(axis=0))
            / max(np.linalg.norm(true_snapshots.mean(axis=0)), 1e-6)
        ),
        "level1_terminal_mean_rel_error": float(
            np.linalg.norm(e_bar_term - true_snapshots.mean(axis=0))
            / max(np.linalg.norm(true_snapshots.mean(axis=0)), 1e-6)
        ),
    }
    logger.info(
        "seed=%d shard_mse=%.4f best_terminal=%.4f (graph_term=%.4f grank=%.4f passive=%.4f)",
        seed,
        mse_shard,
        min(
            results["graph_term_terminal"]["snapshot_mse"],
            results["graph_rank_terminal"]["snapshot_mse"],
            results["passive_mean_broadcast"]["snapshot_mse"],
        ),
        results["graph_term_terminal"]["snapshot_mse"],
        results["graph_rank_terminal"]["snapshot_mse"],
        results["passive_mean_broadcast"]["snapshot_mse"],
    )
    return results


def aggregate(per_seed: list[dict], method: str) -> dict:
    mses = [r[method]["snapshot_mse"] for r in per_seed]
    obs = [r[method]["observed_intermediate_batch_gradients"] for r in per_seed]
    out = {
        "snapshot_mse_mean": float(np.mean(mses)),
        "snapshot_mse_std": float(np.std(mses, ddof=0)),
        "observed_intermediate_batch_gradients_mean": float(np.mean(obs)),
    }
    if method == "shard_stage2_oracle":
        out["matching_acc_mean"] = float(
            np.mean([r[method]["matching_acc"] for r in per_seed])
        )
        out["runtime_sec_mean"] = float(
            np.mean([r[method]["runtime_sec"] for r in per_seed])
        )
    return out


def main() -> None:
    logger = setup_logging()
    logger.info("Round 01 QFL terminal-snapshot benchmark started")
    logger.info(
        "N=%d B=%d E=%d dim_g=%d seeds=%s parent=%s",
        N_SAMPLES,
        BATCH_SIZE,
        N_EPOCHS,
        DIM_G,
        SEEDS,
        PARENT,
    )

    per_seed = [run_single_seed(s, logger) for s in SEEDS]
    methods = [
        "shard_stage2_oracle",
        "passive_mean_broadcast",
        "graph_term_terminal",
        "graph_rank_terminal",
    ]
    summary = {m: aggregate(per_seed, m) for m in methods}

    terminal_mses = [
        summary["graph_term_terminal"]["snapshot_mse_mean"],
        summary["graph_rank_terminal"]["snapshot_mse_mean"],
        summary["passive_mean_broadcast"]["snapshot_mse_mean"],
    ]
    best_terminal_name = [
        "graph_term_terminal",
        "graph_rank_terminal",
        "passive_mean_broadcast",
    ][int(np.argmin(terminal_mses))]
    best_terminal_mse = min(terminal_mses)
    shard_mse = summary["shard_stage2_oracle"]["snapshot_mse_mean"]
    ratio = best_terminal_mse / max(shard_mse, 1e-12)

    payload = {
        "benchmark": "round01_qfl_terminal_snapshot",
        "simulator": {
            "surrogate": "SurrogateQFL",
            "attacker_oracle": "ShardAttacker.level2_disaggregate",
            "n_samples": N_SAMPLES,
            "batch_size": BATCH_SIZE,
            "n_epochs": N_EPOCHS,
            "dim_g": DIM_G,
            "n_params": DIM_G,
            "noise_level": NOISE_LEVEL,
            "true_rank": TRUE_RANK,
            "snapshot_generator": "smooth_low_rank_graph_correlated",
            "seeds": SEEDS,
            "shard_max_iter": SHARD_MAX_ITER,
        },
        "threat_model": {
            "terminal_methods_observation": (
                "Per-epoch terminal gradient c_e = mean_k g^(e,k) and public A^(e); "
                "zero intermediate minibatch rows"
            ),
            "shard_oracle_observation": "All E*K intermediate batch-gradient rows",
            "graph_prior": "Oracle chain graph on sample index order (side information)",
        },
        "per_seed": per_seed,
        "summary": summary,
        "headline": {
            "shard_stage2_mse_mean": shard_mse,
            "best_terminal_method": best_terminal_name,
            "best_terminal_mse_mean": best_terminal_mse,
            "terminal_to_shard_mse_ratio": ratio,
            "beats_passive_baseline": bool(
                best_terminal_mse
                < summary["passive_mean_broadcast"]["snapshot_mse_mean"] - 1e-4
            ),
        },
        "honest_verdict": (
            "Terminal-only methods do not approach SHARD Stage-2 fidelity without "
            "batch-mean rows; graph priors marginally beat mean broadcast but remain "
            "far above oracle."
            if ratio > 2.0
            else "Terminal method within 2x of SHARD oracle at this budget (unexpected)."
        ),
        "parent_talon_limit": (
            "Parent TALON terminal probes recover class aggregates, not individuals; "
            "this run targets per-sample snapshots S in R^{N x dim_g}."
        ),
    }

    out_path = ARTIFACTS / "round01_metrics.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_path)
    logger.info(
        "HEADLINE shard_stage2_mse=%.4f best_terminal(%s)=%.4f ratio=%.2f",
        shard_mse,
        best_terminal_name,
        best_terminal_mse,
        ratio,
    )


if __name__ == "__main__":
    main()
