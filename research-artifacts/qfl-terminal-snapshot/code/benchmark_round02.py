#!/usr/bin/env python3
"""Round-02 QFL terminal-snapshot benchmark.

Fixes graph MAP anchor scaling, adds B=1 per-client terminal tier, partial
terminal trajectory leak, hardened SHARD oracle diagnostics, and optional MNIST.
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
from terminal_attacks import (  # noqa: E402
    fixed_order_snapshot_mse,
    graph_rank_terminal,
    graph_term_map,
    pad_partial_for_shard,
    partial_epoch_gradients,
    passive_mean_broadcast,
    permuted_chain_laplacian,
    snapshot_row_std,
)

SEEDS = [3, 7, 11, 19, 23]
N_SAMPLES = 32
BATCH_SIZE = 4
N_EPOCHS = 10
DIM_G = 32
NOISE_LEVEL = 0.01
TRUE_RANK = 4
SHARD_MAX_ITER = 200
PARTIAL_ROWS = [1, 2, 7]  # K-1 when B=4, N=32
GRAPH_LAMBDAS = [0.01, 0.1, 0.5, 2.0, 10.0]


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
    return logging.getLogger("round02")


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
    """MNIST-backed snapshots via SurrogateQFL encode (subset, resized)."""
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
    images = torch.stack([ds[int(i)][0] for i in idx]).flatten(1)  # (N, 64)
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
        "shard_max_iter": attacker.max_iter,
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


def run_single_seed(
    seed: int,
    logger: logging.Logger,
    *,
    snapshot_source: str = "smooth",
) -> dict:
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

    # T1: epoch-averaged terminal only
    terminal_batch_grads = [[c] for c in terminal_gradients]
    e_bar_term = attacker.level1_mean_recovery(coeff_matrices, terminal_batch_grads)
    s_passive = passive_mean_broadcast(e_bar_term, N_SAMPLES)

    graph_lambda_best = 0.5
    graph_scale_best = 0.35
    best_graph_mse = float("inf")
    graph_ablation: list[dict] = []
    for lam in GRAPH_LAMBDAS:
        for scale in (0.2, 0.35, 0.5):
            s_g = graph_term_map(
                e_bar_term,
                N_SAMPLES,
                graph_lambda=lam,
                spread_scale=scale,
            )
            mse_g = hungarian_snapshot_mse(s_g, true_snapshots)
            graph_ablation.append(
                {
                    "graph_lambda": lam,
                    "spread_scale": scale,
                    "snapshot_mse": mse_g,
                    "snapshot_row_std": snapshot_row_std(s_g),
                    "max_row_diff_from_passive": float(
                        np.max(np.abs(s_g - s_passive))
                    ),
                }
            )
            if mse_g < best_graph_mse:
                best_graph_mse = mse_g
                graph_lambda_best = lam
                graph_scale_best = scale

    s_graph = graph_term_map(
        e_bar_term,
        N_SAMPLES,
        graph_lambda=graph_lambda_best,
        spread_scale=graph_scale_best,
    )
    s_grank = graph_rank_terminal(
        e_bar_term, N_SAMPLES, DIM_G, rank=TRUE_RANK, graph_lambda=0.35
    )
    s_graph_wrong = graph_term_map(
        e_bar_term,
        N_SAMPLES,
        graph_lambda=graph_lambda_best,
        graph=permuted_chain_laplacian(N_SAMPLES, seed),
    )

    # B=1 per-client terminal: one gradient per sample per epoch (no within-epoch steps)
    b1_attacker = ShardAttacker(
        dim_g=DIM_G,
        n_samples=N_SAMPLES,
        batch_size=1,
        max_iter=SHARD_MAX_ITER,
        tol=1e-8,
        random_seed=seed,
    )
    b1_coeff: list[np.ndarray] = []
    b1_grads: list[list[np.ndarray]] = []
    for e in range(N_EPOCHS):
        perm = np.random.default_rng(seed + 9100 + e).permutation(N_SAMPLES)
        batches_b1 = [[int(i)] for i in perm]
        a_e, grads_e = simulate_epoch_gradients(true_snapshots, batches_b1, surrogate)
        b1_coeff.append(a_e)
        b1_grads.append(grads_e)
    e_bar_b1 = b1_attacker.level1_mean_recovery(b1_coeff, b1_grads)
    t0 = time.perf_counter()
    s_b1 = b1_attacker.level2_disaggregate(
        e_bar_b1, b1_coeff, b1_grads, true_snapshots
    )
    b1_runtime = time.perf_counter() - t0
    b1_terminal_rows = sum(len(g) for g in b1_grads)

    # Partial terminal trajectory (last p rows per epoch, B=4)
    k_full = N_SAMPLES // BATCH_SIZE
    partial_results: dict[str, dict] = {}
    for p in PARTIAL_ROWS:
        partial_grads = partial_epoch_gradients(batch_gradients, rows_per_epoch=p)
        padded = pad_partial_for_shard(
            coeff_matrices, partial_grads, e_bar_term, k_full
        )
        e_bar_p = attacker.level1_mean_recovery(coeff_matrices, partial_grads)
        s_p = attacker.level2_disaggregate(
            e_bar_p, coeff_matrices, padded, true_snapshots
        )
        partial_results[f"partial_terminal_last_{p}_rows"] = method_record(
            s_p,
            true_snapshots,
            n_intermediate=0,
            n_terminal_rows=sum(len(g) for g in partial_grads),
            extra={
                "rows_per_epoch": p,
                "imputed_early_rows_per_epoch": k_full - p,
            },
        )

    results = {
        "seed": seed,
        "snapshot_source": snapshot_source,
        "shard_stage2_oracle": shard,
        "passive_mean_broadcast": method_record(
            s_passive, true_snapshots, n_terminal_rows=N_EPOCHS
        ),
        "graph_term_terminal": method_record(
            s_graph,
            true_snapshots,
            n_terminal_rows=N_EPOCHS,
            extra={
                "graph_lambda": graph_lambda_best,
                "spread_scale": graph_scale_best,
            },
        ),
        "graph_rank_terminal": method_record(s_grank, true_snapshots, n_terminal_rows=N_EPOCHS),
        "graph_term_wrong_graph": method_record(
            s_graph_wrong, true_snapshots, n_terminal_rows=N_EPOCHS
        ),
        "b1_client_terminal": method_record(
            s_b1,
            true_snapshots,
            n_terminal_rows=b1_terminal_rows,
            extra={
                "runtime_sec": b1_runtime,
                "matching_acc": ShardAttacker._matching_accuracy(s_b1, true_snapshots),
            },
        ),
        "graph_lambda_ablation": graph_ablation,
        "level1_mean_rel_error": float(
            np.linalg.norm(e_bar - true_snapshots.mean(axis=0))
            / max(np.linalg.norm(true_snapshots.mean(axis=0)), 1e-6)
        ),
        "level1_terminal_mean_rel_error": float(
            np.linalg.norm(e_bar_term - true_snapshots.mean(axis=0))
            / max(np.linalg.norm(true_snapshots.mean(axis=0)), 1e-6)
        ),
    }
    results.update(partial_results)

    logger.info(
        "seed=%d src=%s shard_mse=%.4f match=%.1f%% "
        "graph_term=%.4f(row_std=%.3f) b1=%.4f partial_1=%.4f",
        seed,
        snapshot_source,
        shard["snapshot_mse"],
        100.0 * shard["matching_acc"],
        results["graph_term_terminal"]["snapshot_mse"],
        results["graph_term_terminal"]["snapshot_row_std"],
        results["b1_client_terminal"]["snapshot_mse"],
        partial_results["partial_terminal_last_1_rows"]["snapshot_mse"],
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
    if method == "shard_stage2_oracle":
        out["matching_acc_mean"] = float(
            np.mean([r[method]["matching_acc"] for r in per_seed])
        )
        out["runtime_sec_mean"] = float(
            np.mean([r[method]["runtime_sec"] for r in per_seed])
        )
        out["shard_max_iter"] = SHARD_MAX_ITER
    if method == "b1_client_terminal":
        out["matching_acc_mean"] = float(
            np.mean([r[method].get("matching_acc", 0.0) for r in per_seed])
        )
    if method == "graph_term_terminal":
        out["graph_lambda_mean"] = float(
            np.mean([r[method].get("graph_lambda", 0.5) for r in per_seed])
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mnist",
        action="store_true",
        help="Also run MNIST-backed snapshot track (adds runtime).",
    )
    args = parser.parse_args()

    logger = setup_logging("experiment_round02.log")
    logger.info("Round 02 QFL terminal-snapshot benchmark started")
    logger.info(
        "N=%d B=%d E=%d dim_g=%d seeds=%s shard_max_iter=%d",
        N_SAMPLES,
        BATCH_SIZE,
        N_EPOCHS,
        DIM_G,
        SEEDS,
        SHARD_MAX_ITER,
    )

    tracks: dict[str, list[dict]] = {"smooth": []}
    if args.mnist:
        tracks["mnist"] = []

    for src in tracks:
        tracks[src] = [run_single_seed(s, logger, snapshot_source=src) for s in SEEDS]

    methods = [
        "shard_stage2_oracle",
        "passive_mean_broadcast",
        "graph_term_terminal",
        "graph_rank_terminal",
        "graph_term_wrong_graph",
        "b1_client_terminal",
        "partial_terminal_last_1_rows",
        "partial_terminal_last_2_rows",
        "partial_terminal_last_7_rows",
    ]

    all_summaries: dict[str, dict] = {}
    headlines: dict[str, dict] = {}

    for src, per_seed in tracks.items():
        summary = {m: aggregate(per_seed, m) for m in methods}
        terminal_methods = [
            "graph_term_terminal",
            "graph_rank_terminal",
            "passive_mean_broadcast",
            "b1_client_terminal",
            "partial_terminal_last_1_rows",
            "partial_terminal_last_2_rows",
            "partial_terminal_last_7_rows",
        ]
        best_name = min(
            terminal_methods,
            key=lambda m: summary[m]["snapshot_mse_mean"],
        )
        best_mse = summary[best_name]["snapshot_mse_mean"]
        shard_mse = summary["shard_stage2_oracle"]["snapshot_mse_mean"]
        ratio_worse = best_mse / max(shard_mse, 1e-12)

        headlines[src] = {
            "shard_stage2_mse_mean": shard_mse,
            "best_terminal_method": best_name,
            "best_terminal_mse_mean": best_mse,
            "terminal_mse_over_shard_mse": ratio_worse,
            "interpretation": (
                f"Terminal best is {ratio_worse:.2f}× SHARD MSE "
                f"({'within 2×' if ratio_worse <= 2.0 else 'worse than 2×'})."
            ),
            "graph_beats_passive": bool(
                summary["graph_term_terminal"]["snapshot_mse_mean"]
                < summary["passive_mean_broadcast"]["snapshot_mse_mean"] - 1e-4
            ),
            "graph_differs_from_passive": bool(
                summary["graph_term_terminal"]["snapshot_row_std_mean"]
                > summary["passive_mean_broadcast"]["snapshot_row_std_mean"] + 1e-3
            ),
        }
        all_summaries[src] = summary

    payload = {
        "benchmark": "round02_qfl_terminal_snapshot",
        "simulator": {
            "surrogate": "SurrogateQFL",
            "attacker_oracle": "ShardAttacker.level2_disaggregate",
            "n_samples": N_SAMPLES,
            "batch_size": BATCH_SIZE,
            "n_epochs": N_EPOCHS,
            "dim_g": DIM_G,
            "noise_level": NOISE_LEVEL,
            "true_rank": TRUE_RANK,
            "seeds": SEEDS,
            "shard_max_iter": SHARD_MAX_ITER,
            "graph_lambdas_swept": GRAPH_LAMBDAS,
            "partial_rows_per_epoch": PARTIAL_ROWS,
        },
        "threat_model": {
            "T1_epoch_terminal": "One epoch-averaged gradient per round; 0 intermediate rows",
            "T1b_b1_client_terminal": (
                "B=1: one terminal gradient per sample per epoch; "
                "0 within-epoch intermediate rows"
            ),
            "partial_terminal": "Last p minibatch rows per epoch only (no earlier rows)",
            "shard_oracle": "All E×K intermediate batch-gradient rows",
        },
        "tracks": tracks,
        "summary": all_summaries,
        "headline": headlines,
        "round01_fixes": {
            "graph_anchor_weight_removed": True,
            "graph_lambda_ablation_per_seed": True,
            "shard_iterations_increased": f"{50} -> {SHARD_MAX_ITER}",
        },
    }

    out_path = ARTIFACTS / "round02_metrics.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_path)

    smooth_h = headlines["smooth"]
    smooth_s = all_summaries["smooth"]
    logger.info(
        "HEADLINE[smooth] shard_mse=%.4f best=%s mse=%.4f ratio=%.2f "
        "shard_match=%.1f%% graph_row_std=%.4f passive_row_std=%.4f",
        smooth_h["shard_stage2_mse_mean"],
        smooth_h["best_terminal_method"],
        smooth_h["best_terminal_mse_mean"],
        smooth_h["terminal_mse_over_shard_mse"],
        100.0 * smooth_s["shard_stage2_oracle"]["matching_acc_mean"],
        smooth_s["graph_term_terminal"]["snapshot_row_std_mean"],
        smooth_s["passive_mean_broadcast"]["snapshot_row_std_mean"],
    )


if __name__ == "__main__":
    main()
