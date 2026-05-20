#!/usr/bin/env python3
"""Round-04 Stage-2 benchmark under missing batch-gradient observations.

This benchmark isolates SHARD's Stage-2 identifiability bottleneck in an
oracle-incidence synthetic simulator:

    observed batch means = H_obs @ S_true + noise

where H is the batch-incidence matrix induced by reshuffling, S_true are
individual snapshots, and only a subset of batch rows is observed.  The SHARD
baseline is the least-squares update used when assignments are fixed.  The
proposed method, GARD, replaces the full-rank incidence requirement with a
Gaussian graph-prior posterior:

    p(S | observations) proportional to likelihood times
    exp(-lambda * Tr(S^T L S)).

The simulator is intentionally not a full unknown-assignment attack.  It is a
direct test of the information bottleneck: if fixed-assignment SHARD LS fails
when H_obs is rank deficient, the full alternating SHARD procedure cannot be
expected to solve the same missing-row linear system without additional prior
structure.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RUN_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = RUN_ROOT / "artifacts"
LOGS = RUN_ROOT / "logs"


@dataclass(frozen=True)
class SimConfig:
    n_samples: int = 48
    batch_size: int = 4
    dim_g: int = 32
    true_rank: int = 4
    n_epochs: int = 5
    noise_std: float = 0.01
    mean_anchor_weight: float = 100.0
    graph_lambda: float = 0.35


def setup_logging() -> logging.Logger:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOGS / "experiment_round04.log", mode="w"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("round04")


def make_low_rank_snapshots(cfg: SimConfig, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Smooth latent coordinates mimic ordered or graph-correlated client data.
    # The attacker may know such a graph from acquisition order, temporal
    # locality, or public metadata, without observing private snapshots.
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


def make_incidence(cfg: SimConfig, seed: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
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


def subsample_rows(
    h_full: np.ndarray,
    metadata: list[tuple[int, int]],
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray]:
    rng = np.random.default_rng(seed + 20_000)
    n_rows = h_full.shape[0]
    n_keep = max(1, int(round(fraction * n_rows)))
    keep = np.sort(rng.choice(n_rows, size=n_keep, replace=False))
    return h_full[keep], [metadata[i] for i in keep], keep


def shard_style_ls(
    h_obs: np.ndarray,
    m_obs: np.ndarray,
    mean_snapshot: np.ndarray,
    cfg: SimConfig,
) -> np.ndarray:
    """SHARD update with fixed assignments: least squares plus mean anchor."""
    anchor_row = np.ones((1, cfg.n_samples), dtype=np.float64) / cfg.n_samples
    weight = np.sqrt(cfg.mean_anchor_weight)
    h_aug = np.vstack([h_obs, weight * anchor_row])
    m_aug = np.vstack([m_obs, weight * mean_snapshot[None, :]])
    return np.linalg.lstsq(h_aug, m_aug, rcond=None)[0]


def chain_laplacian(n_samples: int) -> np.ndarray:
    lap = np.zeros((n_samples, n_samples), dtype=np.float64)
    for i in range(n_samples - 1):
        lap[i, i] += 1.0
        lap[i + 1, i + 1] += 1.0
        lap[i, i + 1] -= 1.0
        lap[i + 1, i] -= 1.0
    return lap


def gard_recover(
    h_obs: np.ndarray,
    m_obs: np.ndarray,
    mean_snapshot: np.ndarray,
    cfg: SimConfig,
) -> np.ndarray:
    """Graph-Augmented Recovery for Disaggregation (posterior mean / MAP)."""
    anchor = np.ones((cfg.n_samples, 1), dtype=np.float64) / cfg.n_samples
    lap = chain_laplacian(cfg.n_samples)
    lhs = (
        h_obs.T @ h_obs
        + cfg.graph_lambda * lap
        + cfg.mean_anchor_weight * (anchor @ anchor.T)
    )
    rhs = h_obs.T @ m_obs + cfg.mean_anchor_weight * anchor @ mean_snapshot[None, :]
    return np.linalg.solve(lhs + 1e-8 * np.eye(cfg.n_samples), rhs)


def metrics(s_hat: np.ndarray, s_true: np.ndarray, h_obs: np.ndarray, m_obs: np.ndarray) -> dict:
    mse = float(np.mean((s_hat - s_true) ** 2))
    rel_fro = float(
        np.linalg.norm(s_hat - s_true, "fro") / (np.linalg.norm(s_true, "fro") + 1e-12)
    )
    obs_mse = float(np.mean((h_obs @ s_hat - m_obs) ** 2))
    return {
        "snapshot_mse": mse,
        "relative_fro_error": rel_fro,
        "observed_batch_mse": obs_mse,
    }


def run_one(seed: int, fraction: float, cfg: SimConfig) -> dict:
    s_true = make_low_rank_snapshots(cfg, seed)
    mean_snapshot = s_true.mean(axis=0)
    h_full, metadata = make_incidence(cfg, seed)
    h_obs, kept_meta, keep = subsample_rows(h_full, metadata, fraction, seed)
    rng = np.random.default_rng(seed + 50_000)
    m_clean = h_obs @ s_true
    m_obs = m_clean + cfg.noise_std * rng.normal(size=m_clean.shape)

    s_ls = shard_style_ls(h_obs, m_obs, mean_snapshot, cfg)
    s_gard = gard_recover(h_obs, m_obs, mean_snapshot, cfg)

    full_rows = h_full.shape[0]
    result = {
        "seed": seed,
        "observation_fraction": fraction,
        "observed_batch_gradients": int(h_obs.shape[0]),
        "full_batch_gradients": int(full_rows),
        "kept_rows": keep.tolist(),
        "incidence_rank_observed": int(np.linalg.matrix_rank(h_obs)),
        "incidence_rank_with_mean_anchor": int(
            np.linalg.matrix_rank(
                np.vstack([h_obs, np.ones((1, cfg.n_samples)) / cfg.n_samples])
            )
        ),
        "shard_style_ls": metrics(s_ls, s_true, h_obs, m_obs),
        "gard_graph_map": metrics(s_gard, s_true, h_obs, m_obs),
    }
    result["improvement_snapshot_mse_x"] = (
        result["shard_style_ls"]["snapshot_mse"]
        / max(result["gard_graph_map"]["snapshot_mse"], 1e-12)
    )
    result["rank_deficient_vs_n"] = result["incidence_rank_with_mean_anchor"] < cfg.n_samples
    result["observed_epochs"] = sorted({int(e) for e, _ in kept_meta})
    return result


def aggregate(per_seed: list[dict]) -> dict:
    out: dict[str, float] = {}
    for method in ["shard_style_ls", "gard_graph_map"]:
        for metric_name in ["snapshot_mse", "relative_fro_error", "observed_batch_mse"]:
            vals = np.array([run[method][metric_name] for run in per_seed], dtype=np.float64)
            out[f"{method}_{metric_name}_mean"] = float(vals.mean())
            out[f"{method}_{metric_name}_std"] = float(vals.std(ddof=0))
    gains = np.array([run["improvement_snapshot_mse_x"] for run in per_seed], dtype=np.float64)
    ranks = np.array([run["incidence_rank_with_mean_anchor"] for run in per_seed], dtype=np.float64)
    out["improvement_snapshot_mse_x_mean"] = float(gains.mean())
    out["improvement_snapshot_mse_x_std"] = float(gains.std(ddof=0))
    out["incidence_rank_with_mean_anchor_mean"] = float(ranks.mean())
    out["rank_deficient_rate"] = float(np.mean([run["rank_deficient_vs_n"] for run in per_seed]))
    return out


def plot_results(results: dict) -> None:
    fractions = [row["observation_fraction"] for row in results["aggregate_by_fraction"]]
    shard_mse = [
        row["shard_style_ls_snapshot_mse_mean"]
        for row in results["aggregate_by_fraction"]
    ]
    shard_std = [
        row["shard_style_ls_snapshot_mse_std"]
        for row in results["aggregate_by_fraction"]
    ]
    gard_mse = [
        row["gard_graph_map_snapshot_mse_mean"]
        for row in results["aggregate_by_fraction"]
    ]
    gard_std = [
        row["gard_graph_map_snapshot_mse_std"]
        for row in results["aggregate_by_fraction"]
    ]

    plt.figure(figsize=(7.0, 4.5))
    plt.errorbar(fractions, shard_mse, yerr=shard_std, marker="o", label="SHARD-style LS")
    plt.errorbar(fractions, gard_mse, yerr=gard_std, marker="s", label="GARD graph MAP")
    plt.yscale("log")
    plt.gca().invert_xaxis()
    plt.xlabel("Observed batch-gradient fraction")
    plt.ylabel("Snapshot MSE (log scale)")
    plt.title("Round 04: Stage-2 Recovery Under Missing Batch Gradients")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "round04_stage2_missing_observations.png", dpi=180)
    plt.close()


def main() -> None:
    logger = setup_logging()
    cfg = SimConfig()
    seeds = [3, 7, 11, 19, 23]
    fractions = [1.0, 0.6, 0.4, 0.25, 0.15]
    logger.info("Round 04 config: %s", cfg)
    logger.info("Seeds=%s fractions=%s", seeds, fractions)

    t0 = time.perf_counter()
    per_fraction: list[dict] = []
    all_runs: list[dict] = []
    for fraction in fractions:
        logger.info("=== observation_fraction=%.2f ===", fraction)
        runs = []
        for seed in seeds:
            run = run_one(seed, fraction, cfg)
            runs.append(run)
            all_runs.append(run)
            logger.info(
                "seed=%d rows=%d/%d rank=%d shard_mse=%.4g gard_mse=%.4g gain=%.2fx",
                seed,
                run["observed_batch_gradients"],
                run["full_batch_gradients"],
                run["incidence_rank_with_mean_anchor"],
                run["shard_style_ls"]["snapshot_mse"],
                run["gard_graph_map"]["snapshot_mse"],
                run["improvement_snapshot_mse_x"],
            )
        agg = aggregate(runs)
        agg["observation_fraction"] = fraction
        agg["observed_batch_gradients_mean"] = float(
            np.mean([r["observed_batch_gradients"] for r in runs])
        )
        per_fraction.append(agg)

    results = {
        "benchmark": "round04_oracle_incidence_stage2_missing_observations",
        "method_selected": "GARD: Graph-Augmented Recovery for Disaggregation",
        "notes": [
            "Synthetic oracle-incidence Stage-2 simulator; assignments are fixed/known.",
            "SHARD baseline is the fixed-assignment least-squares update, favorable to SHARD.",
            "GARD assumes a public/side-information graph and preserves the mean anchor.",
        ],
        "config": cfg.__dict__,
        "seeds": seeds,
        "fractions": fractions,
        "aggregate_by_fraction": per_fraction,
        "per_seed": all_runs,
        "runtime_sec": time.perf_counter() - t0,
    }
    (ARTIFACTS / "round04_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    plot_results(results)
    logger.info("Wrote %s", ARTIFACTS / "round04_metrics.json")
    logger.info("Wrote %s", ARTIFACTS / "round04_stage2_missing_observations.png")
    logger.info("Runtime %.2fs", results["runtime_sec"])


if __name__ == "__main__":
    main()
