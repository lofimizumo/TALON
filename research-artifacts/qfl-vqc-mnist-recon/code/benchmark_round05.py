#!/usr/bin/env python3
"""Round-05: 28×28 MNIST reconstruction — dim_g sweep + quality JOLI (no LAPIN on 784).

Experiment-only round (no supervisor review):
  - dim_g ∈ {100, 160, 256}
  - Snapshots: lasa_qterm_T1p, gard_sparse_oracle, shard_oracle
  - L3: JOLI profile ``mnist28_quality`` (adam=400, lbfgs=200, n_batch cap 128, no LAPIN)
  - Sequential seeds, per-image L3 timing logs
  - Grid: ``artifacts/round05_recon_grid.png``
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

RUN_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = RUN_ROOT.parent.parent
sys.path.insert(0, str(RUN_ROOT / "code"))
from _paths import ARTIFACTS, LOGS  # noqa: E402

os.chdir(WORKSPACE)

from joli_invert import joli_invert_single  # noqa: E402
from qterm_attack import QtermAttack, QtermConfig, QtermTier  # noqa: E402
from shard_sim.data_loader import FederatedDataLoader  # noqa: E402
from shard_sim.metrics import (  # noqa: E402
    compute_matching_accuracy,
    compute_reconstruction_mse,
)
from shard_sim.surrogate_model import SurrogateQFL  # noqa: E402

from l3_budget import (  # noqa: E402
    estimate_joli_seconds,
    joli_profile,
    shard_n_batch_capped,
)

# Round-04 helpers (GARD-SPARSE oracle, SHARD R02, metrics)
_r04_spec = importlib.util.spec_from_file_location(
    "mnist_benchmark_round04",
    RUN_ROOT / "code" / "benchmark_round04.py",
)
_r04 = importlib.util.module_from_spec(_r04_spec)
assert _r04_spec.loader is not None
_r04_spec.loader.exec_module(_r04)

ShardAttackerR02 = _r04.ShardAttackerR02
build_oracle_incidence = _r04.build_oracle_incidence
hungarian_snapshot_mse = _r04.hungarian_snapshot_mse
psnr_from_mse = _r04.psnr_from_mse
simulate_b1_epoch_gradients = _r04.simulate_b1_epoch_gradients
recover_gard_sparse_oracle = _r04.recover_gard_sparse_oracle
load_r3_tv_lbfgs = _r04.load_r3_tv_lbfgs
meets_targets = _r04.meets_targets
plot_recon_grid = _r04.plot_recon_grid

# --- Round-05 knobs ---
RESEARCH_ROUND = 5
RESIZE = 28
SEEDS = [3, 7, 11]
# Full sweep (100,160,256) ≈ 21h CPU; default primary dim from Round-02 sweet spot.
DIM_G_SWEEP = (160,) if os.environ.get("QFL_FULL_DIM_SWEEP") != "1" else (100, 160, 256)
N_SAMPLES = 32
BATCH_SIZE = 4
N_EPOCHS = 10
NOISE_LEVEL = 0.01
PARTIAL_ROWS = 8
PARTIAL_MAX_ITER = 150
PARTIAL_GRAPH_LAMBDA = 0.5
SHARD_MAX_ITER = 200
SHARD_TOL = 1e-9
SHARD_GRAPH_LAMBDA = 0.5
SHARD_GRAPH_SPREAD = 0.35

TARGET_INPUT_MSE = 0.05
TARGET_PSNR_DB = 18.0
PASS_MIN_SEEDS = 2

SNAPSHOT_PATHS = (
    "lasa_qterm_T1p",
    "gard_sparse_oracle",
    "shard_oracle",
)
L3_INVERTER = "joli_quality"
GRID_PATHS = SNAPSHOT_PATHS
GRID_SEED = 7
GRID_DIM_G = 160


def setup_logging() -> logging.Logger:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    logging.getLogger("shard_sim.attacker").setLevel(logging.WARNING)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(
                LOGS / "experiment_round05.log", mode="w", encoding="utf-8"
            ),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return logging.getLogger("mnist_recon_r05")


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def estimate_benchmark_seconds() -> tuple[float, dict]:
    """Total JOLI wall-clock estimate from l3_budget empirical fit."""
    d = RESIZE * RESIZE
    profile = joli_profile(d, research_round=RESEARCH_ROUND)
    by_dim_g: dict[str, float] = {}
    total = 0.0
    n_blocks = len(SEEDS) * len(SNAPSHOT_PATHS)
    for dim_g in DIM_G_SWEEP:
        per_block = estimate_joli_seconds(
            d,
            dim_g,
            N_SAMPLES,
            profile.adam_steps,
            profile.n_batch_cap,
        )
        dim_total = per_block * n_blocks
        by_dim_g[str(dim_g)] = {
            "estimate_joli_sec_per_seed_path": per_block,
            "estimate_joli_sec_dim_g": dim_total,
        }
        total += dim_total
    return total, {
        "profile": profile.name,
        "adam_steps": profile.adam_steps,
        "lbfgs_iter": profile.lbfgs_iter,
        "n_batch_cap": profile.n_batch_cap,
        "include_lapin": profile.include_lapin,
        "n_images_per_l3": N_SAMPLES,
        "n_seed_path_blocks_per_dim_g": n_blocks,
        "by_dim_g": by_dim_g,
        "estimate_joli_sec_total": total,
    }


def recover_snapshots(
    path: str,
    *,
    attacker: ShardAttackerR02,
    e_bar: np.ndarray,
    coeff_matrices: list[np.ndarray],
    batch_gradients: list[list[np.ndarray]],
    true_snapshots: np.ndarray,
    h_full: np.ndarray,
    seed: int,
    dim_g: int,
) -> tuple[np.ndarray, dict]:
    if path == "gard_sparse_oracle":
        return recover_gard_sparse_oracle(
            h_full=h_full,
            coeff_matrices=coeff_matrices,
            batch_gradients=batch_gradients,
            e_bar=e_bar,
            n_samples=N_SAMPLES,
            dim_g=dim_g,
            seed=seed,
        )

    if path == "lasa_qterm_T1p":
        q = QtermAttack(
            QtermConfig(
                tier=QtermTier.T1P,
                n_samples=N_SAMPLES,
                batch_size=BATCH_SIZE,
                dim_g=dim_g,
                partial_rows_per_epoch=PARTIAL_ROWS,
                partial_graph_lambda=PARTIAL_GRAPH_LAMBDA,
                partial_max_iter=PARTIAL_MAX_ITER,
                random_seed=seed,
            )
        )
        r = q.recover(e_bar, coeff_matrices, batch_gradients=batch_gradients)
        return r.snapshots, {
            "tier": r.tier,
            "method": r.method,
            "observed_terminal_gradient_rows": r.observed_terminal_gradient_rows,
            "observed_intermediate_batch_gradients": r.observed_intermediate_batch_gradients,
            **r.meta,
        }

    if path == "shard_oracle":
        s = attacker.level2_disaggregate(
            e_bar, coeff_matrices, batch_gradients, true_snapshots
        )
        return s, {
            "observed_intermediate_batch_gradients": sum(len(g) for g in batch_gradients),
            "observed_terminal_gradient_rows": 0,
            "l2_init": "graph_term_map",
            "shard_max_iter": attacker.max_iter,
        }

    raise ValueError(path)


def run_l3(
    snapshots: np.ndarray,
    surrogate: SurrogateQFL,
    device: str,
    *,
    profile,
    tv_lbfgs: float,
    logger: logging.Logger,
) -> np.ndarray:
    d = int(surrogate.W_enc.shape[1])
    dim_g = int(surrogate.dim_g)
    n_batch = shard_n_batch_capped(d, dim_g, profile.n_batch_cap)
    n = snapshots.shape[0]
    out = np.empty((n, d), dtype=np.float64)
    for i in range(n):
        t_img = time.perf_counter()
        out[i], _ = joli_invert_single(
            snapshots[i],
            surrogate,
            n_batch=n_batch,
            adam_steps=profile.adam_steps,
            lbfgs_iter=profile.lbfgs_iter,
            tv_adam=0.0,
            tv_lbfgs=tv_lbfgs,
            seed=42 + i,
            device=device,
        )
        if profile.log_every_image:
            logger.info(
                "  %s image %d/%d n_batch=%d %.1fs",
                L3_INVERTER,
                i + 1,
                n,
                n_batch,
                time.perf_counter() - t_img,
            )
    return out


def evaluate_reconstruction(
    x_rec: np.ndarray,
    x_true: np.ndarray,
    s_rec: np.ndarray,
    s_true: np.ndarray,
) -> dict:
    _, snap_assign = hungarian_snapshot_mse(s_rec, s_true)
    snap_mse = float(np.mean((s_rec - s_true[snap_assign]) ** 2))
    _, img_assign = compute_matching_accuracy(x_rec, x_true)
    input_mse = compute_reconstruction_mse(x_rec, x_true, img_assign)
    return {
        "snapshot_mse": snap_mse,
        "input_mse": input_mse,
        "input_psnr_db": psnr_from_mse(input_mse),
        "image_matching_acc": float(compute_matching_accuracy(x_rec, x_true)[0]),
    }


def run_single_seed(
    seed: int,
    dim_g: int,
    logger: logging.Logger,
    device: str,
    *,
    profile,
    tv_lbfgs: float,
    capture_grid: bool,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    loader = FederatedDataLoader(
        dataset="mnist",
        n_samples=N_SAMPLES,
        batch_size=BATCH_SIZE,
        seed=seed,
        resize=RESIZE,
    )
    images = loader.images.float()
    d = images.shape[1]
    image_shape = loader.image_shape

    surrogate = SurrogateQFL(
        input_dim=d,
        dim_g=dim_g,
        n_params=dim_g,
        noise_level=NOISE_LEVEL,
        seed=seed,
    )

    coeff_matrices: list[np.ndarray] = []
    batch_gradients: list[list[np.ndarray]] = []
    epoch_batches_list: list[list[list[int]]] = []

    for e in range(N_EPOCHS):
        epoch_batches = loader.get_epoch_batches()
        epoch_batches_list.append(epoch_batches)
        a_e, grads_e = surrogate.compute_batch_gradients(images, epoch_batches)
        coeff_matrices.append(a_e)
        batch_gradients.append(grads_e)
        simulate_b1_epoch_gradients(images, surrogate, seed, e)

    h_full, _ = build_oracle_incidence(epoch_batches_list, N_SAMPLES)

    with torch.no_grad():
        true_snapshots = surrogate.encode(images).numpy().astype(np.float64)
    x_true = images.numpy().astype(np.float64)

    attacker = ShardAttackerR02(
        dim_g=dim_g,
        n_samples=N_SAMPLES,
        batch_size=BATCH_SIZE,
        max_iter=SHARD_MAX_ITER,
        tol=SHARD_TOL,
        random_seed=seed,
        graph_lambda=SHARD_GRAPH_LAMBDA,
        spread_scale=SHARD_GRAPH_SPREAD,
    )
    e_bar = attacker.level1_mean_recovery(coeff_matrices, batch_gradients)

    est_block = estimate_joli_seconds(
        d, dim_g, N_SAMPLES, profile.adam_steps, profile.n_batch_cap
    )
    est_seed = est_block * len(SNAPSHOT_PATHS)
    logger.info(
        "seed=%d dim_g=%d snapshot+L3 est_joli~%.0fs (%.0fs/path × %d paths)",
        seed,
        dim_g,
        est_seed,
        est_block,
        len(SNAPSHOT_PATHS),
    )

    out: dict = {
        "seed": seed,
        "dim_g": dim_g,
        "resize": RESIZE,
        "d": d,
        "image_shape": list(image_shape),
        "estimate_joli_sec": est_seed,
        "paths": {},
        "grid_panels": [] if capture_grid else None,
    }

    for path in SNAPSHOT_PATHS:
        t_snap = time.perf_counter()
        s_rec, snap_meta = recover_snapshots(
            path,
            attacker=attacker,
            e_bar=e_bar,
            coeff_matrices=coeff_matrices,
            batch_gradients=batch_gradients,
            true_snapshots=true_snapshots,
            h_full=h_full,
            seed=seed,
            dim_g=dim_g,
        )
        snap_time = time.perf_counter() - t_snap
        snap_metrics = {
            "snapshot_mse": hungarian_snapshot_mse(s_rec, true_snapshots)[0],
            **snap_meta,
            "recovery_runtime_sec": snap_time,
        }
        path_out = {"snapshot_recovery": snap_metrics, "l3": {}}

        logger.info(
            "L3 start seed=%d dim_g=%d path=%s inv=%s (N=%d d=%d profile=%s)",
            seed,
            dim_g,
            path,
            L3_INVERTER,
            s_rec.shape[0],
            d,
            profile.name,
        )
        t0 = time.perf_counter()
        try:
            x_rec = run_l3(
                s_rec,
                surrogate,
                device,
                profile=profile,
                tv_lbfgs=tv_lbfgs,
                logger=logger,
            )
            l3_time = time.perf_counter() - t0
            metrics = evaluate_reconstruction(x_rec, x_true, s_rec, true_snapshots)
            metrics["l3_runtime_sec"] = l3_time
            metrics["tv_lbfgs"] = tv_lbfgs
            metrics["adam_steps"] = profile.adam_steps
            metrics["lbfgs_iter"] = profile.lbfgs_iter
            metrics["n_batch"] = shard_n_batch_capped(d, dim_g, profile.n_batch_cap)
            metrics["targets"] = meets_targets(metrics)
            path_out["l3"][L3_INVERTER] = metrics
            logger.info(
                "seed=%d dim_g=%d %s %s snap=%.4f mse=%.4f psnr=%.2f pass=%s (L3 %.0fs)",
                seed,
                dim_g,
                path,
                L3_INVERTER,
                metrics["snapshot_mse"],
                metrics["input_mse"],
                metrics["input_psnr_db"],
                metrics["targets"]["both_ok"],
                l3_time,
            )
            if capture_grid and path in GRID_PATHS:
                out["grid_panels"].append(
                    {
                        "label": path,
                        "x_true": x_true[:6].tolist(),
                        "x_rec": x_rec[:6].tolist(),
                        "image_shape": list(image_shape),
                    }
                )
        except Exception as exc:
            path_out["l3"][L3_INVERTER] = {"error": str(exc)}
            logger.exception(
                "L3 failed seed=%d dim_g=%d path=%s", seed, dim_g, path
            )

        out["paths"][path] = path_out

    return out


def aggregate_method(per_seed: list[dict], path: str) -> dict:
    rows = []
    for r in per_seed:
        l3 = r["paths"][path]["l3"].get(L3_INVERTER, {})
        if "error" not in l3 and "input_mse" in l3:
            rows.append(l3)
    if not rows:
        return {"n_ok": 0}
    keys = ("snapshot_mse", "input_mse", "input_psnr_db", "l3_runtime_sec")
    out: dict = {"n_ok": len(rows)}
    for k in keys:
        vals = [x[k] for x in rows]
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals, ddof=0))
    pass_both = sum(1 for x in rows if x.get("targets", {}).get("both_ok"))
    n = len(rows)
    out["pass_count_both"] = pass_both
    out["pass_rate_both"] = pass_both / n
    out["passes_acceptance"] = pass_both >= PASS_MIN_SEEDS
    return out


def main() -> None:
    logger = setup_logging()
    device = pick_device()
    tv_lbfgs = load_r3_tv_lbfgs()
    d = RESIZE * RESIZE
    profile = joli_profile(d, research_round=RESEARCH_ROUND)
    est_total, est_detail = estimate_benchmark_seconds()

    logger.info(
        "Round-05 MNIST 28×28 recon seeds=%s dim_g=%s device=%s",
        SEEDS,
        DIM_G_SWEEP,
        device,
    )
    logger.info(
        "L3 profile=%s adam=%d lbfgs=%d n_batch_cap=%d lapin=%s "
        "parallel_workers=%d est_total_joli_sec~%.0f (~%.1f min)",
        profile.name,
        profile.adam_steps,
        profile.lbfgs_iter,
        profile.n_batch_cap,
        profile.include_lapin,
        profile.parallel_seed_workers,
        est_total,
        est_total / 60.0,
    )
    logger.info("Runtime breakdown: %s", json.dumps(est_detail, indent=2))
    logger.info("Root-cause doc: paper/root_cause_l3_stall.md")

    per_seed_by_dim: dict[int, list[dict]] = {}
    wall_start = time.perf_counter()

    for dim_g in DIM_G_SWEEP:
        logger.info("=== dim_g=%d (sequential seeds) ===", dim_g)
        per_seed: list[dict] = []
        for seed in SEEDS:
            per_seed.append(
                run_single_seed(
                    seed,
                    dim_g,
                    logger,
                    device,
                    profile=profile,
                    tv_lbfgs=tv_lbfgs,
                    capture_grid=(seed == GRID_SEED and dim_g == GRID_DIM_G),
                )
            )
        per_seed_by_dim[dim_g] = per_seed

    wall_sec = time.perf_counter() - wall_start

    summary_by_dim: dict[str, dict] = {}
    accepting_all: list[dict] = []
    for dim_g, per_seed in per_seed_by_dim.items():
        summary: dict = {}
        for path in SNAPSHOT_PATHS:
            key = f"{path}__{L3_INVERTER}"
            agg = aggregate_method(per_seed, path)
            summary[key] = agg
            if agg.get("passes_acceptance"):
                accepting_all.append(
                    {
                        "dim_g": dim_g,
                        "path": path,
                        "inverter": L3_INVERTER,
                        "pass_count": agg["pass_count_both"],
                        "input_mse_mean": agg.get("input_mse_mean"),
                        "input_psnr_db_mean": agg.get("input_psnr_db_mean"),
                    }
                )
        summary_by_dim[str(dim_g)] = summary

    grid_path = ARTIFACTS / "round05_recon_grid.png"
    rec = per_seed_by_dim.get(GRID_DIM_G)
    if rec:
        row = next((r for r in rec if r["seed"] == GRID_SEED), None)
        if row and row.get("grid_panels"):
            panels: list[tuple[str, np.ndarray, np.ndarray, tuple]] = []
            for p in row["grid_panels"]:
                panels.append(
                    (
                        p["label"],
                        np.asarray(p["x_true"]),
                        np.asarray(p["x_rec"]),
                        tuple(p["image_shape"]),
                    )
                )
            plot_recon_grid(
                panels,
                grid_path,
                title=(
                    f"Round-05 MNIST 28×28 {profile.name} "
                    f"(dim_g={GRID_DIM_G}, seed={GRID_SEED})"
                ),
            )
            logger.info("Wrote grid %s", grid_path)

    per_seed_json = []
    for dim_g, per_seed in per_seed_by_dim.items():
        for r in per_seed:
            rj = {k: v for k, v in r.items() if k != "grid_panels"}
            per_seed_json.append(rj)

    targets_met = len(accepting_all) > 0
    payload = {
        "benchmark": "round05_qfl_vqc_mnist_recon",
        "supervisor_review": False,
        "method_name_stack": (
            "SurrogateQFL + T1p/GARD-SPARSE/SHARD snapshots + JOLI quality L3"
        ),
        "round05_focus": {
            "dim_g_sweep": list(DIM_G_SWEEP),
            "snapshot_paths": list(SNAPSHOT_PATHS),
            "sequential_seeds": True,
            "per_image_l3_logs": profile.log_every_image,
        },
        "l3_profile": {
            "name": profile.name,
            "adam_steps": profile.adam_steps,
            "lbfgs_iter": profile.lbfgs_iter,
            "n_batch_cap": profile.n_batch_cap,
            "include_lapin": profile.include_lapin,
            "lapin_only_if_d_le": profile.lapin_only_if_d_le,
            "tv_lbfgs": tv_lbfgs,
            "inverter": L3_INVERTER,
        },
        "expected_runtime": {
            **est_detail,
            "estimate_joli_sec_total": est_total,
            "estimate_minutes": est_total / 60.0,
            "actual_wall_sec": wall_sec,
            "root_cause_doc": "paper/root_cause_l3_stall.md",
        },
        "simulator": {
            "n_samples": N_SAMPLES,
            "batch_size": BATCH_SIZE,
            "n_epochs": N_EPOCHS,
            "partial_rows_t1p": PARTIAL_ROWS,
            "noise_level": NOISE_LEVEL,
            "seeds": SEEDS,
            "device": device,
        },
        "targets": {
            "input_mse_max": TARGET_INPUT_MSE,
            "psnr_min_db": TARGET_PSNR_DB,
            "pass_min_seeds_of_3": PASS_MIN_SEEDS,
        },
        "summary_by_dim_g": summary_by_dim,
        "per_seed": per_seed_json,
        "accepting_configs": accepting_all,
        "targets_met_any_config": targets_met,
        "grid_png": str(grid_path) if grid_path.is_file() else None,
        "honest_verdict": (
            f"Acceptance: {len(accepting_all)} config(s) with MSE≤{TARGET_INPUT_MSE} "
            f"and PSNR≥{TARGET_PSNR_DB} on ≥{PASS_MIN_SEEDS}/3 seeds."
            if targets_met
            else (
                f"No config reached both targets on ≥{PASS_MIN_SEEDS}/3 seeds; "
                "see summary_by_dim_g."
            )
        ),
    }

    out_json = ARTIFACTS / "round05_metrics.json"
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Wrote %s targets_met=%s wall=%.0fs est_joli~%.0fs (~%.1f min)",
        out_json,
        targets_met,
        wall_sec,
        est_total,
        est_total / 60.0,
    )
    print(
        f"EXPECTED_RUNTIME_SEC={est_total:.0f} "
        f"EXPECTED_RUNTIME_MIN={est_total / 60.0:.1f} "
        f"ACTUAL_WALL_SEC={wall_sec:.0f}"
    )


if __name__ == "__main__":
    main()
