#!/usr/bin/env python3
"""Round-04: R3 JOLI L3 + improved snapshot paths (GARD-SPARSE oracle, T1p+graph prior).

Combines parent Round-03 JOLI operating point (tv_lbfgs) with:
  - GARD-SPARSE: oracle assignment + partial terminal rows + chain graph MAP
  - T1p: 8 partial rows/epoch + graph prior polish on snapshots before L3
  - Optional 14×14 MNIST if 28×28 misses acceptance on weak/T1p paths

Target: ANY config with input_mse <= 0.05 AND psnr >= 18 on >= 2/3 seeds.
"""

from __future__ import annotations

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
from scipy.optimize import linear_sum_assignment

RUN_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = RUN_ROOT.parent.parent
sys.path.insert(0, str(RUN_ROOT / "code"))
from _paths import ARTIFACTS, DATA, LOGS, QTERM_CODE  # noqa: E402

os.chdir(WORKSPACE)

from joli_invert import joli_invert_single, shard_n_batch  # noqa: E402
from qterm_attack import QtermAttack, QtermConfig, QtermTier  # noqa: E402
from shard_sim.attacker import ShardAttacker  # noqa: E402
from shard_sim.data_loader import FederatedDataLoader  # noqa: E402
from shard_sim.metrics import (  # noqa: E402
    compute_matching_accuracy,
    compute_reconstruction_mse,
)
from shard_sim.surrogate_model import SurrogateQFL  # noqa: E402
from terminal_attacks import chain_laplacian, graph_term_map  # noqa: E402

from l3_budget import estimate_joli_seconds, joli_profile  # noqa: E402

# Import Round-02 SHARD L2 graph init (local module; avoid name clash with sibling runs)
import importlib.util

_r02_spec = importlib.util.spec_from_file_location(
    "mnist_benchmark_round02",
    RUN_ROOT / "code" / "benchmark_round02.py",
)
_r02 = importlib.util.module_from_spec(_r02_spec)
assert _r02_spec.loader is not None
_r02_spec.loader.exec_module(_r02)
ShardAttackerR02 = _r02.ShardAttackerR02
hungarian_snapshot_mse = _r02.hungarian_snapshot_mse
psnr_from_mse = _r02.psnr_from_mse
simulate_b1_epoch_gradients = _r02.simulate_b1_epoch_gradients

# --- Round-04 knobs ---
SEEDS = [3, 7, 11]
N_SAMPLES = 32
BATCH_SIZE = 4
N_EPOCHS = 10
NOISE_LEVEL = 0.01
PARTIAL_ROWS = 8
PARTIAL_MAX_ITER = 150
SHARD_MAX_ITER = 200
SHARD_TOL = 1e-9
SHARD_GRAPH_LAMBDA = 0.5
SHARD_GRAPH_SPREAD = 0.35
PARTIAL_GRAPH_LAMBDA = 0.5

# Parent Round-03 Pareto operating point (compressive 28×28, dim_g=100)
JOLI_TV_LBFGS_R3 = 0.005
JOLI_ADAM_STEPS = 250
JOLI_LBFGS_ITER = 150
JOLI_N_BATCH_CAP = 50

TARGET_INPUT_MSE = 0.05
TARGET_PSNR_DB = 18.0
TARGET_SNAPSHOT_MSE_WEAK = 0.15
PASS_MIN_SEEDS = 2  # of 3

DIM_G_PRIMARY = 160
DIM_G_ALT = 100
RESIZE_28 = 28
RESIZE_14 = 14

SNAPSHOT_PATHS = (
    "gard_sparse_oracle",
    "lasa_qterm_T1p_graph",
    "shard_oracle",
)
L3_INVERTERS = ("joli_r3",)
GRID_PATHS = SNAPSHOT_PATHS
GRID_SEED = 7
GRID_DIM_G = DIM_G_PRIMARY
GRID_RESIZE = RESIZE_28

# GARD-SPARSE (privacy-breakthrough Round-01 patterns)
GARD_MEAN_ANCHOR = 10.0
GARD_VAL_FRACTION = 0.25
GARD_GRAPH_LAMBDA_GRID = (0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
T1P_GRAPH_PRIOR_BLEND = 0.35
T1P_GRAPH_PRIOR_LAMBDA = 0.35


def setup_logging() -> logging.Logger:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    logging.getLogger("shard_sim.attacker").setLevel(logging.WARNING)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(
                LOGS / "experiment_round04.log", mode="w", encoding="utf-8"
            ),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return logging.getLogger("mnist_recon_r04")


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_r3_tv_lbfgs() -> float:
    """Prefer parent artifacts/round03_metrics.json operating point."""
    r03 = WORKSPACE / "artifacts" / "round03_metrics.json"
    if r03.is_file():
        try:
            data = json.loads(r03.read_text(encoding="utf-8"))
            return float(data["operating_point"]["selected_tv_lbfgs"])
        except (KeyError, TypeError, json.JSONDecodeError):
            pass
    return JOLI_TV_LBFGS_R3


def build_oracle_incidence(
    epoch_batches_list: list[list[list[int]]],
    n_samples: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    rows: list[np.ndarray] = []
    meta: list[tuple[int, int]] = []
    for e, epoch_batches in enumerate(epoch_batches_list):
        for k, members in enumerate(epoch_batches):
            row = np.zeros(n_samples, dtype=np.float64)
            inv = 1.0 / len(members)
            for idx in members:
                row[idx] = inv
            rows.append(row)
            meta.append((e, k))
    return np.vstack(rows), meta


def partial_row_indices(
    n_epochs: int,
    k_batches: int,
    rows_per_epoch: int,
) -> np.ndarray:
    """Last ``rows_per_epoch`` minibatch rows per epoch (T1p-aligned budget)."""
    idx: list[int] = []
    for e in range(n_epochs):
        base = e * k_batches
        start = base + max(0, k_batches - rows_per_epoch)
        for r in range(start, base + k_batches):
            idx.append(r)
    return np.asarray(idx, dtype=int)


def split_train_val(n_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 40_000 + n_rows)
    n_val = max(1, int(round(GARD_VAL_FRACTION * n_rows)))
    if n_rows >= 8:
        n_val = max(2, n_val)
    n_val = min(n_val, n_rows - 1) if n_rows > 1 else 0
    perm = rng.permutation(n_rows)
    val = np.sort(perm[:n_val])
    train = np.sort(perm[n_val:])
    return train, val


def gard_solve_map(
    h_obs: np.ndarray,
    m_obs: np.ndarray,
    mean_snapshot: np.ndarray,
    *,
    graph_lambda: float = 0.0,
    laplacian: np.ndarray | None = None,
) -> np.ndarray:
    n_samples = h_obs.shape[1]
    if mean_snapshot.ndim == 2:
        mean_snapshot = mean_snapshot.mean(axis=0)
    mean_snapshot = np.asarray(mean_snapshot, dtype=np.float64).reshape(-1)
    anchor = np.ones((n_samples, 1), dtype=np.float64) / n_samples
    lhs = h_obs.T @ h_obs + GARD_MEAN_ANCHOR * (anchor @ anchor.T)
    if graph_lambda > 0.0 and laplacian is not None:
        lhs = lhs + graph_lambda * laplacian
    rhs = h_obs.T @ m_obs + GARD_MEAN_ANCHOR * anchor @ mean_snapshot[None, :]
    return np.linalg.solve(lhs + 1e-8 * np.eye(n_samples), rhs)


def gard_validation_mse(s_hat: np.ndarray, h_val: np.ndarray, m_val: np.ndarray) -> float:
    if h_val.shape[0] == 0:
        return 0.0
    return float(np.mean((h_val @ s_hat - m_val) ** 2))


def gard_select_graph_map(
    h_train: np.ndarray,
    m_train: np.ndarray,
    h_val: np.ndarray,
    m_val: np.ndarray,
    h_all: np.ndarray,
    m_all: np.ndarray,
    mean_snapshot: np.ndarray,
    lap: np.ndarray,
    rank_augmented: int,
) -> tuple[np.ndarray, float]:
    lambda_grid = (0.0,) if rank_augmented >= h_all.shape[1] else GARD_GRAPH_LAMBDA_GRID
    best_lambda = lambda_grid[0]
    best_score = float("inf")
    for lam in lambda_grid:
        s_candidate = gard_solve_map(
            h_train, m_train, mean_snapshot, graph_lambda=lam, laplacian=lap
        )
        score = gard_validation_mse(s_candidate, h_val, m_val)
        if score < best_score:
            best_score = score
            best_lambda = lam
    return (
        gard_solve_map(
            h_all, m_all, mean_snapshot, graph_lambda=best_lambda, laplacian=lap
        ),
        float(best_lambda),
    )


def _grad_to_snapshot_proxy(coeff: np.ndarray, grad: np.ndarray, dim_g: int) -> np.ndarray:
    """Invert g = A @ s_bar to a batch-mean snapshot proxy (LASA linearization)."""
    g = np.asarray(grad, dtype=np.float64)
    if coeff.shape[0] == dim_g:
        return np.linalg.solve(coeff, g)
    return np.linalg.lstsq(coeff, g, rcond=None)[0]


def recover_gard_sparse_oracle(
    *,
    h_full: np.ndarray,
    coeff_matrices: list[np.ndarray],
    batch_gradients: list[list[np.ndarray]],
    e_bar: np.ndarray,
    n_samples: int,
    dim_g: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """GARD-SPARSE with oracle incidence and T1p-aligned partial rows."""
    k_batches = len(batch_gradients[0])
    keep = partial_row_indices(len(batch_gradients), k_batches, PARTIAL_ROWS)
    h_obs = h_full[keep]
    m_rows: list[np.ndarray] = []
    for e, grads_e in enumerate(batch_gradients):
        p = min(PARTIAL_ROWS, len(grads_e))
        a_e = coeff_matrices[e]
        for g in grads_e[-p:]:
            m_rows.append(_grad_to_snapshot_proxy(a_e, g, dim_g))
    m_obs = np.vstack(m_rows)
    train_idx, val_idx = split_train_val(len(keep), seed)
    h_train, m_train = h_obs[train_idx], m_obs[train_idx]
    h_val, m_val = h_obs[val_idx], m_obs[val_idx]
    rank_aug = int(
        np.linalg.matrix_rank(
            np.vstack([h_obs, np.ones((1, n_samples), dtype=np.float64) / n_samples])
        )
    )
    lap = chain_laplacian(n_samples)
    s_gard, lam = gard_select_graph_map(
        h_train,
        m_train,
        h_val,
        m_val,
        h_obs,
        m_obs,
        e_bar,
        lap,
        rank_aug,
    )
    s_gard -= s_gard.mean(axis=0, keepdims=True)
    s_gard += e_bar[None, :]
    n_obs = int(len(keep))
    return s_gard, {
        "method": "gard_sparse_oracle_partial",
        "observed_intermediate_batch_gradients": n_obs,
        "observed_terminal_gradient_rows": 0,
        "partial_rows_per_epoch": PARTIAL_ROWS,
        "oracle_assignment": True,
        "selected_graph_lambda": lam,
        "rank_augmented": rank_aug,
    }


def graph_prior_polish(
    snapshots: np.ndarray,
    e_bar: np.ndarray,
    n_samples: int,
    *,
    graph_lambda: float = T1P_GRAPH_PRIOR_LAMBDA,
    blend: float = T1P_GRAPH_PRIOR_BLEND,
) -> np.ndarray:
    """Graph prior on recovered snapshots before L3 (chain MAP anchor)."""
    s_graph = graph_term_map(
        e_bar,
        n_samples,
        graph_lambda=graph_lambda,
        graph=chain_laplacian(n_samples),
    )
    b = float(np.clip(blend, 0.0, 1.0))
    s = (1.0 - b) * snapshots + b * s_graph
    s -= s.mean(axis=0, keepdims=True)
    s += e_bar[None, :]
    return s


def recover_snapshots(
    path: str,
    *,
    attacker: ShardAttackerR02,
    e_bar: np.ndarray,
    coeff_matrices: list[np.ndarray],
    batch_gradients: list[list[np.ndarray]],
    true_snapshots: np.ndarray,
    h_full: np.ndarray,
    epoch_batches_list: list[list[list[int]]],
    b1_coeff: list[np.ndarray] | None,
    b1_grads: list[list[np.ndarray]] | None,
    seed: int,
    dim_g: int,
) -> tuple[np.ndarray, dict]:
    del epoch_batches_list, b1_coeff, b1_grads

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

    if path == "lasa_qterm_T1p_graph":
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
        s_polished = graph_prior_polish(r.snapshots, e_bar, N_SAMPLES)
        return s_polished, {
            "tier": r.tier,
            "method": "lasa_qterm_T1p_graph_prior",
            "observed_terminal_gradient_rows": r.observed_terminal_gradient_rows,
            "observed_intermediate_batch_gradients": r.observed_intermediate_batch_gradients,
            "graph_prior_blend": T1P_GRAPH_PRIOR_BLEND,
            "graph_prior_lambda": T1P_GRAPH_PRIOR_LAMBDA,
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
    inverter: str,
    device: str,
    *,
    tv_lbfgs: float,
    logger: logging.Logger | None = None,
) -> np.ndarray:
    if inverter == "joli_r3":
        d = int(surrogate.W_enc.shape[1])
        dim_g = int(surrogate.dim_g)
        n_batch = min(shard_n_batch(d, dim_g), JOLI_N_BATCH_CAP)
        n = snapshots.shape[0]
        out = np.empty((n, d), dtype=np.float64)
        for i in range(n):
            t_img = time.perf_counter()
            out[i], _ = joli_invert_single(
                snapshots[i],
                surrogate,
                n_batch=n_batch,
                adam_steps=JOLI_ADAM_STEPS,
                lbfgs_iter=JOLI_LBFGS_ITER,
                tv_adam=0.0,
                tv_lbfgs=tv_lbfgs,
                seed=42 + i,
                device=device,
            )
            if logger is not None:
                logger.info(
                    "  joli_r3 image %d/%d %.1fs",
                    i + 1,
                    n,
                    time.perf_counter() - t_img,
                )
        return out
    raise ValueError(inverter)


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


def meets_targets(metrics: dict) -> dict:
    mse_ok = metrics.get("input_mse", float("inf")) <= TARGET_INPUT_MSE
    psnr_ok = metrics.get("input_psnr_db", 0.0) >= TARGET_PSNR_DB
    return {
        "input_mse_ok": mse_ok,
        "psnr_ok": psnr_ok,
        "both_ok": mse_ok and psnr_ok,
    }


def run_single_seed(
    seed: int,
    dim_g: int,
    resize: int,
    logger: logging.Logger,
    device: str,
    tv_lbfgs: float,
    *,
    capture_grid: bool,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    loader = FederatedDataLoader(
        dataset="mnist",
        n_samples=N_SAMPLES,
        batch_size=BATCH_SIZE,
        seed=seed,
        resize=resize,
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
    b1_coeff: list[np.ndarray] = []
    b1_grads: list[list[np.ndarray]] = []

    for e in range(N_EPOCHS):
        epoch_batches = loader.get_epoch_batches()
        epoch_batches_list.append(epoch_batches)
        a_e, grads_e = surrogate.compute_batch_gradients(images, epoch_batches)
        coeff_matrices.append(a_e)
        batch_gradients.append(grads_e)
        a_b1, g_b1 = simulate_b1_epoch_gradients(images, surrogate, seed, e)
        b1_coeff.append(a_b1)
        b1_grads.append(g_b1)

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

    out: dict = {
        "seed": seed,
        "dim_g": dim_g,
        "resize": resize,
        "d": d,
        "image_shape": list(image_shape),
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
            epoch_batches_list=epoch_batches_list,
            b1_coeff=b1_coeff,
            b1_grads=b1_grads,
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

        for inverter in L3_INVERTERS:
            logger.info(
                "L3 start resize=%d seed=%d path=%s inv=%s (N=%d d=%d)",
                resize,
                seed,
                path,
                inverter,
                s_rec.shape[0],
                d,
            )
            t0 = time.perf_counter()
            try:
                x_rec = run_l3(
                    s_rec,
                    surrogate,
                    inverter,
                    device,
                    tv_lbfgs=tv_lbfgs,
                    logger=logger,
                )
                l3_time = time.perf_counter() - t0
                metrics = evaluate_reconstruction(x_rec, x_true, s_rec, true_snapshots)
                metrics["l3_runtime_sec"] = l3_time
                metrics["tv_lbfgs"] = tv_lbfgs
                metrics["targets"] = meets_targets(metrics)
                path_out["l3"][inverter] = metrics
                logger.info(
                    "resize=%d seed=%d dim_g=%d %s %s snap=%.4f mse=%.4f psnr=%.2f pass=%s",
                    resize,
                    seed,
                    dim_g,
                    path,
                    inverter,
                    metrics["snapshot_mse"],
                    metrics["input_mse"],
                    metrics["input_psnr_db"],
                    metrics["targets"]["both_ok"],
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
                path_out["l3"][inverter] = {"error": str(exc)}
                logger.exception(
                    "L3 failed resize=%d seed=%d dim_g=%d path=%s inv=%s",
                    resize,
                    seed,
                    dim_g,
                    path,
                    inverter,
                )

        out["paths"][path] = path_out

    return out


def aggregate_method(per_seed: list[dict], path: str, inverter: str) -> dict:
    rows = []
    for r in per_seed:
        l3 = r["paths"][path]["l3"].get(inverter, {})
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


def find_accepting_configs(summary: dict) -> list[dict]:
    accepting = []
    for key, agg in summary.items():
        if agg.get("passes_acceptance"):
            path, inverter = key.split("__", 1)
            accepting.append(
                {
                    "config_key": key,
                    "path": path,
                    "inverter": inverter,
                    "pass_count": agg["pass_count_both"],
                    "input_mse_mean": agg["input_mse_mean"],
                    "input_psnr_db_mean": agg["input_psnr_db_mean"],
                }
            )
    return accepting


def plot_recon_grid(
    panels: list[tuple[str, np.ndarray, np.ndarray, tuple]],
    save_path: Path,
    *,
    title: str,
) -> None:
    n_cols = panels[0][1].shape[0] if panels else 0
    n_rows = len(panels)
    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(1.4 * n_cols, 2.4 * n_rows))
    if n_cols == 1:
        axes = np.atleast_2d(axes)
    for row_i, (label, x_true, x_rec, shape) in enumerate(panels):
        side = shape[0]
        for j in range(n_cols):
            axes[row_i * 2, j].imshow(
                x_true[j].reshape(side, side), cmap="gray", vmin=0, vmax=1
            )
            axes[row_i * 2, j].axis("off")
            axes[row_i * 2 + 1, j].imshow(
                x_rec[j].reshape(side, side), cmap="gray", vmin=0, vmax=1
            )
            axes[row_i * 2 + 1, j].axis("off")
        axes[row_i * 2, 0].set_ylabel(f"{label}\norig", fontsize=7)
        axes[row_i * 2 + 1, 0].set_ylabel("rec", fontsize=7)
    plt.suptitle(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_resolution_block(
    resize: int,
    dim_g_list: tuple[int, ...],
    logger: logging.Logger,
    device: str,
    tv_lbfgs: float,
) -> dict:
    per_seed_by_dim: dict[int, list[dict]] = {}
    summary_by_dim: dict[str, dict] = {}

    for dim_g in dim_g_list:
        logger.info("=== resize=%d dim_g=%d ===", resize, dim_g)
        per_seed = [
            run_single_seed(
                s,
                dim_g,
                resize,
                logger,
                device,
                tv_lbfgs,
                capture_grid=(
                    s == GRID_SEED and dim_g == GRID_DIM_G and resize == GRID_RESIZE
                ),
            )
            for s in SEEDS
        ]
        per_seed_by_dim[dim_g] = per_seed
        summary: dict = {}
        for path in SNAPSHOT_PATHS:
            for inverter in L3_INVERTERS:
                key = f"{path}__{inverter}"
                summary[key] = aggregate_method(per_seed, path, inverter)
        summary_by_dim[str(dim_g)] = summary

    grid_path = None
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
            grid_path = ARTIFACTS / f"round04_recon_grid_{resize}x{resize}_dim{GRID_DIM_G}.png"
            plot_recon_grid(
                panels,
                grid_path,
                title=(
                    f"Round-04 MNIST {resize}×{resize} JOLI-R3 "
                    f"(dim_g={GRID_DIM_G}, seed={GRID_SEED})"
                ),
            )

    accepting_all: list[dict] = []
    for dim_s, sm in summary_by_dim.items():
        acc = find_accepting_configs(sm)
        for a in acc:
            a["resize"] = resize
            a["dim_g"] = int(dim_s)
        accepting_all.extend(acc)

    return {
        "resize": resize,
        "dim_g_list": list(dim_g_list),
        "per_seed_by_dim": per_seed_by_dim,
        "summary_by_dim_g": summary_by_dim,
        "accepting_configs": accepting_all,
        "grid_png": str(grid_path) if grid_path else None,
    }


def main() -> None:
    from quick_config import is_quick_mode, patch_module_globals

    logger = setup_logging()
    qcfg = patch_module_globals(sys.modules[__name__], research_round=4)
    device = pick_device()
    tv_lbfgs = load_r3_tv_lbfgs()
    logger.info(
        "Round-04 MNIST recon seeds=%s device=%s joli_tv_lbfgs_r3=%g",
        SEEDS,
        device,
        tv_lbfgs,
    )
    if qcfg is not None:
        logger.info(
            "QFL_QUICK: resize=%d N=%d paths=%s (full: QFL_QUICK=0 QFL_FULL=1)",
            qcfg.resize,
            qcfg.n_samples,
            qcfg.snapshot_paths,
        )

    if is_quick_mode():
        block_28 = {
            "resize": qcfg.resize if qcfg else RESIZE_14,
            "dim_g_list": list(qcfg.dim_g_list) if qcfg else (DIM_G_ALT,),
            "per_seed_by_dim": {},
            "summary_by_dim_g": {},
            "accepting_configs": [],
            "grid_png": None,
        }
        block_14 = run_resolution_block(
            qcfg.resize if qcfg else RESIZE_14,
            tuple(qcfg.dim_g_list) if qcfg else (DIM_G_ALT,),
            logger,
            device,
            tv_lbfgs,
        )
        accepting_28 = []
        run_14 = True
    else:
        # Round-04 focuses on snapshot paths; dim_g=160 was Round-02 sweet spot on 28×28.
        block_28 = run_resolution_block(
            RESIZE_28,
            (DIM_G_PRIMARY,),
            logger,
            device,
            tv_lbfgs,
        )
        accepting_28 = block_28["accepting_configs"]
        run_14 = len(accepting_28) == 0
        block_14 = None
        if run_14:
            logger.info("28×28: no config with >=2/3 seed pass — running 14×14 fallback")
            block_14 = run_resolution_block(
                RESIZE_14,
                (DIM_G_ALT,),
                logger,
                device,
                tv_lbfgs,
            )

    per_seed_json = []
    for block in (block_28, block_14):
        if block is None:
            continue
        for dim_g, per_seed in block["per_seed_by_dim"].items():
            for r in per_seed:
                rj = {k: v for k, v in r.items() if k != "grid_panels"}
                per_seed_json.append(rj)

    all_accepting = accepting_28 + (block_14["accepting_configs"] if block_14 else [])
    targets_met = len(all_accepting) > 0

    payload = {
        "benchmark": "round04_qfl_vqc_mnist_recon",
        "method_name_stack": (
            "SurrogateQFL + GARD-SPARSE/T1p-graph/SHARD snapshots + JOLI-R3 L3"
        ),
        "round04_innovations": {
            "l3": "joli_r3 tv_lbfgs from parent Round-03 Pareto",
            "gard_sparse_oracle": (
                "oracle incidence + partial_rows=8/epoch + chain graph MAP "
                "(privacy-breakthrough GARD patterns)"
            ),
            "lasa_qterm_T1p_graph": (
                "T1p partial honest disaggregate + graph_term_map polish before L3"
            ),
            "fallback_14x14": run_14,
        },
        "mnist_tracks": {
            "primary_28x28": {
                "resize": RESIZE_28,
                "dim_g_sweep": [DIM_G_PRIMARY],
                "accepting_configs": accepting_28,
            },
            "optional_14x14": (
                {
                    "resize": RESIZE_14,
                    "dim_g": [DIM_G_ALT],
                    "accepting_configs": block_14["accepting_configs"] if block_14 else [],
                    "ran_because_28_failed": True,
                }
                if run_14
                else {"ran": False}
            ),
        },
        "joli_r3": {
            "tv_lbfgs": tv_lbfgs,
            "adam_steps": JOLI_ADAM_STEPS,
            "lbfgs_iter": JOLI_LBFGS_ITER,
            "n_batch_cap": JOLI_N_BATCH_CAP,
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
            "snapshot_mse_max_weak": TARGET_SNAPSHOT_MSE_WEAK,
            "pass_min_seeds_of_3": PASS_MIN_SEEDS,
        },
        "summary_28x28_by_dim_g": block_28["summary_by_dim_g"],
        "summary_14x14_by_dim_g": block_14["summary_by_dim_g"] if block_14 else {},
        "per_seed": per_seed_json,
        "accepting_configs": all_accepting,
        "targets_met_any_config": targets_met,
        "grid_png_28": block_28.get("grid_png"),
        "grid_png_14": block_14.get("grid_png") if block_14 else None,
        "honest_verdict": (
            f"Acceptance target met: {len(all_accepting)} config(s) with "
            f"MSE≤{TARGET_INPUT_MSE} and PSNR≥{TARGET_PSNR_DB} on ≥{PASS_MIN_SEEDS}/3 seeds."
            if targets_met
            else (
                f"No config reached both targets on ≥{PASS_MIN_SEEDS}/3 seeds "
                f"(28×28{' + 14×14 fallback' if run_14 else ''}). "
                "Best weak-path candidates in summary_28x28_by_dim_g."
            )
        ),
    }

    out_json = ARTIFACTS / "round04_metrics.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote %s targets_met=%s", out_json, targets_met)


if __name__ == "__main__":
    main()
