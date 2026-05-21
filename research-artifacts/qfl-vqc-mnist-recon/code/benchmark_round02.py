#!/usr/bin/env python3
"""Round-02: full 28×28 MNIST reconstruction (d=784) with dim_g sweep and tuned SHARD/T1p.

Supervisor focus (Round 02):
  - Primary track: FederatedDataLoader MNIST 28×28 (d=784)
  - SHARD L2: more iterations, graph level-1 anchor init, optional B=1 oracle diagnostic
  - T1p: terminal rows + partial disaggregation budget (no T1b row flood in primary grid)
  - dim_g ∈ {100, 160, 256} — compressive regime for JOLI TV
  - Grids: shard_oracle vs lasa_qterm_T1p vs lasa_qterm_T1b (JOLI L3)
  - Per-seed pass rate vs input_mse≤0.05 and psnr≥18
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
sys.path.insert(0, str(RUN_ROOT / "code"))
from _paths import ARTIFACTS, DATA, LOGS, QTERM_CODE  # noqa: E402

os.chdir(RUN_ROOT.parent.parent)

from joli_invert import joli_invert  # noqa: E402
from qterm_attack import QtermAttack, QtermConfig, QtermTier  # noqa: E402
from shard_sim.attacker import ShardAttacker  # noqa: E402
from shard_sim.data_loader import FederatedDataLoader  # noqa: E402
from shard_sim.metrics import (  # noqa: E402
    compute_matching_accuracy,
    compute_reconstruction_mse,
)
from shard_sim.surrogate_model import SurrogateQFL  # noqa: E402
from terminal_attacks import graph_term_map  # noqa: E402

# --- Round-02 experiment knobs ---
RESIZE = 28
SEEDS = [3, 7, 11]
DIM_G_SWEEP = (100, 160, 256)
N_SAMPLES = 32
BATCH_SIZE = 4
N_EPOCHS = 10
NOISE_LEVEL = 0.01
SHARD_MAX_ITER = 500
SHARD_TOL = 1e-9
SHARD_GRAPH_LAMBDA = 0.5
SHARD_GRAPH_SPREAD = 0.35
PARTIAL_ROWS = 8
PARTIAL_MAX_ITER = 150
PARTIAL_GRAPH_LAMBDA = 0.5
TARGET_INPUT_MSE = 0.05
TARGET_PSNR_DB = 18.0
TARGET_SNAPSHOT_MSE_WEAK = 0.15

SNAPSHOT_PATHS = ("shard_oracle", "lasa_qterm_T1p", "lasa_qterm_T1b")
L3_INVERTERS = ("shard_l3", "joli_l3")
GRID_PATHS = ("shard_oracle", "lasa_qterm_T1p", "lasa_qterm_T1b")
GRID_INVERTER = "joli_l3"
GRID_SEED = 7
GRID_DIM_G = 160


class ShardAttackerR02(ShardAttacker):
    """SHARD L2 with graph level-1 anchor (replaces Gaussian Phase-2 init)."""

    def __init__(
        self,
        *args,
        graph_lambda: float = SHARD_GRAPH_LAMBDA,
        spread_scale: float = SHARD_GRAPH_SPREAD,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.graph_lambda = graph_lambda
        self.spread_scale = spread_scale

    def level2_disaggregate(
        self,
        e_bar_data: np.ndarray,
        coeff_matrices: list[np.ndarray],
        batch_gradients: list[list[np.ndarray]],
        true_snapshots: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.batch_size == 1:
            return super().level2_disaggregate(
                e_bar_data, coeff_matrices, batch_gradients, true_snapshots
            )

        N = self.n_samples
        B = self.batch_size
        K = N // B
        E = len(coeff_matrices)
        dim_g = self.dim_g
        D = coeff_matrices[0].shape[0]

        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        batch_means = np.empty((E, K, dim_g))
        for e in range(E):
            A_e = coeff_matrices[e]
            G_e = np.array(batch_gradients[e])
            if D == dim_g:
                batch_means[e] = np.linalg.solve(A_e, G_e.T).T
            else:
                batch_means[e] = np.linalg.lstsq(A_e, G_e.T, rcond=None)[0].T

        S = graph_term_map(
            e_bar_data,
            N,
            graph_lambda=self.graph_lambda,
            spread_scale=self.spread_scale,
        )
        alpha = 0.6
        best_S = S.copy()
        best_residual = np.inf

        iter_range = range(1, self.max_iter + 1)
        pbar = (
            tqdm(
                iter_range,
                desc="Level 2 (graph init)",
                unit="iter",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}]",
            )
            if tqdm is not None
            else iter_range
        )

        frob_change = np.inf
        for t in pbar:
            S_old = S.copy()
            if dim_g > 500:
                if t == 1:
                    proj_dim = min(200, dim_g)
                    _rng_proj = np.random.default_rng(42)
                    R = _rng_proj.standard_normal((dim_g, proj_dim)) / np.sqrt(proj_dim)
                S_proj = S @ R
                bm_proj = batch_means @ R
                ebar_proj = e_bar_data @ R
                assign_arr = self._assign_all_epochs(
                    S_proj, bm_proj, B, ebar_proj, max_swap_rounds=3
                )
            else:
                assign_arr = self._assign_all_epochs(
                    S, batch_means, B, e_bar_data, max_swap_rounds=5
                )

            PtP = np.zeros((N, N))
            PtM = np.zeros((N, dim_g))
            w = 1.0 / B
            w2 = w * w
            for e in range(E):
                ae = assign_arr[e]
                for k_idx in range(K):
                    members = np.where(ae == k_idx)[0]
                    for ii in members:
                        for jj in members:
                            PtP[ii, jj] += w2
                        PtM[ii] += w * batch_means[e, k_idx]
            S_new = np.linalg.lstsq(PtP, PtM, rcond=None)[0]
            S = alpha * S_new + (1.0 - alpha) * S_old
            S -= S.mean(axis=0) - e_bar_data

            total_residual = 0.0
            for e in range(E):
                ae = assign_arr[e]
                for k_idx in range(K):
                    members = np.where(ae == k_idx)[0]
                    pred_mean = S[members].mean(axis=0)
                    total_residual += float(
                        np.sum((pred_mean - batch_means[e, k_idx]) ** 2)
                    )
            if total_residual < best_residual:
                best_residual = total_residual
                best_S = S.copy()

            frob_change = np.linalg.norm(S - S_old, "fro")
            if true_snapshots is not None:
                acc = self._matching_accuracy(S, true_snapshots)
                if tqdm is not None and hasattr(pbar, "set_postfix_str"):
                    pbar.set_postfix_str(
                        f"acc={acc:.0%} res={total_residual:.1e}"
                    )
            elif tqdm is not None and hasattr(pbar, "set_postfix_str"):
                pbar.set_postfix_str(f"ΔS={frob_change:.1e} res={total_residual:.1e}")

            if frob_change < self.tol:
                break
        else:
            warnings.warn(
                f"SHARD did not converge within {self.max_iter} iterations "
                f"(final ||ΔS||_F = {frob_change:.2e}).",
                stacklevel=2,
            )
        if tqdm is not None and hasattr(pbar, "close"):
            pbar.close()
        return best_S


def setup_logging() -> logging.Logger:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    logging.getLogger("shard_sim.attacker").setLevel(logging.WARNING)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOGS / "experiment_round02.log", mode="w"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return logging.getLogger("mnist_recon_r02")


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def hungarian_snapshot_mse(recovered: np.ndarray, truth: np.ndarray) -> tuple[float, np.ndarray]:
    r_sq = np.sum(recovered**2, axis=1, keepdims=True)
    t_sq = np.sum(truth**2, axis=1, keepdims=True)
    dist = r_sq + t_sq.T - 2.0 * recovered @ truth.T
    np.maximum(dist, 0.0, out=dist)
    row, col = linear_sum_assignment(dist)
    mse = float(np.mean((recovered[row] - truth[col]) ** 2))
    return mse, col


def psnr_from_mse(mse: float, peak: float = 1.0) -> float:
    if mse <= 1e-16:
        return float("inf")
    return float(10.0 * np.log10((peak**2) / mse))


def simulate_b1_epoch_gradients(
    images: torch.Tensor,
    surrogate: SurrogateQFL,
    seed: int,
    epoch: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    n = images.shape[0]
    perm = np.random.default_rng(seed + 9100 + epoch).permutation(n)
    batches = [[int(i)] for i in perm]
    return surrogate.compute_batch_gradients(images, batches)


def recover_snapshots(
    path: str,
    *,
    attacker: ShardAttackerR02,
    e_bar: np.ndarray,
    coeff_matrices: list[np.ndarray],
    batch_gradients: list[list[np.ndarray]],
    true_snapshots: np.ndarray,
    b1_coeff: list[np.ndarray] | None,
    b1_grads: list[list[np.ndarray]] | None,
    seed: int,
    dim_g: int,
) -> tuple[np.ndarray, dict]:
    if path == "shard_oracle":
        s = attacker.level2_disaggregate(
            e_bar, coeff_matrices, batch_gradients, true_snapshots
        )
        meta = {
            "observed_intermediate_batch_gradients": sum(len(g) for g in batch_gradients),
            "observed_terminal_gradient_rows": 0,
            "l2_init": "graph_term_map",
            "shard_max_iter": attacker.max_iter,
        }
        return s, meta

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

    if path == "lasa_qterm_T1b":
        if b1_coeff is None or b1_grads is None:
            raise ValueError("T1b requires b1_coeff and b1_grads")
        e_bar_b1 = attacker.level1_mean_recovery(b1_coeff, b1_grads)
        q = QtermAttack(
            QtermConfig(
                tier=QtermTier.T1B,
                n_samples=N_SAMPLES,
                dim_g=dim_g,
                random_seed=seed,
            )
        )
        r = q.recover(e_bar_b1, b1_coeff, b1_gradients=b1_grads)
        return r.snapshots, {
            "tier": r.tier,
            "method": r.method,
            "observed_terminal_gradient_rows": r.observed_terminal_gradient_rows,
            "observed_intermediate_batch_gradients": r.observed_intermediate_batch_gradients,
        }

    raise ValueError(path)


def run_l3(
    snapshots: np.ndarray,
    surrogate: SurrogateQFL,
    inverter: str,
    device: str,
) -> np.ndarray:
    if inverter == "shard_l3":
        attacker = ShardAttacker(
            dim_g=surrogate.dim_g,
            n_samples=N_SAMPLES,
            batch_size=BATCH_SIZE,
        )
        return attacker.level3_invert(snapshots, surrogate, device=device)
    if inverter == "joli_l3":
        return joli_invert(snapshots, surrogate, device=device)
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
    logger: logging.Logger,
    device: str,
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
    b1_coeff: list[np.ndarray] = []
    b1_grads: list[list[np.ndarray]] = []

    for e in range(N_EPOCHS):
        epoch_batches = loader.get_epoch_batches()
        a_e, grads_e = surrogate.compute_batch_gradients(images, epoch_batches)
        coeff_matrices.append(a_e)
        batch_gradients.append(grads_e)
        a_b1, g_b1 = simulate_b1_epoch_gradients(images, surrogate, seed, e)
        b1_coeff.append(a_b1)
        b1_grads.append(g_b1)

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
    )
    e_bar = attacker.level1_mean_recovery(coeff_matrices, batch_gradients)

    out: dict = {
        "seed": seed,
        "dim_g": dim_g,
        "image_shape": list(image_shape),
        "d": d,
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
            key = f"{path}__{inverter}"
            t0 = time.perf_counter()
            try:
                x_rec = run_l3(s_rec, surrogate, inverter, device)
                l3_time = time.perf_counter() - t0
                metrics = evaluate_reconstruction(x_rec, x_true, s_rec, true_snapshots)
                metrics["l3_runtime_sec"] = l3_time
                metrics["targets"] = meets_targets(metrics)
                path_out["l3"][inverter] = metrics
                logger.info(
                    "seed=%d dim_g=%d %s %s snap=%.4f mse=%.4f psnr=%.2f pass=%s",
                    seed,
                    dim_g,
                    path,
                    inverter,
                    metrics["snapshot_mse"],
                    metrics["input_mse"],
                    metrics["input_psnr_db"],
                    metrics["targets"]["both_ok"],
                )
                if capture_grid and inverter == GRID_INVERTER and path in GRID_PATHS:
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
                    "L3 failed seed=%d dim_g=%d path=%s inv=%s", seed, dim_g, path, inverter
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
    out = {"n_ok": len(rows)}
    for k in keys:
        vals = [x[k] for x in rows]
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals, ddof=0))
    pass_both = sum(1 for x in rows if x.get("targets", {}).get("both_ok"))
    pass_mse = sum(1 for x in rows if x.get("targets", {}).get("input_mse_ok"))
    pass_psnr = sum(1 for x in rows if x.get("targets", {}).get("psnr_ok"))
    n = len(rows)
    out["pass_rate_both"] = pass_both / n
    out["pass_rate_input_mse"] = pass_mse / n
    out["pass_rate_psnr"] = pass_psnr / n
    return out


def gap_report(value: float, target: float, *, higher_is_better: bool) -> dict:
    if higher_is_better:
        gap = target - value
        meets = value >= target
    else:
        gap = value - target
        meets = value <= target
    return {"value": value, "target": target, "gap": gap, "meets_target": meets}


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


def save_grids(per_seed_by_dim: dict[int, list[dict]]) -> dict[str, str]:
    paths: dict[str, str] = {}
    for dim_g, per_seed in per_seed_by_dim.items():
        rec = next((r for r in per_seed if r["seed"] == GRID_SEED), None)
        if not rec or not rec.get("grid_panels"):
            continue
        by_label = {p["label"]: p for p in rec["grid_panels"]}
        panels: list[tuple[str, np.ndarray, np.ndarray, tuple]] = []
        for label in GRID_PATHS:
            if label not in by_label:
                continue
            p = by_label[label]
            panels.append(
                (
                    label,
                    np.asarray(p["x_true"]),
                    np.asarray(p["x_rec"]),
                    tuple(p["image_shape"]),
                )
            )
        if not panels:
            continue
        out = ARTIFACTS / f"round02_recon_grid_dim{dim_g}.png"
        plot_recon_grid(
            panels,
            out,
            title=(
                f"Round-02 MNIST {RESIZE}×{RESIZE} JOLI L3 "
                f"(dim_g={dim_g}, seed={GRID_SEED}): oracle / T1p / T1b"
            ),
        )
        paths[str(dim_g)] = str(out)
    return paths


def build_dim_g_table(summary_by_dim: dict) -> list[dict]:
    """Compact 28×28 table rows for reporting."""
    rows = []
    for dim_g in DIM_G_SWEEP:
        sm = summary_by_dim.get(str(dim_g), {})
        for path in SNAPSHOT_PATHS:
            for inverter in L3_INVERTERS:
                key = f"{path}__{inverter}"
                s = sm.get(key, {})
                if s.get("n_ok", 0) == 0:
                    continue
                rows.append(
                    {
                        "dim_g": dim_g,
                        "path": path,
                        "inverter": inverter,
                        "snapshot_mse_mean": s.get("snapshot_mse_mean"),
                        "input_mse_mean": s.get("input_mse_mean"),
                        "input_psnr_db_mean": s.get("input_psnr_db_mean"),
                        "pass_rate_both": s.get("pass_rate_both"),
                        "pass_rate_input_mse": s.get("pass_rate_input_mse"),
                        "pass_rate_psnr": s.get("pass_rate_psnr"),
                    }
                )
    return rows


def main() -> None:
    logger = setup_logging()
    device = pick_device()
    logger.info(
        "Round-02 MNIST %dx%d dim_g_sweep=%s seeds=%s device=%s",
        RESIZE,
        RESIZE,
        DIM_G_SWEEP,
        SEEDS,
        device,
    )

    per_seed_by_dim: dict[int, list[dict]] = {}
    summary_by_dim: dict[str, dict] = {}

    for dim_g in DIM_G_SWEEP:
        logger.info("=== dim_g=%d ===", dim_g)
        per_seed = [
            run_single_seed(
                s,
                dim_g,
                logger,
                device,
                capture_grid=(s == GRID_SEED and dim_g == GRID_DIM_G),
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

    grid_paths = save_grids(per_seed_by_dim)

    per_seed_json = []
    for dim_g, per_seed in per_seed_by_dim.items():
        for r in per_seed:
            rj = {k: v for k, v in r.items() if k != "grid_panels"}
            per_seed_json.append(rj)

    table_28 = build_dim_g_table(summary_by_dim)

    # Best compressive (dim_g < d) path by mean input MSE using joli_l3
    compressive_rank = []
    for dim_g in DIM_G_SWEEP:
        if dim_g >= RESIZE * RESIZE:
            continue
        for path in SNAPSHOT_PATHS:
            key = f"{path}__joli_l3"
            s = summary_by_dim[str(dim_g)].get(key, {})
            if s.get("n_ok") == len(SEEDS):
                compressive_rank.append(
                    (dim_g, path, s["input_mse_mean"], s["input_psnr_db_mean"])
                )
    compressive_rank.sort(key=lambda x: x[2])
    best_dim_g, best_path, best_mse, best_psnr = (
        compressive_rank[0] if compressive_rank else (None, None, float("nan"), float("nan"))
    )
    best_key = f"{best_path}__joli_l3" if best_path else "none"

    targets = {
        "input_mse": gap_report(best_mse, TARGET_INPUT_MSE, higher_is_better=False),
        "psnr_db": gap_report(best_psnr, TARGET_PSNR_DB, higher_is_better=True),
    }

    payload = {
        "benchmark": "round02_qfl_vqc_mnist_recon",
        "method_name_stack": "SurrogateQFL + SHARD/LASA-QTERM + L3 (shard/joli)",
        "mnist_choice": {
            "resolution": RESIZE,
            "rationale": "Full 28×28 FederatedDataLoader primary track (d=784).",
            "loader": "shard_sim.FederatedDataLoader",
            "data_root": str(DATA),
        },
        "round02_tuning": {
            "shard_max_iter": SHARD_MAX_ITER,
            "shard_tol": SHARD_TOL,
            "shard_l2_init": "graph_term_map",
            "partial_rows_t1p": PARTIAL_ROWS,
            "partial_max_iter_t1p": PARTIAL_MAX_ITER,
            "partial_graph_lambda_t1p": PARTIAL_GRAPH_LAMBDA,
            "dim_g_sweep": list(DIM_G_SWEEP),
        },
        "simulator": {
            "n_samples": N_SAMPLES,
            "batch_size": BATCH_SIZE,
            "n_epochs": N_EPOCHS,
            "noise_level": NOISE_LEVEL,
            "seeds": SEEDS,
            "device": device,
        },
        "targets": {
            "input_mse_max": TARGET_INPUT_MSE,
            "psnr_min_db": TARGET_PSNR_DB,
            "snapshot_mse_max_weak": TARGET_SNAPSHOT_MSE_WEAK,
        },
        "per_seed": per_seed_json,
        "summary_by_dim_g": summary_by_dim,
        "table_28x28": table_28,
        "grid_png_by_dim_g": grid_paths,
        "headline": {
            "best_compressive_dim_g": best_dim_g,
            "best_compressive_path": best_key,
            "best_input_mse_mean": best_mse,
            "best_input_psnr_db_mean": best_psnr,
            "gap_to_targets": targets,
            "compressive_ranking_joli": compressive_rank,
        },
        "honest_verdict": (
            f"Best compressive JOLI path {best_key} @ dim_g={best_dim_g}: "
            f"input MSE {best_mse:.4f} (target ≤{TARGET_INPUT_MSE}), "
            f"PSNR {best_psnr:.2f} dB (target ≥{TARGET_PSNR_DB}). "
            + (
                "28×28 targets not met in Round 2."
                if best_mse > TARGET_INPUT_MSE or best_psnr < TARGET_PSNR_DB
                else "28×28 targets met on best compressive path."
            )
        ),
    }

    out_json = ARTIFACTS / "round02_metrics.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_json)


if __name__ == "__main__":
    main()
