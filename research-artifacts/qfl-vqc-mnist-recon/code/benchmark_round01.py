#!/usr/bin/env python3
"""Round-01 end-to-end MNIST reconstruction via SurrogateQFL + snapshot recovery + L3.

Paths:
  - SHARD Stage-2 oracle (full intermediate batch gradients)
  - LASA-QTERM T1p (partial terminal rows)
  - LASA-QTERM T1b (B=1 per-client terminals)

Inversion: ShardAttacker.level3_invert and joli_invert (parent code/).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
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

os.chdir(RUN_ROOT.parent.parent)  # workspace root for ./data MNIST path

from joli_invert import joli_invert  # noqa: E402
from qterm_attack import QtermAttack, QtermConfig, QtermTier  # noqa: E402
from shard_sim.attacker import ShardAttacker  # noqa: E402
from shard_sim.data_loader import FederatedDataLoader  # noqa: E402
from shard_sim.metrics import (  # noqa: E402
    compute_matching_accuracy,
    compute_reconstruction_mse,
)
from shard_sim.surrogate_model import SurrogateQFL  # noqa: E402

# Round-01: 8x8 MNIST (d=64) for tractable L3 across 3 snapshot paths × 2 inverters × 3 seeds.
# config.json prefers 28x28; defer full-res to Round 2+ after snapshot/L3 tuning.
RESIZE = 8
SEEDS = [3, 7, 11]
N_SAMPLES = 32
BATCH_SIZE = 4
N_EPOCHS = 10
DIM_G = 100
NOISE_LEVEL = 0.01
SHARD_MAX_ITER = 200
PARTIAL_ROWS = 7
TARGET_INPUT_MSE = 0.05
TARGET_PSNR_DB = 18.0
TARGET_SNAPSHOT_MSE_WEAK = 0.15

SNAPSHOT_PATHS = ("shard_oracle", "lasa_qterm_T1p", "lasa_qterm_T1b")
L3_INVERTERS = ("shard_l3", "joli_l3")


def setup_logging() -> logging.Logger:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    logging.getLogger("shard_sim.attacker").setLevel(logging.WARNING)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOGS / "experiment_round01.log", mode="w"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return logging.getLogger("mnist_recon_r01")


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
    """T1b: one gradient row per client (B=1), reshuffled each epoch."""
    n = images.shape[0]
    perm = np.random.default_rng(seed + 9100 + epoch).permutation(n)
    batches = [[int(i)] for i in perm]
    return surrogate.compute_batch_gradients(images, batches)


def recover_snapshots(
    path: str,
    *,
    attacker: ShardAttacker,
    e_bar: np.ndarray,
    coeff_matrices: list[np.ndarray],
    batch_gradients: list[list[np.ndarray]],
    true_snapshots: np.ndarray,
    b1_coeff: list[np.ndarray] | None,
    b1_grads: list[list[np.ndarray]] | None,
    seed: int,
) -> tuple[np.ndarray, dict]:
    if path == "shard_oracle":
        s = attacker.level2_disaggregate(
            e_bar, coeff_matrices, batch_gradients, true_snapshots
        )
        meta = {
            "observed_intermediate_batch_gradients": sum(len(g) for g in batch_gradients),
            "observed_terminal_gradient_rows": 0,
        }
        return s, meta

    if path == "lasa_qterm_T1p":
        q = QtermAttack(
            QtermConfig(
                tier=QtermTier.T1P,
                n_samples=N_SAMPLES,
                batch_size=BATCH_SIZE,
                dim_g=DIM_G,
                partial_rows_per_epoch=PARTIAL_ROWS,
                random_seed=seed,
            )
        )
        r = q.recover(e_bar, coeff_matrices, batch_gradients=batch_gradients)
        return r.snapshots, {
            "tier": r.tier,
            "method": r.method,
            "observed_terminal_gradient_rows": r.observed_terminal_gradient_rows,
            "observed_intermediate_batch_gradients": r.observed_intermediate_batch_gradients,
        }

    if path == "lasa_qterm_T1b":
        if b1_coeff is None or b1_grads is None:
            raise ValueError("T1b requires b1_coeff and b1_grads")
        e_bar_b1 = attacker.level1_mean_recovery(b1_coeff, b1_grads)
        q = QtermAttack(
            QtermConfig(
                tier=QtermTier.T1B,
                n_samples=N_SAMPLES,
                dim_g=DIM_G,
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
        attacker = ShardAttacker(dim_g=DIM_G, n_samples=N_SAMPLES, batch_size=BATCH_SIZE)
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
        "image_matching_acc": float(
            compute_matching_accuracy(x_rec, x_true)[0]
        ),
    }


def run_single_seed(seed: int, logger: logging.Logger, device: str) -> dict:
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
        dim_g=DIM_G,
        n_params=DIM_G,
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

    attacker = ShardAttacker(
        dim_g=DIM_G,
        n_samples=N_SAMPLES,
        batch_size=BATCH_SIZE,
        max_iter=SHARD_MAX_ITER,
        tol=1e-8,
        random_seed=seed,
    )
    e_bar = attacker.level1_mean_recovery(coeff_matrices, batch_gradients)

    out: dict = {
        "seed": seed,
        "image_shape": list(image_shape),
        "d": d,
        "dim_g": DIM_G,
        "paths": {},
        "grid_sample": None,
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
                path_out["l3"][inverter] = metrics
                logger.info(
                    "seed=%d %s %s snap_mse=%.4f input_mse=%.4f psnr=%.2f",
                    seed,
                    path,
                    inverter,
                    metrics["snapshot_mse"],
                    metrics["input_mse"],
                    metrics["input_psnr_db"],
                )
                if seed == SEEDS[1]:
                    if "grid_panels" not in out:
                        out["grid_panels"] = []
                    out["grid_panels"].append(
                        {
                            "label": key,
                            "x_true": x_true[:6].tolist(),
                            "x_rec": x_rec[:6].tolist(),
                            "image_shape": list(image_shape),
                        }
                    )
            except Exception as exc:
                path_out["l3"][inverter] = {"error": str(exc)}
                logger.exception("L3 failed seed=%d path=%s inv=%s", seed, path, inverter)

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
) -> None:
    """Rows = methods; cols = sample index (fixed order, no Hungarian for display)."""
    n_cols = panels[0][1].shape[0] if panels else 0
    n_rows = len(panels)
    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(1.6 * n_cols, 2.2 * n_rows))
    if n_cols == 1:
        axes = np.atleast_2d(axes)
    for row_i, (label, x_true, x_rec, shape) in enumerate(panels):
        side = shape[0]
        for j in range(n_cols):
            axes[row_i * 2, j].imshow(x_true[j].reshape(side, side), cmap="gray", vmin=0, vmax=1)
            axes[row_i * 2, j].axis("off")
            axes[row_i * 2 + 1, j].imshow(
                x_rec[j].reshape(side, side), cmap="gray", vmin=0, vmax=1
            )
            axes[row_i * 2 + 1, j].axis("off")
        axes[row_i * 2, 0].set_ylabel(f"{label}\norig", fontsize=7)
        axes[row_i * 2 + 1, 0].set_ylabel("rec", fontsize=7)
    plt.suptitle(
        f"Round-01 MNIST {RESIZE}x{RESIZE} recon (fixed index; seeds aggregated in JSON)",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_grid_from_per_seed(per_seed: list[dict], display_seed: int) -> None:
    rec = next((r for r in per_seed if r["seed"] == display_seed), None)
    if not rec or not rec.get("grid_panels"):
        return
    order = [
        "shard_oracle__shard_l3",
        "shard_oracle__joli_l3",
        "lasa_qterm_T1p__joli_l3",
        "lasa_qterm_T1b__joli_l3",
    ]
    by_label = {p["label"]: p for p in rec["grid_panels"]}
    panels: list[tuple[str, np.ndarray, np.ndarray, tuple]] = []
    for label in order:
        if label not in by_label:
            continue
        p = by_label[label]
        shape = tuple(p["image_shape"])
        panels.append(
            (label, np.asarray(p["x_true"]), np.asarray(p["x_rec"]), shape)
        )
    if panels:
        plot_recon_grid(panels, ARTIFACTS / "round01_recon_grid.png")


def main() -> None:
    logger = setup_logging()
    device = pick_device()
    logger.info(
        "Round-01 MNIST E2E recon: resize=%d dim_g=%d N=%d seeds=%s device=%s qterm=%s",
        RESIZE,
        DIM_G,
        N_SAMPLES,
        SEEDS,
        device,
        QTERM_CODE,
    )

    per_seed = [run_single_seed(s, logger, device) for s in SEEDS]

    summary: dict = {}
    for path in SNAPSHOT_PATHS:
        for inverter in L3_INVERTERS:
            key = f"{path}__{inverter}"
            summary[key] = aggregate_method(per_seed, path, inverter)

    # Best path by mean input MSE (lower better)
    ranked = sorted(
        [
            (k, v["input_mse_mean"])
            for k, v in summary.items()
            if v.get("n_ok", 0) == len(SEEDS)
        ],
        key=lambda x: x[1],
    )
    best_key, best_mse = ranked[0] if ranked else ("none", float("nan"))
    best_psnr = summary[best_key].get("input_psnr_db_mean", float("nan"))

    targets = {
        "input_mse": gap_report(best_mse, TARGET_INPUT_MSE, higher_is_better=False),
        "psnr_db": gap_report(best_psnr, TARGET_PSNR_DB, higher_is_better=True),
    }

    grid_path = ARTIFACTS / "round01_recon_grid.png"
    save_grid_from_per_seed(per_seed, SEEDS[1])
    if not grid_path.exists():
        logger.warning("Grid PNG missing after save_grid_from_per_seed")

    # Drop bulky tensors from JSON (grid is in PNG).
    per_seed_json = []
    for r in per_seed:
        rj = {k: v for k, v in r.items() if k != "grid_panels"}
        per_seed_json.append(rj)

    payload = {
        "benchmark": "round01_qfl_vqc_mnist_recon",
        "method_name_stack": "SurrogateQFL + SHARD/LASA-QTERM + L3 (shard/joli)",
        "mnist_choice": {
            "resolution": RESIZE,
            "rationale": (
                "8x8 via FederatedDataLoader for Round-1 tractability (d=64); "
                "28x28 deferred to Round 2 after snapshot/L3 tuning."
            ),
            "loader": "shard_sim.FederatedDataLoader",
            "data_root": str(DATA),
        },
        "simulator": {
            "dim_g": DIM_G,
            "n_samples": N_SAMPLES,
            "batch_size": BATCH_SIZE,
            "n_epochs": N_EPOCHS,
            "noise_level": NOISE_LEVEL,
            "shard_max_iter": SHARD_MAX_ITER,
            "partial_rows_t1p": PARTIAL_ROWS,
            "seeds": SEEDS,
            "device": device,
        },
        "targets": {
            "input_mse_max": TARGET_INPUT_MSE,
            "psnr_min_db": TARGET_PSNR_DB,
            "snapshot_mse_max_weak": TARGET_SNAPSHOT_MSE_WEAK,
        },
        "per_seed": per_seed_json,
        "grid_png": str(grid_path),
        "summary": summary,
        "headline": {
            "best_path": best_key,
            "best_input_mse_mean": best_mse,
            "best_input_psnr_db_mean": best_psnr,
            "gap_to_targets": targets,
            "ranking_by_input_mse": ranked,
        },
        "honest_verdict": (
            f"Best path {best_key}: input MSE {best_mse:.4f} "
            f"(target ≤{TARGET_INPUT_MSE}), PSNR {best_psnr:.2f} dB "
            f"(target ≥{TARGET_PSNR_DB}). "
            + (
                "Targets not met in Round 1."
                if best_mse > TARGET_INPUT_MSE or best_psnr < TARGET_PSNR_DB
                else "Targets met (unexpected at 8x8 Round 1)."
            )
        ),
    }

    out_json = ARTIFACTS / "round01_metrics.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_json)
    logger.info(
        "HEADLINE best=%s input_mse=%.4f psnr=%.2f gap_mse=%.4f gap_psnr=%.2f",
        best_key,
        best_mse,
        best_psnr,
        targets["input_mse"]["gap"],
        targets["psnr_db"]["gap"],
    )


if __name__ == "__main__":
    main()
