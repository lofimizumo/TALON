#!/usr/bin/env python3
"""Round-03: L3 hyperparameter grid on 28×28 T1p @ dim_g=100 (JOLI + LAPIN).

Experiment-only round (no supervisor review):
  - L3 grid: adam_steps ∈ {2000, 4000, 6000}, n_batch=max, lbfgs=1000
  - Warm-start: encoding-aware lstsq seeds + random fill to n_batch max
  - JOLI (tv_lbfgs=0.005) + LAPIN on lasa_qterm_T1p snapshots
  - SHARD oracle fix: level2_disaggregate_b1 if present, else B=1 oracle track
  - Per-seed pass rates; round03 recon grids at best adam_steps per inverter
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

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

from joli_invert import joli_invert_single, shard_n_batch  # noqa: E402
from lapin_invert import lapin_invert  # noqa: E402
from qterm_attack import QtermAttack, QtermConfig, QtermTier  # noqa: E402
from shard_sim.attacker import ShardAttacker  # noqa: E402
from shard_sim.data_loader import FederatedDataLoader  # noqa: E402
from shard_sim.metrics import (  # noqa: E402
    compute_matching_accuracy,
    compute_reconstruction_mse,
)
from shard_sim.surrogate_model import SurrogateQFL  # noqa: E402
from terminal_attacks import graph_term_map  # noqa: E402

from l3_budget import L3Profile, joli_profile, shard_n_batch_capped  # noqa: E402

# --- Round-03 knobs ---
RESIZE = 28
DIM_G = 100
SEEDS = [3, 7, 11]
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

ADAM_STEPS_GRID = (2000, 4000, 6000)
LBFGS_ITER = 1000
TV_LBFGS = 0.005  # parent Pareto operating point (dim_g=100 compressive)
L3_INVERTERS = ("joli_l3", "lapin_l3")
L3_PATH = "lasa_qterm_T1p"
GRID_SEED = 7
GRID_N_PANELS = 6


class ShardAttackerR03(ShardAttacker):
    """SHARD L2 with graph level-1 anchor + optional B=1 oracle disaggregation."""

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
        frob_change = np.inf

        for t in range(1, self.max_iter + 1):
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
            if frob_change < self.tol:
                break
        else:
            warnings.warn(
                f"SHARD did not converge within {self.max_iter} iterations "
                f"(final ||ΔS||_F = {frob_change:.2e}).",
                stacklevel=2,
            )
        return best_S

    def level2_disaggregate_b1(
        self,
        e_bar_b1: np.ndarray,
        b1_coeff: list[np.ndarray],
        b1_grads: list[list[np.ndarray]],
        true_snapshots: np.ndarray | None = None,
    ) -> np.ndarray:
        """B=1 oracle track: one client per batch → direct snapshot read-off."""
        b1_attacker = ShardAttackerR03(
            dim_g=self.dim_g,
            n_samples=self.n_samples,
            batch_size=1,
            max_iter=self.max_iter,
            tol=self.tol,
            graph_lambda=self.graph_lambda,
            spread_scale=self.spread_scale,
        )
        return b1_attacker.level2_disaggregate(
            e_bar_b1, b1_coeff, b1_grads, true_snapshots
        )


def setup_logging() -> logging.Logger:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    logging.getLogger("shard_sim.attacker").setLevel(logging.WARNING)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOGS / "experiment_round03.log", mode="w"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return logging.getLogger("mnist_recon_r03")


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def hungarian_snapshot_mse(recovered: np.ndarray, truth: np.ndarray) -> float:
    r_sq = np.sum(recovered**2, axis=1, keepdims=True)
    t_sq = np.sum(truth**2, axis=1, keepdims=True)
    dist = r_sq + t_sq.T - 2.0 * recovered @ truth.T
    np.maximum(dist, 0.0, out=dist)
    row, col = linear_sum_assignment(dist)
    return float(np.mean((recovered[row] - truth[col]) ** 2))


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


def n_batch_for_l3(d: int, dim_g: int) -> int:
    """Max parallel Adam starts (lstsq encoding seeds + uniform random fill)."""
    return shard_n_batch(d, dim_g)


def recover_shard_oracle(
    attacker: ShardAttackerR03,
    *,
    e_bar: np.ndarray,
    coeff_matrices: list[np.ndarray],
    batch_gradients: list[list[np.ndarray]],
    b1_coeff: list[np.ndarray],
    b1_grads: list[list[np.ndarray]],
    true_snapshots: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """SHARD oracle: prefer level2_disaggregate_b1, else B=1 track."""
    if hasattr(attacker, "level2_disaggregate_b1"):
        e_bar_b1 = attacker.level1_mean_recovery(b1_coeff, b1_grads)
        s = attacker.level2_disaggregate_b1(
            e_bar_b1, b1_coeff, b1_grads, true_snapshots
        )
        meta = {
            "l2_method": "level2_disaggregate_b1",
            "batch_size_track": 1,
            "observed_intermediate_batch_gradients": sum(len(g) for g in b1_grads),
        }
    else:
        b1_attacker = ShardAttackerR03(
            dim_g=attacker.dim_g,
            n_samples=attacker.n_samples,
            batch_size=1,
            max_iter=attacker.max_iter,
            tol=attacker.tol,
        )
        e_bar_b1 = b1_attacker.level1_mean_recovery(b1_coeff, b1_grads)
        s = b1_attacker.level2_disaggregate(
            e_bar_b1, b1_coeff, b1_grads, true_snapshots
        )
        meta = {
            "l2_method": "b1_oracle_fallback",
            "batch_size_track": 1,
            "observed_intermediate_batch_gradients": sum(len(g) for g in b1_grads),
        }
    meta["observed_terminal_gradient_rows"] = 0
    return s, meta


def recover_t1p(
    *,
    e_bar: np.ndarray,
    coeff_matrices: list[np.ndarray],
    batch_gradients: list[list[np.ndarray]],
    seed: int,
    dim_g: int,
) -> tuple[np.ndarray, dict]:
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


def run_joli_l3(
    snapshots: np.ndarray,
    surrogate: SurrogateQFL,
    *,
    adam_steps: int,
    lbfgs_iter: int,
    n_batch: int,
    device: str,
    logger: logging.Logger | None = None,
) -> np.ndarray:
    dev = torch.device(device)
    N = snapshots.shape[0]
    d = surrogate.W_enc.shape[1]
    out = np.empty((N, d), dtype=np.float64)
    for i in range(N):
        t0 = time.perf_counter()
        out[i], _ = joli_invert_single(
            snapshots[i],
            surrogate,
            n_batch=n_batch,
            adam_steps=adam_steps,
            lbfgs_iter=lbfgs_iter,
            tv_adam=0.0,
            tv_lbfgs=TV_LBFGS,
            seed=42 + i,
            device=dev,
        )
        if logger is not None:
            logger.info(
                "  joli image %d/%d done %.1fs",
                i + 1,
                N,
                time.perf_counter() - t0,
            )
    return out


def run_l3(
    snapshots: np.ndarray,
    surrogate: SurrogateQFL,
    inverter: str,
    *,
    adam_steps: int,
    n_batch: int,
    device: str,
    lbfgs_iter: int,
    logger: logging.Logger | None = None,
) -> np.ndarray:
    if inverter == "joli_l3":
        return run_joli_l3(
            snapshots,
            surrogate,
            adam_steps=adam_steps,
            lbfgs_iter=lbfgs_iter,
            n_batch=n_batch,
            device=device,
            logger=logger,
        )
    if inverter == "lapin_l3":
        return lapin_invert(snapshots, surrogate)
    raise ValueError(inverter)


def evaluate_reconstruction(
    x_rec: np.ndarray,
    x_true: np.ndarray,
    s_rec: np.ndarray,
    s_true: np.ndarray,
) -> dict:
    r_sq = np.sum(s_rec**2, axis=1, keepdims=True)
    t_sq = np.sum(s_true**2, axis=1, keepdims=True)
    dist = r_sq + t_sq.T - 2.0 * s_rec @ s_true.T
    np.maximum(dist, 0.0, out=dist)
    _, snap_assign = linear_sum_assignment(dist)
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
    logger: logging.Logger,
    device: str,
    *,
    capture_grid: bool,
    l3_profile: L3Profile | None = None,
) -> dict:
    profile = l3_profile or joli_profile(RESIZE * RESIZE, research_round=3)
    adam_grid = profile.adam_grid
    lbfgs_iter = profile.lbfgs_iter
    inverters: tuple[str, ...] = ("joli_l3",)
    if profile.include_lapin and (
        profile.lapin_only_if_d_le is None
        or (RESIZE * RESIZE) <= profile.lapin_only_if_d_le
    ):
        inverters = ("joli_l3", "lapin_l3")
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

    attacker = ShardAttackerR03(
        dim_g=DIM_G,
        n_samples=N_SAMPLES,
        batch_size=BATCH_SIZE,
        max_iter=SHARD_MAX_ITER,
        tol=SHARD_TOL,
        random_seed=seed,
    )
    e_bar = attacker.level1_mean_recovery(coeff_matrices, batch_gradients)

    n_batch_l3 = shard_n_batch_capped(d, DIM_G, profile.n_batch_cap)

    t_oracle = time.perf_counter()
    s_oracle, oracle_meta = recover_shard_oracle(
        attacker,
        e_bar=e_bar,
        coeff_matrices=coeff_matrices,
        batch_gradients=batch_gradients,
        b1_coeff=b1_coeff,
        b1_grads=b1_grads,
        true_snapshots=true_snapshots,
    )
    oracle_time = time.perf_counter() - t_oracle
    oracle_snap_mse = hungarian_snapshot_mse(s_oracle, true_snapshots)

    t_t1p = time.perf_counter()
    s_t1p, t1p_meta = recover_t1p(
        e_bar=e_bar,
        coeff_matrices=coeff_matrices,
        batch_gradients=batch_gradients,
        seed=seed,
        dim_g=DIM_G,
    )
    t1p_time = time.perf_counter() - t_t1p
    t1p_snap_mse = hungarian_snapshot_mse(s_t1p, true_snapshots)

    logger.info(
        "seed=%d oracle_snap_mse=%.4f (B=1 %s) t1p_snap_mse=%.4f",
        seed,
        oracle_snap_mse,
        oracle_meta.get("l2_method", "?"),
        t1p_snap_mse,
    )

    l3_grid: dict[str, dict] = {}
    grid_panels: dict[str, dict] = {}

    lapin_cache: dict | None = None
    for adam_steps in adam_grid:
        for inverter in inverters:
            if inverter == "lapin_l3":
                if lapin_cache is not None:
                    key = f"{inverter}__adam{adam_steps}"
                    l3_grid[key] = {
                        **lapin_cache,
                        "adam_steps": adam_steps,
                        "adam_independent": True,
                    }
                    continue
            key = f"{inverter}__adam{adam_steps}"
            logger.info(
                "seed=%d L3 start %s adam=%d n_batch=%d",
                seed,
                inverter,
                adam_steps,
                n_batch_l3,
            )
            t0 = time.perf_counter()
            try:
                x_rec = run_l3(
                    s_t1p,
                    surrogate,
                    inverter,
                    adam_steps=adam_steps,
                    n_batch=n_batch_l3,
                    device=device,
                    lbfgs_iter=lbfgs_iter,
                    logger=logger if profile.log_every_image else None,
                )
                l3_time = time.perf_counter() - t0
                metrics = evaluate_reconstruction(x_rec, x_true, s_t1p, true_snapshots)
                metrics["l3_runtime_sec"] = l3_time
                metrics["targets"] = meets_targets(metrics)
                cell = {
                    "inverter": inverter,
                    "adam_steps": adam_steps,
                    "lbfgs_iter": lbfgs_iter if inverter == "joli_l3" else None,
                    "n_batch": n_batch_l3 if inverter == "joli_l3" else None,
                    "n_batch_max": n_batch_l3,
                    "warm_start": "encoding_lstsq_seeds_plus_random_fill",
                    "tv_lbfgs": TV_LBFGS if inverter == "joli_l3" else None,
                    **metrics,
                }
                l3_grid[key] = cell
                if inverter == "lapin_l3":
                    lapin_cache = {k: v for k, v in cell.items() if k != "adam_steps"}
                logger.info(
                    "seed=%d %s adam=%d mse=%.4f psnr=%.2f pass=%s",
                    seed,
                    inverter,
                    adam_steps,
                    metrics["input_mse"],
                    metrics["input_psnr_db"],
                    metrics["targets"]["both_ok"],
                )
                if capture_grid:
                    grid_panels[key] = {
                        "x_true": x_true[:GRID_N_PANELS].tolist(),
                        "x_rec": x_rec[:GRID_N_PANELS].tolist(),
                        "image_shape": list(image_shape),
                    }
            except Exception as exc:
                l3_grid[key] = {"error": str(exc)}
                logger.exception(
                    "L3 failed seed=%d inv=%s adam=%d", seed, inverter, adam_steps
                )

    return {
        "seed": seed,
        "dim_g": DIM_G,
        "d": d,
        "image_shape": list(image_shape),
        "l3_budget": {
            "adam_steps_grid": list(ADAM_STEPS_GRID),
            "lbfgs_iter": LBFGS_ITER,
            "n_batch": n_batch_l3,
            "warm_start": "encoding_lstsq_seeds_plus_random_fill",
            "tv_lbfgs": TV_LBFGS,
        },
        "snapshot_recovery": {
            "shard_oracle_b1": {
                "snapshot_mse": oracle_snap_mse,
                "recovery_runtime_sec": oracle_time,
                **oracle_meta,
            },
            L3_PATH: {
                "snapshot_mse": t1p_snap_mse,
                "recovery_runtime_sec": t1p_time,
                **t1p_meta,
            },
        },
        "l3_grid_t1p": l3_grid,
        "grid_panels": grid_panels if capture_grid else None,
    }


def per_seed_pass_table(per_seed: list[dict]) -> list[dict]:
    rows = []
    for r in per_seed:
        seed = r["seed"]
        for key, cell in r["l3_grid_t1p"].items():
            if "error" in cell:
                rows.append(
                    {
                        "seed": seed,
                        "config": key,
                        "pass_both": False,
                        "pass_mse": False,
                        "pass_psnr": False,
                        "input_mse": None,
                        "input_psnr_db": None,
                        "error": cell["error"],
                    }
                )
                continue
            t = cell.get("targets", {})
            rows.append(
                {
                    "seed": seed,
                    "config": key,
                    "inverter": cell.get("inverter"),
                    "adam_steps": cell.get("adam_steps"),
                    "pass_both": t.get("both_ok", False),
                    "pass_mse": t.get("input_mse_ok", False),
                    "pass_psnr": t.get("psnr_ok", False),
                    "input_mse": cell.get("input_mse"),
                    "input_psnr_db": cell.get("input_psnr_db"),
                }
            )
    return rows


def aggregate_l3_grid(per_seed: list[dict]) -> dict[str, dict]:
    keys: set[str] = set()
    for r in per_seed:
        keys.update(r["l3_grid_t1p"].keys())
    summary: dict[str, dict] = {}
    for key in sorted(keys):
        rows = [
            r["l3_grid_t1p"][key]
            for r in per_seed
            if key in r["l3_grid_t1p"] and "input_mse" in r["l3_grid_t1p"][key]
        ]
        if not rows:
            summary[key] = {"n_ok": 0}
            continue
        n = len(rows)
        pass_both = sum(1 for x in rows if x.get("targets", {}).get("both_ok"))
        pass_mse = sum(1 for x in rows if x.get("targets", {}).get("input_mse_ok"))
        pass_psnr = sum(1 for x in rows if x.get("targets", {}).get("psnr_ok"))
        summary[key] = {
            "n_ok": n,
            "input_mse_mean": float(np.mean([x["input_mse"] for x in rows])),
            "input_mse_std": float(np.std([x["input_mse"] for x in rows], ddof=0)),
            "input_psnr_db_mean": float(np.mean([x["input_psnr_db"] for x in rows])),
            "pass_rate_both": pass_both / n,
            "pass_rate_input_mse": pass_mse / n,
            "pass_rate_psnr": pass_psnr / n,
            "per_seed_pass_both": [
                {
                    "seed": r["seed"],
                    "pass": r["l3_grid_t1p"][key].get("targets", {}).get("both_ok", False),
                    "input_mse": r["l3_grid_t1p"][key].get("input_mse"),
                }
                for r in per_seed
                if key in r["l3_grid_t1p"] and "input_mse" in r["l3_grid_t1p"][key]
            ],
        }
    return summary


def pick_best_config(summary: dict[str, dict], inverter: str) -> str | None:
    candidates = [
        (k, v)
        for k, v in summary.items()
        if k.startswith(inverter) and v.get("n_ok", 0) == len(SEEDS)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda kv: kv[1]["input_mse_mean"])[0]


def plot_recon_grid(
    panels: list[tuple[str, np.ndarray, np.ndarray, tuple[int, int]]],
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


def save_round03_grids(
    per_seed: list[dict],
    summary: dict[str, dict],
) -> dict[str, str]:
    paths: dict[str, str] = {}
    rec = next((r for r in per_seed if r["seed"] == GRID_SEED), None)
    if not rec or not rec.get("grid_panels"):
        return paths

    for inverter in L3_INVERTERS:
        best_key = pick_best_config(summary, inverter)
        if not best_key or best_key not in rec["grid_panels"]:
            continue
        p = rec["grid_panels"][best_key]
        shape = tuple(p["image_shape"])
        panels = [
            (
                f"T1p {best_key}",
                np.asarray(p["x_true"]),
                np.asarray(p["x_rec"]),
                shape,
            )
        ]
        adam = rec["l3_grid_t1p"][best_key].get("adam_steps", "?")
        out = ARTIFACTS / f"round03_recon_grid_{inverter}_adam{adam}.png"
        plot_recon_grid(
            panels,
            out,
            title=(
                f"Round-03 MNIST {RESIZE}×{RESIZE} {inverter} on T1p "
                f"(dim_g={DIM_G}, seed={GRID_SEED}, adam_steps={adam})"
            ),
        )
        paths[inverter] = str(out)

    # Combined adam_steps sweep for JOLI
    joli_panels: list[tuple[str, np.ndarray, np.ndarray, tuple[int, int]]] = []
    for adam_steps in ADAM_STEPS_GRID:
        key = f"joli_l3__adam{adam_steps}"
        if key not in rec.get("grid_panels", {}):
            continue
        p = rec["grid_panels"][key]
        shape = tuple(p["image_shape"])
        joli_panels.append(
            (
                f"adam={adam_steps}",
                np.asarray(p["x_true"]),
                np.asarray(p["x_rec"]),
                shape,
            )
        )
    if joli_panels:
        out = ARTIFACTS / "round03_recon_grid_joli_adam_sweep.png"
        plot_recon_grid(
            joli_panels,
            out,
            title=(
                f"Round-03 JOLI L3 adam sweep on T1p "
                f"({RESIZE}×{RESIZE}, dim_g={DIM_G}, seed={GRID_SEED})"
            ),
        )
        paths["joli_adam_sweep"] = str(out)

    return paths


def _run_seed_worker(seed: int) -> dict:
    """Process-pool entry: one seed, dedicated log file."""
    log_path = LOGS / f"experiment_round03_seed{seed}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logger = logging.getLogger(f"mnist_recon_r03_seed{seed}")
    return run_single_seed(
        seed,
        logger,
        pick_device(),
        capture_grid=(seed == GRID_SEED),
    )


def main() -> None:
    from quick_config import is_quick_mode, patch_module_globals

    logger = setup_logging()
    qcfg = patch_module_globals(sys.modules[__name__], research_round=3)
    device = pick_device()
    d = RESIZE * RESIZE
    profile = joli_profile(d, research_round=3)
    if qcfg is not None:
        logger.info(
            "QFL_QUICK: resize=%d N=%d seeds=%s (full: QFL_QUICK=0 QFL_FULL=1)",
            qcfg.resize,
            qcfg.n_samples,
            qcfg.seeds,
        )
    logger.info(
        "Round-03 MNIST %dx%d dim_g=%d seeds=%s device=%s profile=%s "
        "adam_grid=%s lbfgs=%d n_batch_cap=%d workers=%d",
        RESIZE,
        RESIZE,
        DIM_G,
        SEEDS,
        device,
        profile.name,
        profile.adam_grid,
        profile.lbfgs_iter,
        profile.n_batch_cap,
        profile.parallel_seed_workers,
    )
    logger.info("root_cause: paper/root_cause_l3_stall.md")

    per_seed: list[dict] = []
    if profile.parallel_seed_workers <= 1:
        for s in SEEDS:
            per_seed.append(
                run_single_seed(
                    s, logger, device, capture_grid=(s == GRID_SEED), l3_profile=profile
                )
            )
            logger.info("Completed seed=%d", s)
    else:
        workers = min(len(SEEDS), profile.parallel_seed_workers)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_seed_worker, s): s for s in SEEDS}
            for fut in as_completed(futures):
                seed = futures[fut]
                per_seed.append(fut.result())
                logger.info("Completed seed=%d", seed)
    per_seed.sort(key=lambda r: SEEDS.index(r["seed"]))

    summary = aggregate_l3_grid(per_seed)
    pass_table = per_seed_pass_table(per_seed)
    grid_paths = save_round03_grids(per_seed, summary)

    best_joli = pick_best_config(summary, "joli_l3")
    best_lapin = pick_best_config(summary, "lapin_l3")

    payload: dict[str, Any] = {
        "benchmark": "round03_qfl_vqc_mnist_recon",
        "method_name_stack": "SurrogateQFL + SHARD(B=1 oracle) + LASA-QTERM T1p + L3 (joli/lapin)",
        "mnist_choice": {
            "resolution": RESIZE,
            "dim_g": DIM_G,
            "d": d,
            "loader": "shard_sim.FederatedDataLoader",
        },
        "round03_tuning": {
            "adam_steps_grid": list(ADAM_STEPS_GRID),
            "lbfgs_iter": LBFGS_ITER,
            "n_batch": n_batch_for_l3(d, DIM_G),
            "warm_start": "encoding_lstsq_seeds_plus_random_fill",
            "tv_lbfgs": TV_LBFGS,
            "shard_oracle_fix": "level2_disaggregate_b1 or B=1 fallback",
            "l3_path": L3_PATH,
            "l3_inverters": list(L3_INVERTERS),
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
        "per_seed": [{k: v for k, v in r.items() if k != "grid_panels"} for r in per_seed],
        "per_seed_pass_table": pass_table,
        "l3_grid_summary_t1p": summary,
        "grid_png": grid_paths,
        "headline": {
            "best_joli_config": best_joli,
            "best_lapin_config": best_lapin,
            "best_joli": summary.get(best_joli or "", {}),
            "best_lapin": summary.get(best_lapin or "", {}),
        },
    }

    if best_joli and best_joli in summary:
        bj = summary[best_joli]
        payload["honest_verdict"] = (
            f"Best T1p JOLI @ {best_joli}: mean input MSE {bj.get('input_mse_mean', float('nan')):.4f} "
            f"(target ≤{TARGET_INPUT_MSE}), pass_rate_both={bj.get('pass_rate_both', 0):.0%}."
        )

    out_json = ARTIFACTS / "round03_metrics.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_json)
    print(json.dumps(payload["headline"], indent=2))


if __name__ == "__main__":
    main()
