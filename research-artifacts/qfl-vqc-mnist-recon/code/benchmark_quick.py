#!/usr/bin/env python3
"""Unified fast-iteration benchmark (default QFL_QUICK=1).

Produces ``artifacts/quick_metrics.json`` in minutes for scientist/supervisor loops.
Set ``QFL_FULL=1`` for acceptance-grade 28×28 run (hours).

Usage:
  QFL_QUICK=1 python3 code/benchmark_quick.py
  QFL_FULL=1 python3 code/benchmark_quick.py   # slow
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path

RUN_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = RUN_ROOT.parent.parent
sys.path.insert(0, str(RUN_ROOT / "code"))
os.chdir(WORKSPACE)

from quick_config import estimate_benchmark_seconds, get_run_config, is_quick_mode  # noqa: E402

_r04_spec = importlib.util.spec_from_file_location(
    "r04", RUN_ROOT / "code" / "benchmark_round04.py"
)
r04 = importlib.util.module_from_spec(_r04_spec)
assert _r04_spec.loader is not None
_r04_spec.loader.exec_module(r04)

ARTIFACTS = RUN_ROOT / "artifacts"
LOGS = RUN_ROOT / "logs"


def setup_logging() -> logging.Logger:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOGS / "experiment_quick.log", mode="w"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return logging.getLogger("benchmark_quick")


def main() -> None:
    logger = setup_logging()
    cfg = get_run_config(research_round=99)
    est = estimate_benchmark_seconds(cfg)
    logger.info(
        "MODE=%s (%s) resize=%d N=%d seeds=%s dim_g=%s paths=%s est~%.0fs",
        cfg.mode,
        cfg.estimate_label,
        cfg.resize,
        cfg.n_samples,
        cfg.seeds,
        cfg.dim_g_list,
        cfg.snapshot_paths,
        est,
    )
    if is_quick_mode():
        logger.info("Fast iteration ON. Full MNIST: QFL_QUICK=0 QFL_FULL=1")

    t0 = time.perf_counter()
    device = r04.pick_device() if hasattr(r04, "pick_device") else "cpu"
    try:
        tv = r04.load_r3_tv_lbfgs()
    except Exception:
        tv = 0.005

    # Patch round04 knobs via run_single_seed parameters
    all_results: list[dict] = []
    for dim_g in cfg.dim_g_list:
        logger.info("=== dim_g=%d resize=%d ===", dim_g, cfg.resize)
        for seed in cfg.seeds:
            rec = run_single_seed_quick(
                seed,
                dim_g,
                cfg,
                logger,
                device,
                tv,
                capture_grid=(seed == cfg.grid_seed and dim_g == cfg.dim_g_list[0]),
            )
            all_results.append(rec)

    # Aggregate pass rates
    passes = []
    for rec in all_results:
        for path, pdata in rec.get("paths", {}).items():
            for inv, m in pdata.get("l3", {}).items():
                if isinstance(m, dict) and m.get("targets", {}).get("both_ok"):
                    passes.append(
                        {
                            "seed": rec["seed"],
                            "dim_g": rec["dim_g"],
                            "path": path,
                            "inverter": inv,
                        }
                    )

    payload = {
        "benchmark": "benchmark_quick",
        "mode": cfg.mode,
        "quick_env": is_quick_mode(),
        "config": {
            "resize": cfg.resize,
            "n_samples": cfg.n_samples,
            "n_epochs": cfg.n_epochs,
            "seeds": list(cfg.seeds),
            "dim_g_list": list(cfg.dim_g_list),
            "snapshot_paths": list(cfg.snapshot_paths),
            "l3_profile": cfg.l3_profile.__dict__,
        },
        "runtime_sec": time.perf_counter() - t0,
        "per_seed_results": all_results,
        "any_pass_both_targets": len(passes) > 0,
        "pass_cells": passes,
        "targets": {
            "input_mse_max": 0.05,
            "psnr_min_db": 18.0,
        },
    }
    out = ARTIFACTS / ("quick_metrics.json" if cfg.mode == "quick" else "full_metrics.json")
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s passes=%d runtime=%.1fs", out, len(passes), payload["runtime_sec"])


def run_single_seed_quick(
    seed: int,
    dim_g: int,
    cfg,
    logger: logging.Logger,
    device: str,
    tv_lbfgs: float,
    *,
    capture_grid: bool,
) -> dict:
    """Minimal E2E using round04 helpers with quick dimensions."""
    import numpy as np
    import torch
    from shard_sim.data_loader import FederatedDataLoader
    from shard_sim.surrogate_model import SurrogateQFL

    # Round-04 helpers use module-level N_SAMPLES; patch for quick N.
    r04.N_SAMPLES = cfg.n_samples
    r04.BATCH_SIZE = cfg.batch_size
    r04.N_EPOCHS = cfg.n_epochs
    r04.PARTIAL_ROWS = cfg.partial_rows
    r04.PARTIAL_MAX_ITER = max(30, cfg.partial_rows * 8)

    torch.manual_seed(seed)
    loader = FederatedDataLoader(
        dataset="mnist",
        n_samples=cfg.n_samples,
        batch_size=cfg.batch_size,
        seed=seed,
        resize=cfg.resize,
    )
    images = loader.images.float()
    d = images.shape[1]
    surrogate = SurrogateQFL(
        input_dim=d,
        dim_g=dim_g,
        n_params=dim_g,
        noise_level=0.01,
        seed=seed,
    )

    coeff_matrices = []
    batch_gradients = []
    epoch_batches_list = []
    b1_coeff = []
    b1_grads = []

    for e in range(cfg.n_epochs):
        epoch_batches = loader.get_epoch_batches()
        epoch_batches_list.append(epoch_batches)
        a_e, grads_e = surrogate.compute_batch_gradients(images, epoch_batches)
        coeff_matrices.append(a_e)
        batch_gradients.append(grads_e)
        a_b1, g_b1 = r04.simulate_b1_epoch_gradients(images, surrogate, seed, e)
        b1_coeff.append(a_b1)
        b1_grads.append(g_b1)

    with torch.no_grad():
        true_snapshots = surrogate.encode(images).numpy().astype(np.float64)
    x_true = images.numpy().astype(np.float64)

    attacker = r04.ShardAttackerR02(
        dim_g=dim_g,
        n_samples=cfg.n_samples,
        batch_size=cfg.batch_size,
        max_iter=cfg.shard_max_iter,
        tol=1e-9,
        random_seed=seed,
    )
    e_bar = attacker.level1_mean_recovery(coeff_matrices, batch_gradients)
    h_full, _ = r04.build_oracle_incidence(epoch_batches_list, cfg.n_samples)

    out: dict = {
        "seed": seed,
        "dim_g": dim_g,
        "resize": cfg.resize,
        "d": d,
        "paths": {},
    }

    for path in cfg.snapshot_paths:
        t_snap = time.perf_counter()
        s_rec, snap_meta = r04.recover_snapshots(
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

        snap_mse = r04.hungarian_snapshot_mse(s_rec, true_snapshots)[0]
        path_out = {
            "snapshot_recovery": {
                "snapshot_mse": snap_mse,
                "runtime_sec": time.perf_counter() - t_snap,
                **snap_meta,
            },
            "l3": {},
        }

        prof = cfg.l3_profile
        from l3_budget import shard_n_batch_capped
        from joli_invert import joli_invert_single

        n_batch = shard_n_batch_capped(d, dim_g, prof.n_batch_cap)
        for adam in prof.adam_grid:
            logger.info(
                "L3 seed=%d path=%s adam=%d (N=%d d=%d)",
                seed,
                path,
                adam,
                cfg.n_samples,
                d,
            )
            t_l3 = time.perf_counter()
            x_rec = np.empty((cfg.n_samples, d), dtype=np.float64)
            for i in range(cfg.n_samples):
                t_img = time.perf_counter()
                x_rec[i], _ = joli_invert_single(
                    s_rec[i],
                    surrogate,
                    n_batch=n_batch,
                    adam_steps=prof.adam_steps,
                    lbfgs_iter=prof.lbfgs_iter,
                    tv_adam=0.0,
                    tv_lbfgs=tv_lbfgs if dim_g < d else 0.0,
                    seed=42 + i,
                    device=device,
                )
                logger.info(
                    "  img %d/%d %.2fs",
                    i + 1,
                    cfg.n_samples,
                    time.perf_counter() - t_img,
                )
            metrics = r04.evaluate_reconstruction(x_rec, x_true, s_rec, true_snapshots)
            metrics["l3_runtime_sec"] = time.perf_counter() - t_l3
            metrics["targets"] = r04.meets_targets(metrics)
            path_out["l3"]["joli"] = metrics
            logger.info(
                "seed=%d %s snap=%.4f mse=%.4f psnr=%.1f pass=%s",
                seed,
                path,
                snap_mse,
                metrics["input_mse"],
                metrics["input_psnr_db"],
                metrics["targets"]["both_ok"],
            )

        out["paths"][path] = path_out

    return out


if __name__ == "__main__":
    main()
