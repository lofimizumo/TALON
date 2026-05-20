#!/usr/bin/env python3
"""Round-14 SHARD vs TANGO-MB tier cross (synthetic, no MNIST download).

Runs vendored ``ShardAttacker.level3_invert`` on a linear snapshot task with
**intermediate** batch gradients (SHARD Tier 3). Runs TANGO-MB on the TALON
linear-head simulator with **terminal-only** observations (Tier 1).

Metrics are not directly comparable (individual input MSE vs class prototype MSE);
this module records an honest tier table for ``require_baseline_comparison_vs_shard``.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

RUN_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = RUN_ROOT / "vendor"

_spec = importlib.util.spec_from_file_location(
    "benchmark_round09", RUN_ROOT / "code" / "benchmark_round09.py"
)
_r09 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _r09
assert _spec.loader is not None
_spec.loader.exec_module(_r09)

_spec13 = importlib.util.spec_from_file_location(
    "benchmark_round13", RUN_ROOT / "code" / "benchmark_round13.py"
)
_r13 = importlib.util.module_from_spec(_spec13)
sys.modules[_spec13.name] = _r13
assert _spec13.loader is not None
_spec13.loader.exec_module(_r13)

SimConfig = _r09.SimConfig
make_dataset = _r09.make_dataset
make_bias_probes = _r09.make_bias_probes
observe_terminal_updates = _r09.observe_terminal_updates
evaluate_method = _r09.evaluate_method
tango_mb_estimate_sums = _r13.tango_mb_estimate_sums


def _shard_modules():
    if str(VENDOR_ROOT) not in sys.path:
        sys.path.insert(0, str(VENDOR_ROOT))
    from shard_sim.attacker import ShardAttacker
    from shard_sim.metrics import compute_matching_accuracy, compute_reconstruction_mse
    from shard_sim.surrogate_model import SurrogateQFL

    return ShardAttacker, compute_matching_accuracy, compute_reconstruction_mse, SurrogateQFL


def run_shard_synthetic_cross(
    *,
    n_samples: int = 4,
    batch_size: int = 2,
    n_epochs: int = 3,
    dim_g: int = 24,
    input_dim: int = 16,
    n_params: int = 24,
    max_iter: int = 8,
    seed: int = 42,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """SHARD L1–L3 on synthetic uniform images + intermediate batch gradients."""
    log = logger or logging.getLogger("shard_cross")
    ShardAttacker, compute_matching_accuracy, compute_reconstruction_mse, SurrogateQFL = (
        _shard_modules()
    )

    if n_samples % batch_size != 0:
        raise ValueError("n_samples must divide batch_size")

    rng = np.random.default_rng(seed)
    images = torch.rand(n_samples, input_dim)
    surrogate = SurrogateQFL(
        input_dim=input_dim,
        dim_g=dim_g,
        n_params=n_params,
        noise_level=0.0,
        seed=seed,
    )
    coeff_matrices: list[np.ndarray] = []
    batch_gradients: list[list[np.ndarray]] = []
    k_batches = n_samples // batch_size

    for _ in range(n_epochs):
        perm = rng.permutation(n_samples)
        epoch_batches = [
            perm[b * batch_size : (b + 1) * batch_size].tolist()
            for b in range(k_batches)
        ]
        a_e, grads = surrogate.compute_batch_gradients(images, epoch_batches)
        coeff_matrices.append(a_e)
        batch_gradients.append(grads)

    attacker = ShardAttacker(
        dim_g=dim_g,
        n_samples=n_samples,
        batch_size=batch_size,
        max_iter=max_iter,
        random_seed=seed,
    )
    with torch.no_grad():
        true_snapshots = surrogate.encode(images).numpy()

    t0 = time.time()
    e_bar = attacker.level1_mean_recovery(coeff_matrices, batch_gradients)
    true_mean = true_snapshots.mean(axis=0)
    mean_rel_error = float(
        np.linalg.norm(e_bar - true_mean) / max(np.linalg.norm(true_mean), 1e-12)
    )

    s_rec = attacker.level2_disaggregate(
        e_bar, coeff_matrices, batch_gradients, true_snapshots
    )
    matching_acc, assignment = compute_matching_accuracy(s_rec, true_snapshots)

    recovered = attacker.level3_invert(s_rec, surrogate, device="cpu")
    true_np = images.numpy()
    recon_mse = float(compute_reconstruction_mse(recovered, true_np, assignment))
    shard_runtime = time.time() - t0

    log.info(
        "SHARD synthetic: mean_rel_err=%.4f match_acc=%.2f recon_mse=%.6f (%.2fs)",
        mean_rel_error,
        matching_acc,
        recon_mse,
        shard_runtime,
    )

    return {
        "observation_tier": "tier3_intermediate_batch_gradients",
        "method": "shard_level1_level2_level3_invert",
        "code_path": "vendor/shard_sim/attacker.py level3_invert",
        "n_samples": n_samples,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "dim_g": dim_g,
        "input_dim": input_dim,
        "mean_snapshot_rel_error": mean_rel_error,
        "level2_matching_accuracy": float(matching_acc),
        "level3_reconstruction_mse": recon_mse,
        "target": "individual_inputs",
        "runtime_seconds": shard_runtime,
        "terminal_only_feasible": False,
        "note": (
            "Requires all intermediate batch gradients per epoch; "
            "cannot run from terminal deltas alone."
        ),
    }


def run_tango_mb_terminal_cross(
    *,
    seed: int = 42,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """TANGO-MB on minibatch_sgd primary settings (terminal-only)."""
    log = logger or logging.getLogger("shard_cross")
    cfg = SimConfig(use_minibatch=True, minibatch_size=8, local_steps=6)
    x, labels, _ = make_dataset(cfg, seed)
    true_counts = np.asarray(cfg.class_counts, dtype=np.float64)
    true_sums = np.vstack([x[labels == c].sum(axis=0) for c in range(cfg.n_classes)])
    true_proto = true_sums / true_counts[:, None]

    n_rounds = max(cfg.probe_rounds)
    biases = make_bias_probes(cfg, seed, n_rounds, passive=False)
    dw, db, probs = observe_terminal_updates(x, labels, biases, cfg, seed)
    est_proto, _, est_counts = tango_mb_estimate_sums(dw, db, probs, cfg)
    metrics = evaluate_method(
        "tango_mb",
        est_proto,
        est_counts,
        true_proto,
        true_counts,
        true_sums,
        x,
        labels,
        cfg,
    )
    log.info(
        "TANGO-MB terminal-only seed=%s proto_mse=%.6f count_mae=%.4f",
        seed,
        metrics["prototype_mse"],
        metrics["count_mae"],
    )
    return {
        "observation_tier": "tier1_terminal_active_probes",
        "method": "tango_mb",
        "code_path": "code/benchmark_round13.py tango_mb_estimate_sums",
        "scenario": "minibatch_sgd",
        "effective_gradient_steps": float(cfg.local_steps * cfg.n_samples / cfg.minibatch_size),
        "prototype_mse": float(metrics["prototype_mse"]),
        "count_mae": float(metrics["count_mae"]),
        "target": "class_prototypes_and_counts",
        "terminal_only": True,
        "observed_intermediate_batch_gradients": False,
        "note": (
            "Lemma MB-A scope: W0=0 linear head, PyTorch 1/B normalization, "
            "no batch-order oracle."
        ),
    }


def run_cross(logger: logging.Logger | None = None) -> dict[str, Any]:
    """Full tier comparison for Round 14 metrics JSON."""
    log = logger or logging.getLogger("shard_cross")
    shard = run_shard_synthetic_cross(logger=log)
    tango_rows = [
        run_tango_mb_terminal_cross(seed=s, logger=log)
        for s in (3, 7, 11, 19, 23, 29, 31, 37)
    ]
    proto_mses = [r["prototype_mse"] for r in tango_rows]
    count_maes = [r["count_mae"] for r in tango_rows]

    return {
        "comparison_type": "observation_tier_cross_not_same_metric",
        "shard_intermediate": shard,
        "tango_terminal_minibatch_sgd": {
            "seeds": len(tango_rows),
            "prototype_mse_mean": float(np.mean(proto_mses)),
            "prototype_mse_median": float(np.median(proto_mses)),
            "prototype_mse_iqr": float(
                np.percentile(proto_mses, 75) - np.percentile(proto_mses, 25)
            ),
            "count_mae_mean": float(np.mean(count_maes)),
            "count_mae_median": float(np.median(count_maes)),
            "per_seed": tango_rows,
        },
        "terminal_only_shard_level3": {
            "status": "not_applicable",
            "reason": (
                "level3_invert requires recovered per-sample snapshots from "
                "Level-2 disaggregation on intermediate batch-gradient rows; "
                "terminal-only FL logs do not supply A^(e) @ batch_mean_snapshot."
            ),
        },
        "interpretation": (
            "SHARD demonstrates individual-level reconstruction under Tier-3 "
            "observations (synthetic cross: level3 reconstruction MSE reported). "
            "TANGO-MB demonstrates class-prototype recovery under Tier-1 terminal "
            "observations. Neither subsumes the other; Phase-2 scoped ACCEPT "
            "claims aggregate leakage, not SHARD-equivalent sample recovery."
        ),
        "require_baseline_comparison_vs_shard": "addressed_via_tier_table",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_cross()
    import json

    print(json.dumps(result, indent=2))
