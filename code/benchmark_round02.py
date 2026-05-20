#!/usr/bin/env python3
"""Round-02 benchmark: canonical SHARD L3 vs JOLI (structure-aware TV polish).

Multi-seed evaluation; compressive dim_g < d and TV-off ablation regimes.
All metrics computed from arrays (no hardcoded results).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from scipy.optimize import linear_sum_assignment

RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "code"))
from _paths import VENDOR_ROOT  # noqa: E402

sys.path.insert(0, str(VENDOR_ROOT))

from joli_invert import joli_invert, shard_n_batch  # noqa: E402
from shard_sim.attacker import ShardAttacker  # noqa: E402
from shard_sim.surrogate_model import SurrogateQFL  # noqa: E402

ARTIFACTS = RUN_ROOT / "artifacts"
LOGS = RUN_ROOT / "logs"
DATA_DIR = RUN_ROOT / "data"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOGS / "experiment_round02.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def load_mnist_subset(n_samples: int, resize: int) -> torch.Tensor:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose(
        [
            transforms.Resize((resize, resize), antialias=True),
            transforms.ToTensor(),
        ]
    )
    ds = torchvision.datasets.MNIST(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=transform,
    )
    return torch.stack([ds[i][0].flatten() for i in range(n_samples)])


def hungarian_mse(recovered: np.ndarray, truth: np.ndarray) -> float:
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


def evaluate_method(
    x_rec: np.ndarray,
    x_true: np.ndarray,
    snapshots: np.ndarray,
    surrogate: SurrogateQFL,
) -> dict:
    with torch.no_grad():
        snap_true = surrogate.encode(torch.tensor(x_true, dtype=torch.float32)).numpy()
        snap_rec = surrogate.encode(torch.tensor(x_rec, dtype=torch.float32)).numpy()
    mse = hungarian_mse(x_rec, x_true)
    return {
        "input_mse": mse,
        "input_psnr_db": psnr_from_mse(mse),
        "snapshot_match_acc": ShardAttacker._matching_accuracy(snap_rec, snap_true),
        "mean_snapshot_residual": float(np.mean(np.sum((snap_rec - snapshots) ** 2, axis=1))),
    }


def shard_budget(d: int, dim_g: int) -> dict:
    n_batch = shard_n_batch(d, dim_g)
    return {
        "n_batch_starts": n_batch,
        "n_lstsq_seeds": 5,
        "adam_steps": 2000,
        "lbfgs_max_iter": 500,
        "api": "ShardAttacker.level3_invert",
    }


def joli_budget(d: int, dim_g: int, adam_steps: int, tv_adam: float, tv_lbfgs: float) -> dict:
    return {
        "n_batch_starts": shard_n_batch(d, dim_g),
        "n_lstsq_seeds": 5,
        "adam_steps": adam_steps,
        "lbfgs_max_iter": 500,
        "tv_adam": tv_adam,
        "tv_lbfgs": tv_lbfgs,
        "tv_prior_when_dim_g_lt_d": dim_g < d,
        "api": "joli_invert",
    }


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def run_single_config(
    *,
    n_samples: int,
    resize: int,
    dim_g: int,
    seed: int,
    adam_steps: int,
    tv_adam: float,
    tv_lbfgs: float,
    device: str,
) -> dict:
    torch.manual_seed(seed)
    images = load_mnist_subset(n_samples, resize=resize)
    d = images.shape[1]
    surrogate = SurrogateQFL(input_dim=d, dim_g=dim_g, seed=seed)
    with torch.no_grad():
        snapshots = surrogate.encode(images).numpy()
    x_true = images.numpy()

    attacker = ShardAttacker(dim_g=dim_g, n_samples=n_samples, batch_size=4)

    t0 = time.perf_counter()
    x_shard = attacker.level3_invert(snapshots, surrogate, device=device)
    t_shard = time.perf_counter() - t0

    t1 = time.perf_counter()
    x_joli = joli_invert(
        snapshots,
        surrogate,
        adam_steps=adam_steps,
        tv_adam=tv_adam,
        tv_lbfgs=tv_lbfgs,
        device=device,
    )
    t_joli = time.perf_counter() - t1

    shard_metrics = evaluate_method(x_shard, x_true, snapshots, surrogate)
    shard_metrics["runtime_sec"] = t_shard
    joli_metrics = evaluate_method(x_joli, x_true, snapshots, surrogate)
    joli_metrics["runtime_sec"] = t_joli

    return {
        "seed": seed,
        "shard_l3_canonical": shard_metrics,
        "joli": joli_metrics,
        "budget": {
            "shard": shard_budget(d, dim_g),
            "joli": joli_budget(d, dim_g, adam_steps, tv_adam, tv_lbfgs),
        },
    }


def aggregate_runs(per_seed: list[dict], key: str) -> dict:
    metrics = ["input_mse", "input_psnr_db", "snapshot_match_acc", "mean_snapshot_residual", "runtime_sec"]
    out: dict = {}
    for m in metrics:
        vals = [r[key][m] for r in per_seed]
        out[m] = float(np.mean(vals))
        out[f"{m}_std"] = float(np.std(vals, ddof=0))
    return out


def run_regime(
    name: str,
    *,
    n_samples: int,
    resize: int,
    dim_g: int,
    seeds: list[int],
    adam_steps: int,
    tv_adam: float,
    tv_lbfgs: float,
    device: str,
) -> dict:
    logger.info("=== Regime %s: resize=%d dim_g=%d seeds=%s ===", name, resize, dim_g, seeds)
    per_seed = []
    for s in seeds:
        logger.info("  seed=%d", s)
        per_seed.append(
            run_single_config(
                n_samples=n_samples,
                resize=resize,
                dim_g=dim_g,
                seed=s,
                adam_steps=adam_steps,
                tv_adam=tv_adam,
                tv_lbfgs=tv_lbfgs,
                device=device,
            )
        )
    d = per_seed[0]["budget"]["shard"]["n_batch_starts"]
    _ = d  # noqa: F841 — logged via budget block
    return {
        "name": name,
        "n_samples": n_samples,
        "resize": resize,
        "d": resize * resize,
        "dim_g": dim_g,
        "compressive_dim_g_lt_d": dim_g < resize * resize,
        "seeds": seeds,
        "per_seed": per_seed,
        "aggregate": {
            "shard_l3_canonical": aggregate_runs(per_seed, "shard_l3_canonical"),
            "joli": aggregate_runs(per_seed, "joli"),
        },
        "budget": per_seed[0]["budget"],
    }


def plot_regime_bars(regimes: list[dict], out_path: Path) -> None:
    names = [r["name"] for r in regimes]
    x = np.arange(len(names))
    w = 0.35
    shard_mse = [r["aggregate"]["shard_l3_canonical"]["input_mse"] for r in regimes]
    joli_mse = [r["aggregate"]["joli"]["input_mse"] for r in regimes]
    shard_psnr = [r["aggregate"]["shard_l3_canonical"]["input_psnr_db"] for r in regimes]
    joli_psnr = [r["aggregate"]["joli"]["input_psnr_db"] for r in regimes]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    axes[0].bar(x - w / 2, shard_mse, w, label="SHARD L3 (canonical)", color="#4c72b0")
    axes[0].bar(x + w / 2, joli_mse, w, label="JOLI", color="#55a868")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=15, ha="right")
    axes[0].set_ylabel("Input MSE (mean)")
    axes[0].legend(fontsize=8)
    axes[1].bar(x - w / 2, shard_psnr, w, label="SHARD L3", color="#4c72b0")
    axes[1].bar(x + w / 2, joli_psnr, w, label="JOLI", color="#55a868")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=15, ha="right")
    axes[1].set_ylabel("PSNR (dB, mean)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Round-02 multi-seed (canonical SHARD vs JOLI)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_reconstruction(
    regime_result: dict,
    seed_idx: int,
    out_path: Path,
    *,
    adam_steps: int,
    tv_adam: float,
    tv_lbfgs: float,
    device: str,
) -> None:
    """Use last run's arrays re-computed for one seed (stored in per_seed only as metrics)."""
    # Re-run one seed for visualization grid
    r0 = regime_result
    seed = r0["seeds"][seed_idx]
    resize = r0["resize"]
    dim_g = r0["dim_g"]
    n_samples = r0["n_samples"]
    torch.manual_seed(seed)
    images = load_mnist_subset(n_samples, resize=resize)
    d = images.shape[1]
    surrogate = SurrogateQFL(input_dim=d, dim_g=dim_g, seed=seed)
    with torch.no_grad():
        snapshots = surrogate.encode(images).numpy()
    attacker = ShardAttacker(dim_g=dim_g, n_samples=n_samples, batch_size=4)
    x_shard = attacker.level3_invert(snapshots, surrogate)
    x_joli = joli_invert(
        snapshots, surrogate, adam_steps=adam_steps, tv_adam=tv_adam, tv_lbfgs=tv_lbfgs, device=device
    )
    x_true = images.numpy()
    n_show = min(6, n_samples)
    fig, axes = plt.subplots(3, n_show, figsize=(1.6 * n_show, 4.8))
    for j in range(n_show):
        for row, arr in enumerate((x_true, x_shard, x_joli)):
            axes[row, j].imshow(arr[j].reshape(resize, resize), cmap="gray", vmin=0, vmax=1)
            axes[row, j].axis("off")
    axes[0, 0].set_ylabel("Truth", fontsize=9)
    axes[1, 0].set_ylabel("SHARD L3", fontsize=9)
    axes[2, 0].set_ylabel("JOLI", fontsize=9)
    fig.suptitle(f"{r0['name']} (seed={seed})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    seeds = [7, 11, 23]
    adam_steps = 2000  # matched to canonical ShardAttacker.level3_invert
    device = pick_device()
    logger.info("Using device: %s", device)

    regimes = [
        run_regime(
            "compressive_28x28",
            n_samples=12,
            resize=28,
            dim_g=100,
            seeds=seeds,
            adam_steps=adam_steps,
            tv_adam=0.0,
            tv_lbfgs=5e-3,
            device=device,
        ),
        run_regime(
            "reference_14x14",
            n_samples=12,
            resize=14,
            dim_g=160,
            seeds=seeds,
            adam_steps=adam_steps,
            tv_adam=0.0,
            tv_lbfgs=0.0,
            device=device,
        ),
    ]

    results = {
        "round": 2,
        "method": "JOLI",
        "baseline": "ShardAttacker.level3_invert (canonical)",
        "seeds": seeds,
        "regimes": regimes,
        "headline": {},
    }

    for r in regimes:
        agg_s = r["aggregate"]["shard_l3_canonical"]
        agg_j = r["aggregate"]["joli"]
        mse_win = agg_j["input_mse"] < agg_s["input_mse"]
        psnr_win = agg_j["input_psnr_db"] > agg_s["input_psnr_db"]
        snap_win = agg_j["snapshot_match_acc"] > agg_s["snapshot_match_acc"]
        results["headline"][r["name"]] = {
            "joli_beats_shard_mse": mse_win,
            "joli_beats_shard_psnr": psnr_win,
            "joli_beats_shard_snapshot_match": snap_win,
            "shard_mse_mean_std": f"{agg_s['input_mse']:.4f}±{agg_s['input_mse_std']:.4f}",
            "joli_mse_mean_std": f"{agg_j['input_mse']:.4f}±{agg_j['input_mse_std']:.4f}",
            "shard_psnr_mean_std": f"{agg_s['input_psnr_db']:.2f}±{agg_s['input_psnr_db_std']:.2f}",
            "joli_psnr_mean_std": f"{agg_j['input_psnr_db']:.2f}±{agg_j['input_psnr_db_std']:.2f}",
        }

    out_json = ARTIFACTS / "round02_metrics.json"
    out_json.write_text(json.dumps(results, indent=2))
    plot_regime_bars(regimes, ARTIFACTS / "round02_metrics_bar.png")
    plot_reconstruction(
        regimes[0], 0, ARTIFACTS / "round02_reconstruction_compressive.png",
        adam_steps=adam_steps, tv_adam=0.0, tv_lbfgs=5e-3, device=device,
    )
    plot_reconstruction(
        regimes[1], 0, ARTIFACTS / "round02_reconstruction_reference.png",
        adam_steps=adam_steps, tv_adam=0.0, tv_lbfgs=0.0, device=device,
    )

    logger.info("Wrote %s", out_json)
    print(json.dumps(results["headline"], indent=2))


if __name__ == "__main__":
    main()
