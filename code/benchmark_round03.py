#!/usr/bin/env python3
"""Round-03 benchmark: Pareto sweep, regime controls, multi-seed validation.

- tv_lbfgs Pareto on compressive MNIST 28x28 (dim_g=100 < d=784)
- True dim_g >= d controls (14x14 dim_g=256, 28x28 dim_g=784)
- Renamed ablation_tv_off_14x14 (still compressive; TV manually off)
- Held-out seed for lambda operating-point validation
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

PARETO_LAMBDAS = [0.0, 1e-3, 5e-3, 1e-2, 2e-2]
TUNING_SEEDS = [7, 11, 23]
HELDOUT_SEED = 31

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOGS / "experiment_round03.log", mode="w"),
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


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def run_inversion_pair(
    *,
    n_samples: int,
    resize: int,
    dim_g: int,
    seed: int,
    tv_lbfgs: float | None,
    device: str,
    adam_steps: int = 2000,
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
        tv_adam=0.0,
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
        "d": d,
        "dim_g": dim_g,
        "compressive_dim_g_lt_d": dim_g < d,
        "tv_lbfgs": tv_lbfgs,
        "shard_l3_canonical": shard_metrics,
        "joli": joli_metrics,
        "budget": {
            "n_batch_starts": shard_n_batch(d, dim_g),
            "adam_steps": adam_steps,
            "lbfgs_max_iter": 500,
            "tv_adam": 0.0,
            "tv_lbfgs": tv_lbfgs,
        },
    }


def aggregate_runs(per_seed: list[dict], key: str) -> dict:
    metrics = [
        "input_mse",
        "input_psnr_db",
        "snapshot_match_acc",
        "mean_snapshot_residual",
        "runtime_sec",
    ]
    out: dict = {}
    for m in metrics:
        vals = [r[key][m] for r in per_seed]
        out[m] = float(np.mean(vals))
        out[f"{m}_std"] = float(np.std(vals, ddof=0))
    return out


def run_pareto_sweep(
    *,
    n_samples: int,
    resize: int,
    dim_g: int,
    seeds: list[int],
    device: str,
) -> dict:
    logger.info(
        "=== Pareto sweep: resize=%d dim_g=%d seeds=%s lambdas=%s ===",
        resize,
        dim_g,
        seeds,
        PARETO_LAMBDAS,
    )
    by_lambda: dict[float, list[dict]] = {lam: [] for lam in PARETO_LAMBDAS}
    shard_by_seed: dict[int, dict] = {}

    for seed in seeds:
        logger.info("  seed=%d (SHARD baseline once)", seed)
        shard_run = run_inversion_pair(
            n_samples=n_samples,
            resize=resize,
            dim_g=dim_g,
            seed=seed,
            tv_lbfgs=0.0,
            device=device,
        )
        shard_by_seed[seed] = shard_run["shard_l3_canonical"]

        for lam in PARETO_LAMBDAS:
            logger.info("  seed=%d tv_lbfgs=%g", seed, lam)
            if lam == 0.0:
                run = shard_run
            else:
                run = run_inversion_pair(
                    n_samples=n_samples,
                    resize=resize,
                    dim_g=dim_g,
                    seed=seed,
                    tv_lbfgs=lam,
                    device=device,
                )
            entry = {
                "seed": seed,
                "tv_lbfgs": lam,
                "joli": run["joli"],
                "shard_l3_canonical": shard_by_seed[seed],
            }
            by_lambda[lam].append(entry)

    pareto_points = []
    for lam in PARETO_LAMBDAS:
        agg_j = aggregate_runs(by_lambda[lam], "joli")
        agg_s = aggregate_runs(by_lambda[lam], "shard_l3_canonical")
        pareto_points.append(
            {
                "tv_lbfgs": lam,
                "aggregate_joli": agg_j,
                "aggregate_shard": agg_s,
                "per_seed": by_lambda[lam],
            }
        )

    return {
        "name": "compressive_28x28_pareto",
        "n_samples": n_samples,
        "resize": resize,
        "d": resize * resize,
        "dim_g": dim_g,
        "compressive_dim_g_lt_d": dim_g < resize * resize,
        "seeds": seeds,
        "lambdas": PARETO_LAMBDAS,
        "pareto_points": pareto_points,
    }


def select_operating_point(pareto: dict) -> dict:
    """Pick lambda on tuning seeds: best input MSE with snapshot_acc >= 0.05."""
    candidates = []
    for pt in pareto["pareto_points"]:
        lam = pt["tv_lbfgs"]
        if lam == 0.0:
            continue
        agg = pt["aggregate_joli"]
        candidates.append(
            {
                "tv_lbfgs": lam,
                "input_mse": agg["input_mse"],
                "input_psnr_db": agg["input_psnr_db"],
                "snapshot_match_acc": agg["snapshot_match_acc"],
                "mean_snapshot_residual": agg["mean_snapshot_residual"],
            }
        )

    viable = [c for c in candidates if c["snapshot_match_acc"] >= 0.05]
    if not viable:
        viable = candidates
    best = min(viable, key=lambda c: c["input_mse"])

    shard_ref = pareto["pareto_points"][0]["aggregate_shard"]
    mse_ratio = shard_ref["input_mse"] / best["input_mse"] if best["input_mse"] > 0 else float("inf")

    return {
        "selected_tv_lbfgs": best["tv_lbfgs"],
        "criterion": "min input MSE among lambdas with snapshot_match_acc >= 0.05 on tuning seeds",
        "tuning_seeds": pareto["seeds"],
        "heldout_seed": HELDOUT_SEED,
        "tuning_metrics": best,
        "shard_reference_mse": shard_ref["input_mse"],
        "mse_improvement_factor_vs_shard": mse_ratio,
    }


def run_control_regime(
    name: str,
    *,
    n_samples: int,
    resize: int,
    dim_g: int,
    seeds: list[int],
    tv_lbfgs: float | None,
    device: str,
) -> dict:
    logger.info("=== Control %s: resize=%d dim_g=%d tv_lbfgs=%s ===", name, resize, dim_g, tv_lbfgs)
    per_seed = []
    for seed in seeds:
        logger.info("  seed=%d", seed)
        per_seed.append(
            run_inversion_pair(
                n_samples=n_samples,
                resize=resize,
                dim_g=dim_g,
                seed=seed,
                tv_lbfgs=tv_lbfgs,
                device=device,
            )
        )
    d = resize * resize
    return {
        "name": name,
        "n_samples": n_samples,
        "resize": resize,
        "d": d,
        "dim_g": dim_g,
        "compressive_dim_g_lt_d": dim_g < d,
        "overdetermined_dim_g_ge_d": dim_g >= d,
        "seeds": seeds,
        "tv_lbfgs": tv_lbfgs,
        "per_seed": per_seed,
        "aggregate": {
            "shard_l3_canonical": aggregate_runs(per_seed, "shard_l3_canonical"),
            "joli": aggregate_runs(per_seed, "joli"),
        },
    }


def max_abs_metric_diff(control: dict, metric: str) -> float:
    diffs = [
        abs(r["joli"][metric] - r["shard_l3_canonical"][metric])
        for r in control["per_seed"]
    ]
    return float(max(diffs))


def plot_pareto(pareto: dict, operating: dict, out_path: Path) -> None:
    lams = [pt["tv_lbfgs"] for pt in pareto["pareto_points"]]
    mse_j = [pt["aggregate_joli"]["input_mse"] for pt in pareto["pareto_points"]]
    mse_s = pareto["pareto_points"][0]["aggregate_shard"]["input_mse"]
    snap_acc = [pt["aggregate_joli"]["snapshot_match_acc"] for pt in pareto["pareto_points"]]
    snap_res = [pt["aggregate_joli"]["mean_snapshot_residual"] for pt in pareto["pareto_points"]]
    sel = operating["selected_tv_lbfgs"]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

    axes[0].semilogx(lams[1:], mse_j[1:], "o-", color="#55a868", label="JOLI")
    axes[0].axhline(mse_s, color="#4c72b0", ls="--", label="SHARD L3 (mean)")
    axes[0].axvline(sel, color="#c44e52", ls=":", lw=1.5, label=f"selected λ={sel:g}")
    axes[0].set_xlabel("tv_lbfgs (λ)")
    axes[0].set_ylabel("Input MSE ↓")
    axes[0].legend(fontsize=7)
    axes[0].set_title("Input fidelity")

    axes[1].semilogx(lams[1:], snap_acc[1:], "o-", color="#8172b3")
    axes[1].axhline(1.0, color="#4c72b0", ls="--", label="SHARD snap. acc.")
    axes[1].axvline(sel, color="#c44e52", ls=":", lw=1.5)
    axes[1].set_xlabel("tv_lbfgs (λ)")
    axes[1].set_ylabel("Snapshot match acc.")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(fontsize=7)
    axes[1].set_title("Snapshot consistency")

    axes[2].semilogx(lams[1:], snap_res[1:], "o-", color="#ccb974")
    axes[2].axhline(
        pareto["pareto_points"][0]["aggregate_shard"]["mean_snapshot_residual"],
        color="#4c72b0",
        ls="--",
        label="SHARD residual",
    )
    axes[2].axvline(sel, color="#c44e52", ls=":", lw=1.5)
    axes[2].set_xlabel("tv_lbfgs (λ)")
    axes[2].set_ylabel("Mean snap. residual ↓")
    axes[2].legend(fontsize=7)
    axes[2].set_title("Oracle snapshot fit")

    fig.suptitle(
        f"Pareto: compressive MNIST 28×28 (dim_g={pareto['dim_g']}, d={pareto['d']}, "
        f"N={pareto['n_samples']}, seeds={pareto['seeds']})",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pareto_tradeoff_scatter(pareto: dict, operating: dict, out_path: Path) -> None:
    """Classic trade-off: input MSE vs snapshot residual."""
    fig, ax = plt.subplots(figsize=(5, 4))
    for pt in pareto["pareto_points"]:
        lam = pt["tv_lbfgs"]
        if lam == 0.0:
            label = "JOLI λ=0 (=SHARD obj.)"
            color = "#4c72b0"
            marker = "s"
        else:
            label = f"λ={lam:g}"
            color = "#55a868"
            marker = "o"
        ax.scatter(
            pt["aggregate_joli"]["mean_snapshot_residual"],
            pt["aggregate_joli"]["input_mse"],
            c=color,
            marker=marker,
            s=80,
            label=label,
        )
    sel = operating["selected_tv_lbfgs"]
    sel_pt = next(p for p in pareto["pareto_points"] if p["tv_lbfgs"] == sel)
    ax.scatter(
        sel_pt["aggregate_joli"]["mean_snapshot_residual"],
        sel_pt["aggregate_joli"]["input_mse"],
        c="#c44e52",
        marker="*",
        s=200,
        label=f"selected λ={sel:g}",
        zorder=5,
    )
    ax.set_xlabel("Mean snapshot residual (to oracle s)")
    ax.set_ylabel("Input MSE (Hungarian)")
    ax.set_title("Input fidelity vs snapshot fit trade-off")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    logger.info("Using device: %s", device)
    n_samples = 20
    logger.info("N=%d (Round 03; paper simulator default is 12, increased for stability)", n_samples)

    pareto = run_pareto_sweep(
        n_samples=n_samples,
        resize=28,
        dim_g=100,
        seeds=TUNING_SEEDS,
        device=device,
    )
    operating = select_operating_point(pareto)
    selected_lambda = operating["selected_tv_lbfgs"]
    logger.info("Selected operating point: tv_lbfgs=%g", selected_lambda)

    logger.info("=== Held-out validation seed=%d at λ=%g ===", HELDOUT_SEED, selected_lambda)
    heldout = run_inversion_pair(
        n_samples=n_samples,
        resize=28,
        dim_g=100,
        seed=HELDOUT_SEED,
        tv_lbfgs=selected_lambda,
        device=device,
    )
    operating["heldout_validation"] = {
        "seed": HELDOUT_SEED,
        "shard_l3_canonical": heldout["shard_l3_canonical"],
        "joli": heldout["joli"],
        "joli_beats_shard_mse": heldout["joli"]["input_mse"] < heldout["shard_l3_canonical"]["input_mse"],
    }

    controls = [
        run_control_regime(
            "ablation_tv_off_14x14",
            n_samples=n_samples,
            resize=14,
            dim_g=160,
            seeds=TUNING_SEEDS,
            tv_lbfgs=0.0,
            device=device,
        ),
        run_control_regime(
            "overdetermined_14x14",
            n_samples=n_samples,
            resize=14,
            dim_g=256,
            seeds=TUNING_SEEDS,
            tv_lbfgs=None,
            device=device,
        ),
        run_control_regime(
            "square_mnist_28x28",
            n_samples=n_samples,
            resize=28,
            dim_g=784,
            seeds=TUNING_SEEDS,
            tv_lbfgs=None,
            device=device,
        ),
    ]

    control_summary = {}
    for c in controls:
        mse_diff = max_abs_metric_diff(c, "input_mse")
        snap_diff = max_abs_metric_diff(c, "snapshot_match_acc")
        residual_diff = max_abs_metric_diff(c, "mean_snapshot_residual")
        joli_matches_shard = mse_diff < 0.02 and snap_diff < 0.01
        control_summary[c["name"]] = {
            "dim_g_ge_d": c["overdetermined_dim_g_ge_d"],
            "max_abs_input_mse_diff": mse_diff,
            "max_abs_snapshot_acc_diff": snap_diff,
            "max_abs_snapshot_residual_diff": residual_diff,
            "joli_matches_shard_within_tolerance": joli_matches_shard,
            "shard_mse_mean": c["aggregate"]["shard_l3_canonical"]["input_mse"],
            "joli_mse_mean": c["aggregate"]["joli"]["input_mse"],
        }

    results = {
        "round": 3,
        "method": "JOLI (structure-aware Stage-3 polish)",
        "baseline": "ShardAttacker.level3_invert (canonical)",
        "n_samples": n_samples,
        "n_samples_justification": (
            "N=20 (up from 12 in Round 02): doubles batch size while keeping "
            "full multi-seed Pareto sweep feasible (~25–35 min on MPS). "
            "Paper-scale N=50 deferred to Round 04 / full pipeline."
        ),
        "tuning_seeds": TUNING_SEEDS,
        "heldout_seed": HELDOUT_SEED,
        "pareto_sweep": pareto,
        "operating_point": operating,
        "control_regimes": controls,
        "control_summary": control_summary,
        "headline": {},
    }

    sel_pt = next(p for p in pareto["pareto_points"] if p["tv_lbfgs"] == selected_lambda)
    agg_s = pareto["pareto_points"][0]["aggregate_shard"]
    agg_j = sel_pt["aggregate_joli"]
    results["headline"]["compressive_pareto_selected"] = {
        "tv_lbfgs": selected_lambda,
        "shard_mse": f"{agg_s['input_mse']:.4f}±{agg_s['input_mse_std']:.4f}",
        "joli_mse": f"{agg_j['input_mse']:.4f}±{agg_j['input_mse_std']:.4f}",
        "shard_psnr_db": f"{agg_s['input_psnr_db']:.2f}±{agg_s['input_psnr_db_std']:.2f}",
        "joli_psnr_db": f"{agg_j['input_psnr_db']:.2f}±{agg_j['input_psnr_db_std']:.2f}",
        "joli_snapshot_match_acc": f"{agg_j['snapshot_match_acc']:.2f}±{agg_j['snapshot_match_acc_std']:.2f}",
        "joli_mean_snapshot_residual": f"{agg_j['mean_snapshot_residual']:.3f}±{agg_j['mean_snapshot_residual_std']:.3f}",
        "heldout_joli_beats_shard_mse": operating["heldout_validation"]["joli_beats_shard_mse"],
    }
    for name, summary in control_summary.items():
        results["headline"][name] = summary

    out_json = ARTIFACTS / "round03_metrics.json"
    out_json.write_text(json.dumps(results, indent=2))

    pareto_json = ARTIFACTS / "round03_pareto.json"
    pareto_export = {
        "pareto_points": [
            {
                "tv_lbfgs": pt["tv_lbfgs"],
                "input_mse_mean": pt["aggregate_joli"]["input_mse"],
                "input_mse_std": pt["aggregate_joli"]["input_mse_std"],
                "input_psnr_db_mean": pt["aggregate_joli"]["input_psnr_db"],
                "snapshot_match_acc_mean": pt["aggregate_joli"]["snapshot_match_acc"],
                "mean_snapshot_residual_mean": pt["aggregate_joli"]["mean_snapshot_residual"],
                "shard_input_mse_mean": pt["aggregate_shard"]["input_mse"],
            }
            for pt in pareto["pareto_points"]
        ],
        "operating_point": operating,
    }
    pareto_json.write_text(json.dumps(pareto_export, indent=2))

    plot_pareto(pareto, operating, ARTIFACTS / "round03_pareto_curves.png")
    plot_pareto_tradeoff_scatter(pareto, operating, ARTIFACTS / "round03_pareto_tradeoff.png")

    logger.info("Wrote %s", out_json)
    print(json.dumps(results["headline"], indent=2))


if __name__ == "__main__":
    main()
