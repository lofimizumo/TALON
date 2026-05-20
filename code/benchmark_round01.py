#!/usr/bin/env python3
"""Round-01 benchmark: SHARD level3_invert vs LAPIN on MNIST snapshots."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from scipy.optimize import linear_sum_assignment

# Repo imports
RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "code"))
from _paths import VENDOR_ROOT  # noqa: E402

sys.path.insert(0, str(VENDOR_ROOT))

from lapin_invert import lapin_invert  # noqa: E402
from shard_sim.attacker import ShardAttacker  # noqa: E402
from shard_sim.surrogate_model import SurrogateQFL  # noqa: E402

ARTIFACTS = RUN_ROOT / "artifacts"
LOGS = RUN_ROOT / "logs"
DATA_DIR = RUN_ROOT / "data"


def load_mnist_subset(n_samples: int, resize: int = 14) -> torch.Tensor:
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
    imgs = torch.stack([ds[i][0].flatten() for i in range(n_samples)])
    return imgs


def hungarian_mse(recovered: np.ndarray, truth: np.ndarray) -> tuple[float, np.ndarray]:
    """Permutation-aligned MSE via Hungarian on squared distances."""
    r_sq = np.sum(recovered**2, axis=1, keepdims=True)
    t_sq = np.sum(truth**2, axis=1, keepdims=True)
    dist = r_sq + t_sq.T - 2.0 * recovered @ truth.T
    np.maximum(dist, 0.0, out=dist)
    row, col = linear_sum_assignment(dist)
    aligned = recovered[row]
    mse = float(np.mean((aligned - truth[col]) ** 2))
    return mse, col


def psnr_from_mse(mse: float, peak: float = 1.0) -> float:
    if mse <= 1e-16:
        return float("inf")
    return float(10.0 * np.log10((peak**2) / mse))


def snapshot_matching_accuracy(recovered: np.ndarray, truth: np.ndarray) -> float:
    return ShardAttacker._matching_accuracy(recovered, truth)


def pixel_match_accuracy(
    x_rec: np.ndarray, x_true: np.ndarray, thresh: float = 0.1
) -> float:
    """Fraction of samples with per-image L2 error below thresh after matching."""
    r_sq = np.sum(x_rec**2, axis=1, keepdims=True)
    t_sq = np.sum(x_true**2, axis=1, keepdims=True)
    dist = r_sq + t_sq.T - 2.0 * x_rec @ x_true.T
    np.maximum(dist, 0.0, out=dist)
    row, col = linear_sum_assignment(dist)
    errs = np.linalg.norm(x_rec[row] - x_true[col], axis=1)
    return float(np.mean(errs < thresh))


def run_benchmark(
    n_samples: int = 12,
    resize: int = 14,
    dim_g: int = 160,
    seed: int = 7,
    shard_adam_steps: int = 800,
    shard_n_batch_cap: int = 120,
) -> dict:
    torch.manual_seed(seed)
    images = load_mnist_subset(n_samples, resize=resize)
    d = images.shape[1]
    surrogate = SurrogateQFL(input_dim=d, dim_g=dim_g, seed=seed)

    with torch.no_grad():
        snapshots = surrogate.encode(images).numpy()

    attacker = ShardAttacker(dim_g=dim_g, n_samples=n_samples, batch_size=4)
    x_true = images.numpy()

    # --- SHARD L3 (reduced parallel Adam budget; same objective + L-BFGS polish) ---
    t0 = time.perf_counter()

    def level3_fast(self, snapshots, surrogate, device=None):
        """Faithful level3 with smaller Adam batch for benchmark wall-clock."""
        dev = torch.device(device if device is not None else "cpu")
        W_enc = surrogate.W_enc.detach().to(dev)
        b_enc = surrogate.b_enc.detach().to(dev)
        N, d_loc = snapshots.shape[0], W_enc.shape[1]
        dim_g_loc = W_enc.shape[0]
        n_batch = max(30, min(shard_n_batch_cap, 500_000 // max(d_loc, dim_g_loc)))
        reconstructed = np.empty((N, d_loc), dtype=np.float64)
        for i in range(N):
            target = torch.tensor(snapshots[i], dtype=W_enc.dtype, device=dev)
            restart_rng = torch.Generator().manual_seed(42 + i)
            extra = [torch.full((d_loc,), 0.5, dtype=W_enc.dtype)]
            W_cpu = surrogate.W_enc.detach()
            s_clamped = torch.clamp(
                torch.tensor(snapshots[i], dtype=W_cpu.dtype),
                -1.0 + 1e-7,
                1.0 - 1e-7,
            )
            theta = torch.acos(s_clamped)
            b_cpu = surrogate.b_enc.detach()
            for rhs in [
                theta - b_cpu,
                -theta - b_cpu,
                theta + 2 * torch.pi - b_cpu,
                -theta + 2 * torch.pi - b_cpu,
            ]:
                sol = torch.linalg.lstsq(W_cpu, rhs).solution
                extra.append(sol.clamp(0.0, 1.0))
            n_extra = len(extra)
            n_random = n_batch - n_extra
            random_starts = torch.rand(
                n_random, d_loc, dtype=W_enc.dtype, generator=restart_rng
            )
            X = torch.cat([torch.stack(extra), random_starts], dim=0).to(dev)
            X = X.detach().clone().requires_grad_(True)
            adam = torch.optim.Adam([X], lr=0.01)
            for _ in range(shard_adam_steps):
                adam.zero_grad()
                snaps = torch.cos(X @ W_enc.T + b_enc)
                recon_loss = ((snaps - target) ** 2).sum(dim=1)
                loss = recon_loss.sum()
                loss.backward()
                adam.step()
                with torch.no_grad():
                    X.clamp_(0.0, 1.0)
                if recon_loss.min().item() < 1e-8:
                    break
            with torch.no_grad():
                snaps = torch.cos(X @ W_enc.T + b_enc)
                losses = ((snaps - target) ** 2).sum(dim=1)
                best_idx = int(losses.argmin())
            W_lbfgs = surrogate.W_enc.detach()
            b_lbfgs = surrogate.b_enc.detach()
            target_cpu = torch.tensor(snapshots[i], dtype=W_lbfgs.dtype)
            x = X[best_idx].detach().cpu().clone().requires_grad_(True)
            optimizer = torch.optim.LBFGS(
                [x],
                max_iter=300,
                tolerance_grad=1e-12,
                tolerance_change=1e-14,
                line_search_fn="strong_wolfe",
            )

            def closure():
                optimizer.zero_grad()
                snap_x = torch.cos(W_lbfgs @ x + b_lbfgs)
                loss = ((snap_x - target_cpu) ** 2).sum()
                loss.backward()
                return loss

            optimizer.step(closure)
            with torch.no_grad():
                x.clamp_(0.0, 1.0)
            reconstructed[i] = x.detach().numpy().astype(np.float64)
        return reconstructed

    x_shard = level3_fast(attacker, snapshots, surrogate)
    t_shard = time.perf_counter() - t0

    # --- LAPIN ---
    t1 = time.perf_counter()
    x_lapin = lapin_invert(snapshots, surrogate)
    t_lapin = time.perf_counter() - t1

    with torch.no_grad():
        snap_true = surrogate.encode(torch.tensor(x_true, dtype=torch.float32)).numpy()
        snap_rec_shard = surrogate.encode(torch.tensor(x_shard, dtype=torch.float32)).numpy()
        snap_rec_lapin = surrogate.encode(torch.tensor(x_lapin, dtype=torch.float32)).numpy()

    mse_shard, _ = hungarian_mse(x_shard, x_true)
    mse_lapin, _ = hungarian_mse(x_lapin, x_true)

    results = {
        "config": {
            "n_samples": n_samples,
            "resize": resize,
            "d": d,
            "dim_g": dim_g,
            "seed": seed,
            "shard_adam_steps": shard_adam_steps,
            "shard_n_batch": shard_n_batch_cap,
        },
        "shard_l3": {
            "input_mse": mse_shard,
            "input_psnr_db": psnr_from_mse(mse_shard),
            "snapshot_match_acc": snapshot_matching_accuracy(snap_rec_shard, snap_true),
            "pixel_match_acc_0.1": pixel_match_accuracy(x_shard, x_true, 0.1),
            "mean_snapshot_residual": float(
                np.mean(np.sum((snap_rec_shard - snapshots) ** 2, axis=1))
            ),
            "runtime_sec": t_shard,
        },
        "lapin": {
            "input_mse": mse_lapin,
            "input_psnr_db": psnr_from_mse(mse_lapin),
            "snapshot_match_acc": snapshot_matching_accuracy(snap_rec_lapin, snap_true),
            "pixel_match_acc_0.1": pixel_match_accuracy(x_lapin, x_true, 0.1),
            "mean_snapshot_residual": float(
                np.mean(np.sum((snap_rec_lapin - snapshots) ** 2, axis=1))
            ),
            "runtime_sec": t_lapin,
        },
    }
    return results, x_true, x_shard, x_lapin, resize


def plot_metrics_bar(results: dict, out_path: Path) -> None:
    methods = ["shard_l3", "lapin"]
    labels = ["SHARD L3", "LAPIN"]
    mse = [results[m]["input_mse"] for m in methods]
    psnr = [results[m]["input_psnr_db"] for m in methods]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    axes[0].bar(labels, mse, color=["#4c72b0", "#dd8452"])
    axes[0].set_ylabel("Input MSE (matched)")
    axes[0].set_title("Reconstruction error")
    axes[1].bar(labels, psnr, color=["#4c72b0", "#dd8452"])
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].set_title("Reconstruction quality")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_comparison(
    x_true: np.ndarray,
    x_shard: np.ndarray,
    x_lapin: np.ndarray,
    resize: int,
    out_path: Path,
) -> None:
    n_show = min(6, x_true.shape[0])
    fig, axes = plt.subplots(3, n_show, figsize=(1.6 * n_show, 4.8))
    for j in range(n_show):
        for row, arr in enumerate((x_true, x_shard, x_lapin)):
            axes[row, j].imshow(arr[j].reshape(resize, resize), cmap="gray", vmin=0, vmax=1)
            axes[row, j].axis("off")
        axes[0, j].set_title(f"#{j}", fontsize=8)
    axes[0, 0].set_ylabel("Truth", fontsize=9)
    axes[1, 0].set_ylabel("SHARD L3", fontsize=9)
    axes[2, 0].set_ylabel("LAPIN", fontsize=9)
    fig.suptitle("Round-01 MNIST inversion (14×14)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    results, x_true, x_shard, x_lapin, resize = run_benchmark()
    out_json = ARTIFACTS / "round01_metrics.json"
    out_json.write_text(json.dumps(results, indent=2))
    plot_comparison(
        x_true,
        x_shard,
        x_lapin,
        resize,
        ARTIFACTS / "round01_reconstruction_grid.png",
    )
    plot_metrics_bar(results, ARTIFACTS / "round01_metrics_bar.png")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
