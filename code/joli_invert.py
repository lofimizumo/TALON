"""JOLI: structure-aware Stage-3 polish with compressive inversion prior.

Mirrors SHARD L3 (matched n_batch, Adam steps, L-BFGS budget). In the
compressive regime (dim_g < d), adds a light isotropic-TV prior **only in the
L-BFGS polish** to select smoother inputs from the snapshot null space.
When dim_g >= d, TV is disabled and JOLI reduces to the SHARD objective.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from shard_sim.surrogate_model import SurrogateQFL

logger = logging.getLogger(__name__)


def shard_n_batch(d: int, dim_g: int) -> int:
    """Match ShardAttacker.level3_invert parallel start count."""
    return max(50, min(500, 500_000 // max(d, dim_g)))


def _tv_penalty(X: torch.Tensor, side: int) -> torch.Tensor:
    """Mean isotropic TV over batch rows (differentiable)."""
    n, d = X.shape
    img = X.view(n, 1, side, side)
    dh = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).mean(dim=(1, 2, 3))
    dw = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).mean(dim=(1, 2, 3))
    return dh + dw


def _lstsq_seeds(
    target: np.ndarray,
    W_cpu: torch.Tensor,
    b_cpu: torch.Tensor,
) -> list[torch.Tensor]:
    s = torch.clamp(
        torch.tensor(target, dtype=W_cpu.dtype),
        -1.0 + 1e-7,
        1.0 - 1e-7,
    )
    theta = torch.acos(s)
    seeds: list[torch.Tensor] = [torch.full((W_cpu.shape[1],), 0.5, dtype=W_cpu.dtype)]
    for rhs in (theta - b_cpu, -theta - b_cpu, theta + 2 * torch.pi - b_cpu, -theta + 2 * torch.pi - b_cpu):
        sol = torch.linalg.lstsq(W_cpu, rhs).solution
        seeds.append(sol.clamp(0.0, 1.0))
    return seeds


def joli_invert_single(
    target: np.ndarray,
    surrogate: "SurrogateQFL",
    *,
    n_batch: int | None = None,
    adam_steps: int = 2000,
    adam_lr: float = 0.01,
    lbfgs_iter: int = 500,
    tv_adam: float = 0.0,
    tv_lbfgs: float = 0.0,
    seed: int = 0,
    device: torch.device | None = None,
) -> tuple[np.ndarray, float]:
    """Structure-aware L3: SHARD-compatible Adam + TV-aware L-BFGS polish."""
    dev = device or torch.device("cpu")
    W_enc = surrogate.W_enc.detach().to(dev)
    b_enc = surrogate.b_enc.detach().to(dev)
    dim_g, d = W_enc.shape
    if n_batch is None:
        n_batch = shard_n_batch(d, dim_g)
    side = int(round(d**0.5)) if abs(int(round(d**0.5)) ** 2 - d) < 1 else None

    W_cpu = surrogate.W_enc.detach()
    b_cpu = surrogate.b_enc.detach()
    extra = _lstsq_seeds(target, W_cpu, b_cpu)
    n_extra = len(extra)
    n_random = max(0, n_batch - n_extra)
    rng = torch.Generator().manual_seed(seed)
    random_starts = torch.rand(n_random, d, dtype=W_enc.dtype, generator=rng)
    X = torch.cat([torch.stack(extra), random_starts], dim=0).to(dev)
    X = X.detach().clone().requires_grad_(True)

    target_t = torch.tensor(target, dtype=W_enc.dtype, device=dev)
    adam = torch.optim.Adam([X], lr=adam_lr)
    for _ in range(adam_steps):
        adam.zero_grad()
        snaps = torch.cos(X @ W_enc.T + b_enc)
        recon = ((snaps - target_t) ** 2).sum(dim=1)
        loss = recon.sum()
        if tv_adam > 0.0 and side is not None:
            loss = loss + tv_adam * _tv_penalty(X, side).sum()
        loss.backward()
        adam.step()
        with torch.no_grad():
            X.clamp_(0.0, 1.0)
        if recon.min().item() < 1e-8:
            break

    with torch.no_grad():
        snaps = torch.cos(X @ W_enc.T + b_enc)
        recon = ((snaps - target_t) ** 2).sum(dim=1)
        best_idx = int(recon.argmin())

    W_lbfgs = surrogate.W_enc.detach()
    b_lbfgs = surrogate.b_enc.detach()
    target_cpu = torch.tensor(target, dtype=W_lbfgs.dtype)
    x = X[best_idx].detach().cpu().clone().requires_grad_(True)
    opt = torch.optim.LBFGS(
        [x],
        max_iter=lbfgs_iter,
        tolerance_grad=1e-12,
        tolerance_change=1e-14,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        opt.zero_grad()
        snap = torch.cos(W_lbfgs @ x + b_lbfgs)
        loss = ((snap - target_cpu) ** 2).sum()
        if tv_lbfgs > 0.0 and side is not None:
            loss = loss + tv_lbfgs * _tv_penalty(x.unsqueeze(0), side).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        x.clamp_(0.0, 1.0)
        snap = torch.cos(W_lbfgs @ x + b_lbfgs)
        res = float(((snap - target_cpu) ** 2).sum())
    return x.detach().numpy().astype(np.float64), res


def joli_invert(
    snapshots: np.ndarray,
    surrogate: "SurrogateQFL",
    *,
    adam_steps: int = 2000,
    lbfgs_iter: int = 500,
    tv_adam: float | None = None,
    tv_lbfgs: float | None = None,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Invert snapshots; default TV only in L-BFGS when dim_g < d."""
    dev = torch.device(device if device is not None else "cpu")
    W = surrogate.W_enc.detach()
    d, dim_g = W.shape[1], W.shape[0]
    compressive = dim_g < d
    if tv_adam is None:
        tv_adam = 0.0
    if tv_lbfgs is None:
        tv_lbfgs = 5e-3 if compressive else 0.0
    elif not compressive:
        tv_lbfgs = 0.0
    n_batch = shard_n_batch(d, dim_g)
    N = snapshots.shape[0]
    out = np.empty((N, d), dtype=np.float64)
    for i in range(N):
        out[i], _ = joli_invert_single(
            snapshots[i],
            surrogate,
            n_batch=n_batch,
            adam_steps=adam_steps,
            lbfgs_iter=lbfgs_iter,
            tv_adam=tv_adam,
            tv_lbfgs=tv_lbfgs,
            seed=42 + i,
            device=dev,
        )
    logger.info(
        "JOLI inversion complete: N=%d, d=%d, dim_g=%d, n_batch=%d, "
        "adam_steps=%d, tv_adam=%.4f, tv_lbfgs=%.4f",
        N,
        d,
        dim_g,
        n_batch,
        adam_steps,
        tv_adam,
        tv_lbfgs,
    )
    return out
