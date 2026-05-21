"""LAPIN: Latent Projected Gauss-Newton INversion for cos-encoded snapshots.

Multi-start projected Gauss-Newton with analytic Jacobian diag(-sin(z)) W,
encoding-aware acos lstsq seeds, and a single L-BFGS polish. No Adam restarts.

Complexity per image (rough): O(n_starts * max_gn * d^3) for JTJ solves in R^d.
At d=784 default n_random=48, max_gn=25 this is ~100s/image — see l3_budget.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from shard_sim.surrogate_model import SurrogateQFL

logger = logging.getLogger(__name__)


def lapin_budget(d: int, dim_g: int) -> dict:
    """Dimension-aware LAPIN budget to avoid O(d^3) blow-up on MNIST 28×28."""
    if d <= 128:
        return {
            "max_gn": 25,
            "n_random": 48,
            "damping": 1e-2,
            "lbfgs_iter": 300,
            "polish": True,
        }
    if d <= 400:
        return {
            "max_gn": 15,
            "n_random": 12,
            "damping": 1e-2,
            "lbfgs_iter": 120,
            "polish": True,
        }
    # 28×28 MNIST (d=784): full defaults are ~100s/image
    return {
        "max_gn": 8,
        "n_random": 4,
        "damping": 5e-2,
        "lbfgs_iter": 40,
        "polish": True,
    }


def _lstsq_seeds(
    target: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
) -> list[np.ndarray]:
    s = np.clip(target, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = np.arccos(s)
    seeds: list[np.ndarray] = []
    for rhs in (theta - b, -theta - b, theta + 2 * np.pi - b, -theta + 2 * np.pi - b):
        sol, _, _, _ = np.linalg.lstsq(W, rhs, rcond=None)
        seeds.append(np.clip(sol, 0.0, 1.0))
    seeds.append(np.full(W.shape[1], 0.5))
    return seeds


def _lbfgs_polish(
    x0: np.ndarray,
    target: np.ndarray,
    W_t: torch.Tensor,
    b_t: torch.Tensor,
    max_iter: int = 300,
) -> np.ndarray:
    x = torch.tensor(x0, dtype=W_t.dtype, requires_grad=True)
    target_t = torch.tensor(target, dtype=W_t.dtype)
    opt = torch.optim.LBFGS(
        [x],
        max_iter=max_iter,
        tolerance_grad=1e-9,
        tolerance_change=1e-11,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        opt.zero_grad()
        snap = torch.cos(W_t @ x + b_t)
        loss = ((snap - target_t) ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        x.clamp_(0.0, 1.0)
    return x.detach().numpy().astype(np.float64)


def _residual(x: np.ndarray, target: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.cos(W @ x + b) - target


def _gn_from(
    x0: np.ndarray,
    target: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    max_gn: int,
    damping: float,
) -> tuple[np.ndarray, float]:
    d = W.shape[1]
    x = x0.copy()
    best_x, best_res = x.copy(), float("inf")
    for _ in range(max_gn):
        z = W @ x + b
        cz, sz = np.cos(z), np.sin(z)
        r = cz - target
        res = float(np.sum(r * r))
        if res < best_res:
            best_res, best_x = res, x.copy()
        if res < 1e-10:
            break
        J = -sz[:, None] * W
        JTJ = J.T @ J + damping * np.eye(d)
        step = np.linalg.solve(JTJ, J.T @ (-r))
        alpha = 1.0
        for _bt in range(12):
            x_try = np.clip(x + alpha * step, 0.0, 1.0)
            if np.sum(_residual(x_try, target, W, b) ** 2) < res:
                x = x_try
                break
            alpha *= 0.5
        else:
            break
    return best_x, best_res


def lapin_invert_single(
    target: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    *,
    max_gn: int | None = None,
    damping: float | None = None,
    n_random: int | None = None,
    seed: int = 0,
    polish: bool | None = None,
    lbfgs_iter: int | None = None,
    d_hint: int | None = None,
) -> tuple[np.ndarray, float]:
    """Multi-start projected GN (encoding seeds + sparse random), then L-BFGS."""
    d = W.shape[1]
    dim_g = W.shape[0]
    budget = lapin_budget(d_hint or d, dim_g)
    max_gn = max_gn if max_gn is not None else budget["max_gn"]
    damping = damping if damping is not None else budget["damping"]
    n_random = n_random if n_random is not None else budget["n_random"]
    polish = budget["polish"] if polish is None else polish
    lbfgs_iter = lbfgs_iter if lbfgs_iter is not None else budget["lbfgs_iter"]

    best_x, best_res = np.full(d, 0.5), float("inf")
    rng = np.random.default_rng(seed)

    starts = _lstsq_seeds(target, W, b)
    starts.extend([rng.random(d) for _ in range(n_random)])

    for x0 in starts:
        x, res = _gn_from(x0, target, W, b, max_gn, damping)
        if res < best_res:
            best_res, best_x = res, x

    if polish:
        W_t = torch.tensor(W, dtype=torch.float64)
        b_t = torch.tensor(b, dtype=torch.float64)
        x_pol = _lbfgs_polish(best_x, target, W_t, b_t, max_iter=lbfgs_iter)
        res_pol = float(np.sum(_residual(x_pol, target, W, b) ** 2))
        if res_pol < best_res:
            best_x, best_res = x_pol, res_pol
    return best_x, best_res


def lapin_invert(
    snapshots: np.ndarray,
    surrogate: "SurrogateQFL",
    *,
    log_budget: bool = True,
) -> np.ndarray:
    W = surrogate.W_enc.detach().cpu().numpy().astype(np.float64)
    b = surrogate.b_enc.detach().cpu().numpy().astype(np.float64)
    N = snapshots.shape[0]
    d = W.shape[1]
    budget = lapin_budget(d, W.shape[0])
    if log_budget:
        logger.info(
            "LAPIN budget d=%d: max_gn=%d n_random=%d lbfgs=%d (~%.0fs/image est)",
            d,
            budget["max_gn"],
            budget["n_random"],
            budget["lbfgs_iter"],
            0.3 * budget["max_gn"] * (5 + budget["n_random"]) * (d / 200) ** 2,
        )
    out = np.empty((N, d), dtype=np.float64)
    for i in range(N):
        out[i], _ = lapin_invert_single(
            snapshots[i], W, b, seed=42 + i, d_hint=d
        )
    logger.info("LAPIN inversion complete: %d snapshots -> dim %d", N, d)
    return out
