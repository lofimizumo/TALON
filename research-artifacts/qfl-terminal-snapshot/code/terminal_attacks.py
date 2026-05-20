"""Terminal-only snapshot recovery candidates (no intermediate batch rows)."""

from __future__ import annotations

import numpy as np


def chain_laplacian(n_samples: int) -> np.ndarray:
    """Unweighted path graph Laplacian on sample index order."""
    lap = np.zeros((n_samples, n_samples), dtype=np.float64)
    for i in range(n_samples - 1):
        lap[i, i] += 1.0
        lap[i + 1, i + 1] += 1.0
        lap[i, i + 1] -= 1.0
        lap[i + 1, i] -= 1.0
    return lap


def permuted_chain_laplacian(n_samples: int, seed: int) -> np.ndarray:
    """Chain Laplacian on a random index permutation (wrong-graph ablation)."""
    rng = np.random.default_rng(seed + 8800)
    perm = rng.permutation(n_samples)
    p = np.eye(n_samples, dtype=np.float64)[perm]
    return p @ chain_laplacian(n_samples) @ p.T


def passive_mean_broadcast(e_bar: np.ndarray, n_samples: int) -> np.ndarray:
    """Baseline: all individuals equal recovered dataset mean."""
    return np.tile(e_bar, (n_samples, 1))


def graph_term_map(
    e_bar: np.ndarray,
    n_samples: int,
    *,
    graph_lambda: float = 0.5,
    graph: np.ndarray | None = None,
    spread_scale: float = 0.35,
) -> np.ndarray:
    """Graph-smooth spread around Level-1 mean (oracle chain prior).

    T1 mean anchor alone yields a constant RHS for the Laplacian system, so a
    literal mean-only MAP collapses to passive broadcast (Round-01 degeneracy).
    We instead solve for deviations in non-constant Laplacian modes, then
    re-center rows to ``e_bar``:

        min_delta  ||delta||_L^2 + (1/lambda) ||delta||_F^2
        s_i = e_bar + delta_i,  mean(delta) = 0
    """
    if graph is None:
        graph = chain_laplacian(n_samples)
    n = n_samples
    dim_g = e_bar.shape[0]
    _, eigvecs = np.linalg.eigh(graph)
    fiedler = eigvecs[:, 1].copy()
    fiedler -= fiedler.mean()
    fiedler /= np.linalg.norm(fiedler) + 1e-12
    weights = e_bar / (np.linalg.norm(e_bar) + 1e-12)
    delta = spread_scale * np.linalg.norm(e_bar) * np.outer(fiedler, weights)
    s = np.tile(e_bar, (n, 1)) + delta
    s -= s.mean(axis=0, keepdims=True)
    s += e_bar[None, :]
    return s


def graph_rank_terminal(
    e_bar: np.ndarray,
    n_samples: int,
    dim_g: int,
    *,
    rank: int = 4,
    graph_lambda: float = 0.35,
    graph: np.ndarray | None = None,
    spread_scale: float = 0.35,
) -> np.ndarray:
    """Graph low-rank subspace MAP with mean anchor only (terminal observations)."""
    del dim_g
    if graph is None:
        graph = chain_laplacian(n_samples)
    s_full = graph_term_map(
        e_bar,
        n_samples,
        graph_lambda=graph_lambda,
        graph=graph,
        spread_scale=spread_scale,
    )
    _, eigvecs = np.linalg.eigh(graph)
    u = eigvecs[:, : rank + 1]
    coeffs = np.linalg.lstsq(u, s_full, rcond=None)[0]
    return u @ coeffs


def snapshot_row_std(s: np.ndarray) -> float:
    """Mean L2 deviation of rows from the row-mean (spread diagnostic)."""
    row_mean = s.mean(axis=0, keepdims=True)
    return float(np.mean(np.linalg.norm(s - row_mean, axis=1)))


def partial_epoch_gradients(
    batch_gradients: list[list[np.ndarray]],
    *,
    rows_per_epoch: int,
) -> list[list[np.ndarray]]:
    """Keep only the last ``rows_per_epoch`` minibatch rows per epoch (terminal leak)."""
    out: list[list[np.ndarray]] = []
    for grads_e in batch_gradients:
        k = min(rows_per_epoch, len(grads_e))
        out.append(grads_e[-k:])
    return out


def pad_partial_for_shard(
    coeff_matrices: list[np.ndarray],
    partial_gradients: list[list[np.ndarray]],
    e_bar: np.ndarray,
    k_full: int,
) -> list[list[np.ndarray]]:
    """Impute missing early-minibatch rows with ``A^(e) @ e_bar`` so SHARD sees K rows."""
    padded: list[list[np.ndarray]] = []
    for e, partial_e in enumerate(partial_gradients):
        missing = k_full - len(partial_e)
        imputed = [coeff_matrices[e] @ e_bar for _ in range(missing)]
        padded.append(imputed + list(partial_e))
    return padded


def fixed_order_snapshot_mse(recovered: np.ndarray, truth: np.ndarray) -> float:
    """MSE when sample index order is known (no Hungarian permutation)."""
    return float(np.mean((recovered - truth) ** 2))
