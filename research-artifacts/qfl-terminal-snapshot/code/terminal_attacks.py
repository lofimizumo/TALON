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


def passive_mean_broadcast(e_bar: np.ndarray, n_samples: int) -> np.ndarray:
    """Baseline: all individuals equal recovered dataset mean."""
    return np.tile(e_bar, (n_samples, 1))


def graph_term_map(
    e_bar: np.ndarray,
    n_samples: int,
    *,
    graph_lambda: float = 0.5,
    graph: np.ndarray | None = None,
) -> np.ndarray:
    """MAP with mean anchor + graph smoothness; no batch-mean rows.

    min_S ||(1/N) 1^T S - e_bar||^2 + lambda Tr(S^T L S)
    """
    if graph is None:
        graph = chain_laplacian(n_samples)
    anchor = np.ones((1, n_samples), dtype=np.float64) / n_samples
    weight = np.sqrt(100.0)
    h_aug = weight * anchor
    m_aug = weight * e_bar[None, :]
    lhs = h_aug.T @ h_aug + graph_lambda * graph
    rhs = h_aug.T @ m_aug
    return np.linalg.solve(lhs, rhs)


def graph_rank_terminal(
    e_bar: np.ndarray,
    n_samples: int,
    dim_g: int,
    *,
    rank: int = 4,
    graph_lambda: float = 0.35,
    graph: np.ndarray | None = None,
) -> np.ndarray:
    """Graph low-rank subspace MAP with mean anchor only (terminal observations).

    Solve full GRAPH-TERM, then project onto constant + lowest Laplacian modes.
    """
    del dim_g  # inferred from e_bar
    if graph is None:
        graph = chain_laplacian(n_samples)
    s_full = graph_term_map(e_bar, n_samples, graph_lambda=graph_lambda, graph=graph)
    _, eigvecs = np.linalg.eigh(graph)
    u = eigvecs[:, : rank + 1]
    coeffs = np.linalg.lstsq(u, s_full, rcond=None)[0]
    return u @ coeffs
