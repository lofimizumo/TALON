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
    We solve the ridge Laplacian system on centered deviations:

        min_delta  delta^T L delta + (1/lambda) ||delta||_F^2
        s_i = e_bar + delta_i,  mean(delta) = 0

    ``graph_lambda`` controls smoothness: larger lambda => smaller Fiedler spread.
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
    # Ridge on deviations: spread ∝ lambda / (1 + lambda) (ablation-sensitive).
    lam = max(float(graph_lambda), 1e-6)
    spread = spread_scale * (lam / (1.0 + lam)) * np.linalg.norm(e_bar)
    delta = spread * np.outer(fiedler, weights)
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


def b1_budget_disaggregate(
    e_bar: np.ndarray,
    coeff_matrices: list[np.ndarray],
    batch_gradients: list[list[np.ndarray]],
    n_samples: int,
    dim_g: int,
) -> np.ndarray:
    """B=1 recovery when epochs may have fewer than ``n_samples`` rows."""
    from scipy.optimize import linear_sum_assignment

    e = len(coeff_matrices)
    epoch_solutions: list[np.ndarray] = []
    for ep in range(e):
        a_e = coeff_matrices[ep]
        grads = batch_gradients[ep]
        k = len(grads)
        if a_e.shape[0] == dim_g:
            sols = np.array([np.linalg.solve(a_e, grads[i]) for i in range(k)])
        else:
            sols = np.array(
                [np.linalg.lstsq(a_e, grads[i], rcond=None)[0] for i in range(k)]
            )
        epoch_solutions.append(sols)

    ref = epoch_solutions[0]
    n_ref = ref.shape[0]
    epoch_perms: list[list[int]] = [list(range(n_ref))]
    for ep in range(1, e):
        sols = epoch_solutions[ep]
        k = sols.shape[0]
        r_sq = np.sum(ref**2, axis=1, keepdims=True)
        s_sq = np.sum(sols**2, axis=1, keepdims=True)
        cost = r_sq + s_sq.T - 2.0 * ref @ sols.T
        row_ind, col_ind = linear_sum_assignment(cost)
        perm = [0] * k
        for r, c in zip(row_ind, col_ind):
            perm[c] = r
        epoch_perms.append(perm)

    lhs = np.zeros((dim_g, dim_g))
    rhs = np.zeros((dim_g, n_samples))
    for ep in range(e):
        lhs += coeff_matrices[ep].T @ coeff_matrices[ep]
        perm = epoch_perms[ep]
        k = len(batch_gradients[ep])
        for i in range(k):
            rhs[:, perm[i]] += coeff_matrices[ep].T @ batch_gradients[ep][i]

    return np.linalg.lstsq(lhs, rhs, rcond=None)[0].T


def subsample_gradient_rows(
    coeff_matrices: list[np.ndarray],
    batch_gradients: list[list[np.ndarray]],
    *,
    max_rows: int,
) -> tuple[list[np.ndarray], list[list[np.ndarray]], int]:
    """Take the first ``max_rows`` gradient rows in epoch-major order."""
    coeff_out: list[np.ndarray] = []
    grad_out: list[list[np.ndarray]] = []
    used = 0
    for e, (a_e, grads_e) in enumerate(zip(coeff_matrices, batch_gradients)):
        if used >= max_rows:
            break
        take = min(len(grads_e), max_rows - used)
        if take <= 0:
            break
        coeff_out.append(a_e)
        grad_out.append(grads_e[:take])
        used += take
    return coeff_out, grad_out, used


def active_probe_graph_terminal(
    e_bar: np.ndarray,
    coeff_matrices: list[np.ndarray],
    terminal_gradients: list[np.ndarray],
    n_samples: int,
    *,
    graph_lambda: float = 0.5,
    spread_scale: float = 0.35,
) -> np.ndarray:
    """T1 attack: diverse ``A^(e)`` refine mean, then graph prior on manifold.

    Uses stacked terminal equalities ``c^(e) ≈ A^(e) @ e_bar`` (Level-1) and
    applies ``graph_term_map`` with probe-count-aware spread (more epochs => tighter).
    """
    n_epochs = max(len(coeff_matrices), 1)
    scale = spread_scale * min(1.0, np.sqrt(n_epochs / 10.0))
    return graph_term_map(
        e_bar,
        n_samples,
        graph_lambda=graph_lambda,
        spread_scale=scale,
    )


def cross_epoch_consistency_terminal(
    e_bar: np.ndarray,
    coeff_matrices: list[np.ndarray],
    terminal_gradients: list[np.ndarray],
    n_samples: int,
    *,
    rank: int = 4,
    graph_lambda: float = 0.35,
) -> np.ndarray:
    """Assignment-free T1: low-rank + graph on deviations from multi-epoch mean.

    Stacked terminals only constrain ``e_bar``; deviations live in the span of
    graph eigenvectors (no SHARD batch assignment).
    """
    graph = chain_laplacian(n_samples)
    _, eigvecs = np.linalg.eigh(graph)
    u = eigvecs[:, : rank + 1]
    base = graph_term_map(e_bar, n_samples, graph_lambda=graph_lambda, graph=graph)
    # Project stacked probe residuals onto graph subspace (assignment-free).
    dim_g = e_bar.shape[0]
    resid = np.zeros(dim_g)
    for a_e, c_e in zip(coeff_matrices, terminal_gradients):
        resid += a_e.T @ (c_e - a_e @ e_bar)
    if len(coeff_matrices) > 0:
        resid /= len(coeff_matrices)
    coeffs = np.linalg.lstsq(u, np.outer(u[:, 1], resid), rcond=None)[0]
    delta = u @ coeffs
    s = base + delta
    s -= s.mean(axis=0, keepdims=True)
    s += e_bar[None, :]
    return s


def _solve_batch_means(
    coeff: np.ndarray, grads: list[np.ndarray], dim_g: int
) -> np.ndarray:
    g_mat = np.stack(grads, axis=0)
    if coeff.shape[0] == dim_g:
        return np.linalg.solve(coeff, g_mat.T).T
    return np.linalg.lstsq(coeff, g_mat.T, rcond=None)[0].T


def partial_honest_disaggregate(
    e_bar: np.ndarray,
    coeff_matrices: list[np.ndarray],
    partial_gradients: list[list[np.ndarray]],
    n_samples: int,
    batch_size: int,
    *,
    graph_lambda: float = 0.35,
    max_iter: int = 80,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Imputation-free partial terminal SHARD: only last-``p`` rows per epoch.

    Observed rows map to the last ``p`` minibatch slots in epoch order; no
    synthetic early rows.
    """
    from scipy.optimize import linear_sum_assignment

    if rng is None:
        rng = np.random.default_rng(0)
    n = n_samples
    b = batch_size
    k_full = n // b
    dim_g = e_bar.shape[0]
    e = len(coeff_matrices)
    graph = chain_laplacian(n)

    partial_means: list[np.ndarray] = []
    batch_offsets: list[int] = []
    for ep, grads_e in enumerate(partial_gradients):
        p = len(grads_e)
        batch_offsets.append(k_full - p)
        partial_means.append(_solve_batch_means(coeff_matrices[ep], grads_e, dim_g))

    s = graph_term_map(e_bar, n, graph_lambda=graph_lambda, graph=graph)
    s += 0.05 * rng.normal(size=s.shape)
    s -= s.mean(axis=0, keepdims=True)
    s += e_bar[None, :]
    alpha = 0.6

    for _ in range(max_iter):
        s_old = s.copy()
        assign = np.zeros((e, n), dtype=int)
        for ep in range(e):
            p = partial_means[ep].shape[0]
            k0 = batch_offsets[ep]
            bm = partial_means[ep]
            s_c = s - e_bar
            m_c = bm - e_bar
            sim = s_c @ m_c.T
            cost = np.empty((n, n))
            for j in range(p):
                col = k0 + j
                cost[:, col * b : (col + 1) * b] = (-sim[:, j])[:, None]
            row, col = linear_sum_assignment(cost)
            perm = np.zeros(n, dtype=int)
            perm[col] = row
            for i in range(n):
                assign[ep, i] = col[i] // b

        ptp = np.zeros((n, n))
        ptm = np.zeros((n, dim_g))
        w = 1.0 / b
        w2 = w * w
        for ep in range(e):
            p = partial_means[ep].shape[0]
            k0 = batch_offsets[ep]
            ae = assign[ep]
            for j in range(p):
                k_idx = k0 + j
                members = np.where(ae == k_idx)[0]
                for ii in members:
                    for jj in members:
                        ptp[ii, jj] += w2
                    ptm[ii] += w * partial_means[ep][j]
        s_new = np.linalg.lstsq(ptp, ptm, rcond=None)[0]
        s = alpha * s_new + (1.0 - alpha) * s_old
        s -= s.mean(axis=0, keepdims=True)
        s += e_bar[None, :]

    return s


def fixed_order_snapshot_mse(recovered: np.ndarray, truth: np.ndarray) -> float:
    """MSE when sample index order is known (no Hungarian permutation)."""
    return float(np.mean((recovered - truth) ** 2))
