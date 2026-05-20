"""Metrics for evaluating SHARD attack success.

Provides matching accuracy (via Hungarian algorithm) and reconstruction MSE.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_matching_accuracy(
    recovered: np.ndarray, true: np.ndarray
) -> tuple[float, np.ndarray]:
    """Compute optimal matching accuracy using the Hungarian algorithm.

    Builds a pairwise squared-distance matrix between recovered and true
    vectors, solves the linear assignment problem, and reports the fraction
    of matched pairs whose squared distance is below a threshold.  The
    threshold is set to 1% of the mean pairwise squared distance, so a
    "correct" match is one that is much closer than a random pairing.

    Args:
        recovered: Recovered vectors of shape ``(N, dim)``.
        true: Ground-truth vectors of shape ``(N, dim)``.

    Returns:
        ``(accuracy, assignment)`` where *accuracy* is the fraction of
        correctly matched samples in [0, 1] and *assignment* is an integer
        array of shape ``(N,)`` mapping recovered index → true index.
    """
    r_sq = np.sum(recovered ** 2, axis=1, keepdims=True)
    t_sq = np.sum(true ** 2, axis=1, keepdims=True)
    dist_matrix = r_sq + t_sq.T - 2.0 * recovered @ true.T
    # Clamp tiny negatives from floating-point noise
    np.maximum(dist_matrix, 0.0, out=dist_matrix)

    row_ind, col_ind = linear_sum_assignment(dist_matrix)

    # Matched distances
    matched_dists = dist_matrix[row_ind, col_ind]

    # Threshold: 1% of the mean off-diagonal squared distance.
    # This separates "essentially identical" pairs from random pairings.
    mean_dist = float(np.mean(dist_matrix))
    threshold = 0.01 * mean_dist if mean_dist > 0 else 1e-12

    n_correct = int(np.sum(matched_dists < threshold))
    accuracy = n_correct / len(row_ind)
    return float(accuracy), col_ind


def compute_reconstruction_mse(
    recovered: np.ndarray, true: np.ndarray, assignment: np.ndarray
) -> float:
    """Compute MSE between matched recovered and true vectors.

    Args:
        recovered: Recovered vectors of shape ``(N, dim)``.
        true: Ground-truth vectors of shape ``(N, dim)``.
        assignment: Optimal assignment array from :func:`compute_matching_accuracy`,
            mapping recovered[i] → true[assignment[i]].

    Returns:
        Mean squared error averaged over all N matched pairs.
    """
    matched_true = true[assignment]
    return float(np.mean((recovered - matched_true) ** 2))
