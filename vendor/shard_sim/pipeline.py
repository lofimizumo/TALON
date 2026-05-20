"""End-to-end SHARD attack pipeline.

Orchestrates data loading, gradient generation, the 3-level attack,
metrics computation, and visualization.
"""

import logging

import numpy as np
import torch

from .attacker import ShardAttacker
from .data_loader import FederatedDataLoader
from .metrics import compute_matching_accuracy, compute_reconstruction_mse
from .surrogate_model import SurrogateQFL
from .visualization import plot_reconstruction_grid

logger = logging.getLogger(__name__)


def run_shard_pipeline(
    dataset: str = "mnist",
    n_samples: int = 20,
    batch_size: int = 5,
    dim_g: int = 100,
    n_params: int | None = None,
    n_epochs: int = 10,
    noise_level: float = 0.0,
    max_iter: int = 20,
    tol: float = 1e-8,
    seed: int = 42,
) -> dict:
    """Run the complete SHARD attack pipeline.

    Args:
        dataset: ``"mnist"`` or ``"cifar10"``.
        n_samples: Number of samples N.
        batch_size: Batch size B.
        dim_g: DLA dimension.
        n_params: Number of parameters D (defaults to ``dim_g // 4``).
        n_epochs: Number of epochs E.
        noise_level: Gradient noise σ.
        max_iter: SHARD max iterations.
        tol: Convergence threshold ε.
        seed: Random seed.

    Returns:
        Dictionary with keys ``mean_rel_error``, ``matching_accuracy``,
        ``reconstruction_mse``, ``recovered_images``, ``assignment``.
    """
    if n_params is None:
        n_params = dim_g // 4

    # --- Stage 1: Data Loading ---
    try:
        loader = FederatedDataLoader(
            dataset=dataset, n_samples=n_samples, batch_size=batch_size, seed=seed
        )
    except Exception as exc:
        raise RuntimeError(f"Data loading failed: {exc}") from exc

    images = loader.images  # (N, d)
    d = images.shape[1]
    logger.info("Loaded %d %s images (d=%d)", n_samples, dataset, d)

    # --- Stage 2: Gradient Generation ---
    try:
        surrogate = SurrogateQFL(
            input_dim=d, dim_g=dim_g, n_params=n_params,
            noise_level=noise_level, seed=seed,
        )

        coeff_matrices: list[np.ndarray] = []
        batch_gradients: list[list[np.ndarray]] = []

        for e in range(n_epochs):
            epoch_batches = loader.get_epoch_batches()
            A_e, grads_e = surrogate.compute_batch_gradients(images, epoch_batches)
            coeff_matrices.append(A_e)
            batch_gradients.append(grads_e)

        logger.info("Generated gradients for %d epochs (D=%d)", n_epochs, n_params)
    except Exception as exc:
        raise RuntimeError(f"Gradient generation failed: {exc}") from exc

    # --- Stage 3: Level 1 — Mean Recovery ---
    try:
        attacker = ShardAttacker(
            dim_g=dim_g, n_samples=n_samples, batch_size=batch_size,
            max_iter=max_iter, tol=tol,
        )
        e_bar_data = attacker.level1_mean_recovery(coeff_matrices, batch_gradients)

        # Compute true mean for error reporting
        with torch.no_grad():
            true_snapshots = surrogate.encode(images).numpy()
        true_mean = true_snapshots.mean(axis=0)
        mean_rel_error = float(
            np.linalg.norm(e_bar_data - true_mean)
            / max(np.linalg.norm(true_mean), 1e-12)
        )
        logger.info("Level 1 mean recovery: relative error = %.2e", mean_rel_error)
    except Exception as exc:
        raise RuntimeError(f"Level 1 mean recovery failed: {exc}") from exc

    # --- Stage 4: Level 2 — Disaggregation ---
    try:
        S = attacker.level2_disaggregate(
            e_bar_data, coeff_matrices, batch_gradients, true_snapshots
        )
        matching_acc, assignment = compute_matching_accuracy(S, true_snapshots)
        logger.info("Level 2 matching accuracy: %.1f%%", matching_acc * 100)
    except Exception as exc:
        raise RuntimeError(f"Level 2 disaggregation failed: {exc}") from exc

    # --- Stage 5: Level 3 — Inversion ---
    try:
        recovered_images = attacker.level3_invert(S, surrogate)
        true_images_np = images.numpy() if isinstance(images, torch.Tensor) else images
        recon_mse = compute_reconstruction_mse(recovered_images, true_images_np, assignment)
        logger.info("Level 3 reconstruction MSE: %.4f", recon_mse)
    except Exception as exc:
        raise RuntimeError(f"Level 3 inversion failed: {exc}") from exc

    # --- Stage 6: Visualization ---
    try:
        plot_reconstruction_grid(
            true_images_np, recovered_images, assignment, loader.image_shape,
        )
    except Exception as exc:
        logger.warning("Visualization failed: %s", exc)

    # --- Summary ---
    logger.info(
        "Pipeline complete: L1 error=%.2e, L2 acc=%.1f%%, L3 MSE=%.4f",
        mean_rel_error, matching_acc * 100, recon_mse,
    )

    return {
        "mean_rel_error": mean_rel_error,
        "matching_accuracy": matching_acc,
        "reconstruction_mse": recon_mse,
        "recovered_images": recovered_images,
        "assignment": assignment,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_shard_pipeline()
