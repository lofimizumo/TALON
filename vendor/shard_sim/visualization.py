"""Visualization utilities for SHARD attack results."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_reconstruction_grid(
    original_images: np.ndarray,
    reconstructed_images: np.ndarray,
    assignment: np.ndarray,
    image_shape: tuple,
    n_display: int = 5,
    save_path: str = "shard_results.png",
) -> None:
    """Plot a grid comparing original and reconstructed images.

    Row 1 shows ``n_display`` original images; Row 2 shows the
    corresponding reconstructed images matched via *assignment*.

    Args:
        original_images: Ground-truth images of shape ``(N, d)``.
        reconstructed_images: Recovered images of shape ``(N, d)``.
        assignment: Matching array from :func:`compute_matching_accuracy`.
        image_shape: Original spatial shape, e.g. ``(28, 28)`` or ``(3, 32, 32)``.
        n_display: Number of image pairs to show (default 5).
        save_path: File path to save the figure.
    """
    n_display = min(n_display, len(original_images))
    is_color = len(image_shape) == 3 and image_shape[0] in (1, 3)

    fig, axes = plt.subplots(2, n_display, figsize=(2 * n_display, 4.5))
    if n_display == 1:
        axes = axes.reshape(2, 1)

    for j in range(n_display):
        # Original
        orig = original_images[j].reshape(image_shape)
        rec = reconstructed_images[assignment[j]].reshape(image_shape)

        if is_color and image_shape[0] == 3:
            orig = np.clip(orig.transpose(1, 2, 0), 0, 1)
            rec = np.clip(rec.transpose(1, 2, 0), 0, 1)
            axes[0, j].imshow(orig)
            axes[1, j].imshow(rec)
        elif is_color and image_shape[0] == 1:
            axes[0, j].imshow(orig.squeeze(), cmap="gray", vmin=0, vmax=1)
            axes[1, j].imshow(rec.squeeze(), cmap="gray", vmin=0, vmax=1)
        else:
            axes[0, j].imshow(orig, cmap="gray", vmin=0, vmax=1)
            axes[1, j].imshow(rec, cmap="gray", vmin=0, vmax=1)

        axes[0, j].axis("off")
        axes[1, j].axis("off")

    axes[0, 0].set_title("Original", fontsize=10)
    axes[1, 0].set_title("Recovered", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
