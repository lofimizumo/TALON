"""Surrogate QFL Model for SHARD attack simulation.

Implements a classical surrogate for LASA-VQC gradient computation using
Random Fourier Features. The model preserves the linear relationship between
gradients and snapshot vectors that the SHARD attack exploits, without
requiring actual quantum circuit simulation.
"""

import math

import numpy as np
import torch


class SurrogateQFL:
    """Classical surrogate for LASA-VQC gradient computation.

    Uses Random Fourier Features to produce snapshot vectors and generates
    gradients with the same linear algebraic structure as a LASA-VQC.

    Attributes:
        W_enc: Random frequency matrix of shape (dim_g, d), scaled by 1/sqrt(d).
        b_enc: Random phase vector of shape (dim_g,) from U[0, 2π].
        dim_g: DLA dimension.
        n_params: Number of model parameters D.
        noise_level: Standard deviation of additive Gaussian gradient noise.
        coeff_matrices: List of stored coefficient matrices (one per epoch).
        all_gradients: List of stored gradient lists (one list per epoch).
    """

    def __init__(
        self,
        input_dim: int,
        dim_g: int,
        n_params: int | None = None,
        noise_level: float = 0.0,
        seed: int = 42,
    ):
        """Initialize the surrogate QFL model.

        Args:
            input_dim: Dimension d of flattened input images.
            dim_g: DLA dimension dim(g).
            n_params: Number of parameters D. Defaults to dim_g // 4.
            noise_level: Standard deviation of additive Gaussian gradient noise.
            seed: Random seed for reproducibility.
        """
        self.dim_g = dim_g
        self.n_params = n_params if n_params is not None else dim_g // 4
        self.noise_level = noise_level

        # Separate RNGs for reproducibility
        torch_gen = torch.Generator().manual_seed(seed)
        self._np_rng = np.random.default_rng(seed)

        # Random frequency matrix W_enc ~ N(0, σ²), shape (dim_g, d)
        # Scale by 1/sqrt(d) so that W@x remains O(1) for ||x|| ~ O(sqrt(d)).
        # This "low frequency" trick keeps cos(W@x + b) in a smooth regime,
        # preventing high-frequency ripples that trap gradient-based optimizers.
        freq_scale = 1.0 / math.sqrt(input_dim)
        self.W_enc = torch.randn(dim_g, input_dim, generator=torch_gen) * freq_scale

        # Random phase vector b_enc ~ U[0, 2π], shape (dim_g,)
        self.b_enc = torch.rand(dim_g, generator=torch_gen) * (2 * math.pi)

        # Storage for coefficient matrices and gradients (Req 2.6)
        self.coeff_matrices: list[np.ndarray] = []
        self.all_gradients: list[list[np.ndarray]] = []

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Compute snapshot vector e_snap(x) = cos(W_enc @ x + b_enc).

        Args:
            x: Input vector of shape (d,) or batch of shape (N, d).

        Returns:
            Snapshot vector(s) of shape (dim_g,) or (N, dim_g).
        """
        if x.dim() == 1:
            # Single vector: (d,) -> (dim_g,)
            return torch.cos(self.W_enc @ x + self.b_enc)
        else:
            # Batch: (N, d) -> (N, dim_g)
            # x @ W_enc^T gives (N, dim_g), then add b_enc broadcast
            return torch.cos(x @ self.W_enc.T + self.b_enc)

    def generate_coefficient_matrix(self) -> np.ndarray:
        """Generate a random coefficient matrix A^(e) of shape (D, dim_g).

        Returns:
            Coefficient matrix with entries drawn from N(0, 1).
        """
        return self._np_rng.standard_normal((self.n_params, self.dim_g))

    def compute_batch_gradients(
        self,
        images: torch.Tensor,
        epoch_batches: list[list[int]],
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Compute all batch gradients for one epoch.

        Generates a coefficient matrix A^(e), then for each batch k computes:
            g^(e,k) = A^(e) @ (1/B * Σ_{i in batch} e_snap(x_i)) + noise

        The coefficient matrix and gradients are stored for later retrieval.

        Args:
            images: All images, shape (N, d).
            epoch_batches: List of K batch index lists.

        Returns:
            (A_epoch, gradients): Coefficient matrix of shape (D, dim_g) and
                list of K gradient vectors each of shape (D,).
        """
        A_epoch = self.generate_coefficient_matrix()

        # Encode all images at once: (N, dim_g)
        with torch.no_grad():
            all_snapshots = self.encode(images)  # (N, dim_g)

        gradients: list[np.ndarray] = []
        batch_size = len(epoch_batches[0])

        for batch_indices in epoch_batches:
            # Compute mean snapshot for this batch
            batch_snapshots = all_snapshots[batch_indices]  # (B, dim_g)
            mean_snapshot = batch_snapshots.mean(dim=0).numpy()  # (dim_g,)

            # Compute gradient: g = A @ mean_snapshot
            g = A_epoch @ mean_snapshot  # (D,)

            # Add noise if enabled (Req 2.5)
            if self.noise_level > 0:
                noise = self._np_rng.normal(
                    0, self.noise_level, size=g.shape
                )
                g = g + noise

            gradients.append(g)

        # Store for later retrieval (Req 2.6)
        self.coeff_matrices.append(A_epoch)
        self.all_gradients.append(gradients)

        return A_epoch, gradients
