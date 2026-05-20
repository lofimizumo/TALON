"""SHARD Attacker — 3-level privacy attack pipeline.

Implements the SHARD (Snapshot Harvesting via Adaptive Rank Decomposition)
attack for recovering individual data from batched, reshuffled gradients
in a surrogate Quantum Federated Learning setting.

Levels:
    1. Mean Recovery — Recover dataset mean snapshot via stacked least squares.
    2. Individual Disaggregation — SHARD alternating minimization (to be added).
    3. Snapshot Inversion — L-BFGS inversion to input space (to be added).
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


class ShardAttacker:
    """Implements the 3-level SHARD attack pipeline.

    Attributes:
        dim_g: DLA dimension.
        n_samples: Dataset size N.
        batch_size: Batch size B.
        max_iter: Maximum SHARD iterations for Level 2.
        tol: Convergence threshold epsilon for Level 2.
    """

    def __init__(
        self,
        dim_g: int,
        n_samples: int,
        batch_size: int,
        max_iter: int = 20,
        tol: float = 1e-8,
        random_seed: int | None = None,
    ):
        """Initialize the SHARD attacker.

        Args:
            dim_g: DLA dimension.
            n_samples: Dataset size N.
            batch_size: Batch size B.
            max_iter: Maximum SHARD iterations for Level 2.
            tol: Convergence threshold epsilon for Level 2.
        """
        self.dim_g = dim_g
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tol = tol
        self.rng = np.random.default_rng(random_seed)

    def level1_mean_recovery(
        self,
        coeff_matrices: list[np.ndarray],
        batch_gradients: list[list[np.ndarray]],
    ) -> np.ndarray:
        """Level 1: Recover dataset mean snapshot via stacked least squares.

        Constructs the stacked system ``C_stack = A_stack @ e_bar_data`` where:
        - ``A_stack`` is the vertical stack of all epoch coefficient matrices,
          shape ``(E*D, dim_g)``.
        - ``C_stack`` is the concatenation of epoch-averaged gradients,
          shape ``(E*D,)``.

        The epoch-averaged gradient for epoch *e* is the mean of all K batch
        gradients: ``c_e = (1/K) * sum(g^(e,k) for k in range(K))``.
        By the epoch invariant this equals ``A^(e) @ e_bar_data`` (noiseless).

        Args:
            coeff_matrices: List of E coefficient matrices, each of shape
                ``(D, dim_g)``.
            batch_gradients: List of E lists, each containing K gradient
                vectors of shape ``(D,)``.

        Returns:
            Recovered mean snapshot ``e_bar_data`` of shape ``(dim_g,)``.
        """
        E = len(coeff_matrices)
        D = coeff_matrices[0].shape[0]

        # Warn if the system is underdetermined
        if E * D < self.dim_g:
            warnings.warn(
                f"Underdetermined system: E*D={E * D} < dim_g={self.dim_g}. "
                "Recovery may be inaccurate; proceeding with least-norm solution.",
                stacklevel=2,
            )

        # Build A_stack by vertically stacking all coefficient matrices
        A_stack = np.vstack(coeff_matrices)  # (E*D, dim_g)

        # Build C_stack by computing epoch-averaged gradients and concatenating
        c_epochs = []
        for epoch_grads in batch_gradients:
            # epoch_grads is a list of K gradient vectors, each of shape (D,)
            c_e = np.mean(epoch_grads, axis=0)  # (D,)
            c_epochs.append(c_e)

        # For large systems, use normal equations (A^T A x = A^T c) which
        # avoids building the full stacked matrix and is much faster.
        if E * D > 2 * self.dim_g and E * D * self.dim_g > 5_000_000:
            # Normal equations approach
            AtA = np.zeros((self.dim_g, self.dim_g))
            Atc = np.zeros(self.dim_g)
            for e in range(E):
                A_e = coeff_matrices[e]
                AtA += A_e.T @ A_e
                Atc += A_e.T @ c_epochs[e]
            try:
                result = np.linalg.solve(AtA, Atc)
            except np.linalg.LinAlgError:
                warnings.warn(
                    "Normal-equation solve was singular; falling back to "
                    "least-squares recovery for the dataset mean.",
                    stacklevel=2,
                )
                A_stack = np.vstack(coeff_matrices)
                C_stack = np.concatenate(c_epochs)
                result, _, rank, _ = np.linalg.lstsq(A_stack, C_stack, rcond=None)
            else:
                rank = np.linalg.matrix_rank(AtA)
        else:
            A_stack = np.vstack(coeff_matrices)  # (E*D, dim_g)
            C_stack = np.concatenate(c_epochs)  # (E*D,)

            # Check condition number and warn if ill-conditioned
            cond = np.linalg.cond(A_stack)
            if cond > 1e10:
                warnings.warn(
                    f"A_stack condition number is {cond:.2e} (> 1e10). "
                    "Least-squares solution may be numerically unstable.",
                    stacklevel=2,
                )

            # Solve via least squares: A_stack @ e_bar_data = C_stack
            result, residuals, rank, sv = np.linalg.lstsq(A_stack, C_stack, rcond=None)

        logger.info(
            "Level 1 mean recovery: E=%d, D=%d, system shape=(%d, %d), rank=%s",
            E,
            D,
            E * D,
            self.dim_g,
            rank,
        )

        return result  # (dim_g,)

    def level2_disaggregate(
        self,
        e_bar_data: np.ndarray,
        coeff_matrices: list[np.ndarray],
        batch_gradients: list[list[np.ndarray]],
        true_snapshots: np.ndarray | None = None,
    ) -> np.ndarray:
        """Level 2: Two-phase SHARD disaggregation for individual recovery.

        Recovers individual snapshot vectors from batched, reshuffled gradients
        using a two-phase approach:

        **Phase 1 — Batch Mean Recovery**: For each epoch *e* and batch *k*,
        recover the batch mean snapshot by solving the linear system
        ``A^(e) @ m_k = g^(e,k)``.  When ``D >= dim_g`` this is exact.
        All K batch means per epoch are solved in a single batched operation.

        **Phase 2 — Individual Disaggregation**: Alternating minimization
        on the recovered batch means:
        (a) **Assignment**: For each epoch, find the partition of N snapshots
            into K groups of B that best matches the observed batch means.
            Uses Hungarian initialization followed by vectorized pairwise
            swap refinement.
        (b) **Update**: Given assignments, solve the normal equations
            ``(P^T P) S = P^T M`` via Cholesky factorization.

        For ``B=1``, a direct-solve fast path is used.

        Args:
            e_bar_data: Recovered mean snapshot of shape ``(dim_g,)``.
            coeff_matrices: List of E coefficient matrices, each ``(D, dim_g)``.
            batch_gradients: List of E lists, each containing K gradient
                vectors of shape ``(D,)``.
            true_snapshots: Optional ground-truth snapshot matrix ``(N, dim_g)``
                used only for logging matching accuracy at each iteration.

        Returns:
            Recovered snapshot matrix S of shape ``(N, dim_g)``.
        """
        N = self.n_samples
        B = self.batch_size
        K = N // B
        E = len(coeff_matrices)
        D = coeff_matrices[0].shape[0]
        dim_g = self.dim_g

        # === B=1 fast path ===
        if B == 1:
            return self._level2_b1_direct(
                e_bar_data, coeff_matrices, batch_gradients, true_snapshots
            )

        # === Two-phase approach for B > 1 ===
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        # --- Phase 1: Recover batch means (batched per epoch) ---
        batch_means = np.empty((E, K, dim_g))
        for e in range(E):
            A_e = coeff_matrices[e]
            G_e = np.array(batch_gradients[e])  # (K, D)
            # When D == dim_g, A_e is square — use solve (O(n³) vs SVD).
            if D == dim_g:
                batch_means[e] = np.linalg.solve(A_e, G_e.T).T
            else:
                batch_means[e] = np.linalg.lstsq(A_e, G_e.T, rcond=None)[0].T

        logger.info(
            "Phase 1: Recovered %d×%d batch means (D=%d, dim_g=%d)",
            E, K, D, dim_g,
        )

        # Flatten batch means for vectorized residual computation
        M_flat = batch_means.reshape(E * K, dim_g)  # (E*K, dim_g)

        # --- Phase 2: Alternating minimization on batch means ---
        norms = np.linalg.norm(batch_means - e_bar_data, axis=2)  # (E, K)
        sigma_init = max(float(norms.mean()) * 0.5, 0.1)

        S = self.rng.normal(loc=e_bar_data, scale=sigma_init, size=(N, dim_g))
        S -= (S.mean(axis=0) - e_bar_data)

        alpha = 0.6
        best_S = S.copy()
        best_residual = np.inf

        # Progress bar
        iter_range = range(1, self.max_iter + 1)
        if tqdm is not None:
            pbar = tqdm(
                iter_range, desc="Level 2", unit="iter",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}, {rate_fmt}]",
            )
        else:
            pbar = iter_range

        for t in pbar:
            S_old = S.copy()

            # === Assignment Step: vectorized partition matching ===
            # Returns (E, N) array where assign_arr[e, i] = batch index
            # For high-dimensional problems, project to lower dimension for
            # the assignment step (swap refinement) to avoid O(N²×dim_g) cost,
            # then do the full-dim update.
            if dim_g > 500:
                # Random projection to ~200 dims for assignment only
                if t == 1:
                    proj_dim = min(200, dim_g)
                    _rng_proj = np.random.default_rng(42)
                    R = _rng_proj.standard_normal((dim_g, proj_dim)) / np.sqrt(proj_dim)
                S_proj = S @ R
                bm_proj = batch_means @ R  # (E, K, proj_dim)
                ebar_proj = e_bar_data @ R
                assign_arr = self._assign_all_epochs(
                    S_proj, bm_proj, B, ebar_proj,
                    max_swap_rounds=3,
                )
            else:
                assign_arr = self._assign_all_epochs(
                    S, batch_means, B, e_bar_data,
                    max_swap_rounds=5,
                )

            # === Update Step: normal equations (P^T P) S = P^T M ===
            # Build P^T P (N×N) and P^T M (N×dim_g) directly — avoids
            # materializing the full (E*K × N) P matrix and the expensive
            # lstsq on (E*K, N) with dim_g RHS columns.
            PtP = np.zeros((N, N))
            PtM = np.zeros((N, dim_g))
            w = 1.0 / B
            w2 = w * w
            for e in range(E):
                ae = assign_arr[e]  # (N,) batch indices
                for k_idx in range(K):
                    members = np.where(ae == k_idx)[0]
                    # P^T P contribution: w² for all pairs in same batch
                    for ii in members:
                        for jj in members:
                            PtP[ii, jj] += w2
                    # P^T M contribution: w * m_k for each member
                    for ii in members:
                        PtM[ii] += w * batch_means[e, k_idx]

            S_new = np.linalg.lstsq(PtP, PtM, rcond=None)[0]

            # Damped update
            S = alpha * S_new + (1.0 - alpha) * S_old
            S -= (S.mean(axis=0) - e_bar_data)

            # === Residual (computed from assignments) ===
            total_residual = 0.0
            for e in range(E):
                ae = assign_arr[e]
                for k_idx in range(K):
                    members = np.where(ae == k_idx)[0]
                    pred_mean = S[members].mean(axis=0)
                    total_residual += float(
                        np.sum((pred_mean - batch_means[e, k_idx]) ** 2)
                    )

            if total_residual < best_residual:
                best_residual = total_residual
                best_S = S.copy()

            # === Convergence Check ===
            frob_change = np.linalg.norm(S - S_old, "fro")

            if true_snapshots is not None:
                acc = self._matching_accuracy(S, true_snapshots)
                logger.info(
                    "Iteration %d/%d: Matching Acc = %s, residual = %.2e",
                    t, self.max_iter, f"{acc:.1%}", total_residual,
                )
                if tqdm is not None and hasattr(pbar, "set_postfix_str"):
                    pbar.set_postfix_str(
                        f"acc={acc:.0%} res={total_residual:.1e}"
                    )
            else:
                logger.info(
                    "Iteration %d/%d: Frobenius change = %.2e, "
                    "residual = %.2e",
                    t, self.max_iter, frob_change, total_residual,
                )
                if tqdm is not None and hasattr(pbar, "set_postfix_str"):
                    pbar.set_postfix_str(
                        f"ΔS={frob_change:.1e} res={total_residual:.1e}"
                    )

            if frob_change < self.tol:
                logger.info(
                    "Converged at iteration %d "
                    "(||ΔS||_F = %.2e < tol=%.2e)",
                    t, frob_change, self.tol,
                )
                break
        else:
            warnings.warn(
                f"SHARD did not converge within {self.max_iter} iterations "
                f"(final ||ΔS||_F = {frob_change:.2e}).",
                stacklevel=2,
            )

        if tqdm is not None and hasattr(pbar, "close"):
            pbar.close()

        return best_S

    def _level2_b1_direct(
        self,
        e_bar_data: np.ndarray,
        coeff_matrices: list[np.ndarray],
        batch_gradients: list[list[np.ndarray]],
        true_snapshots: np.ndarray | None = None,
    ) -> np.ndarray:
        """B=1 fast path: direct solve per epoch + cross-epoch matching.

        With B=1, each batch gradient ``g^(e,k) = A^(e) @ s_{perm(k)}``.
        We solve each gradient per-epoch (exactly when ``D >= dim_g``,
        via least-squares otherwise), match solutions across epochs via
        Hungarian algorithm, then combine via normal equations.

        Args:
            e_bar_data: Recovered mean snapshot of shape ``(dim_g,)``.
            coeff_matrices: List of E coefficient matrices.
            batch_gradients: List of E lists of K gradient vectors.
            true_snapshots: Optional ground-truth for logging.

        Returns:
            Recovered snapshot matrix S of shape ``(N, dim_g)``.
        """
        N = self.n_samples
        E = len(coeff_matrices)
        D = coeff_matrices[0].shape[0]
        dim_g = self.dim_g

        # Step 1: Per-epoch solutions
        epoch_solutions = []
        for e in range(E):
            A_e = coeff_matrices[e]
            if D == dim_g:
                sols = np.array([
                    np.linalg.solve(A_e, batch_gradients[e][k])
                    for k in range(N)
                ])
            else:
                sols = np.array([
                    np.linalg.lstsq(A_e, batch_gradients[e][k], rcond=None)[0]
                    for k in range(N)
                ])
            epoch_solutions.append(sols)

        # Step 2: Match across epochs using epoch 0 as reference
        ref = epoch_solutions[0]
        epoch_perms = [list(range(N))]
        for e in range(1, E):
            sols = epoch_solutions[e]
            r_sq = np.sum(ref ** 2, axis=1, keepdims=True)
            s_sq = np.sum(sols ** 2, axis=1, keepdims=True)
            cost = r_sq + s_sq.T - 2.0 * ref @ sols.T
            row_ind, col_ind = linear_sum_assignment(cost)
            perm = [0] * N
            for r, c in zip(row_ind, col_ind):
                perm[c] = r
            epoch_perms.append(perm)

        # Step 3: Solve with combined epochs via normal equations
        # LHS = Σ_e A_e^T A_e, RHS = Σ_e A_e^T g_e (reordered)
        LHS = np.zeros((dim_g, dim_g))
        for e in range(E):
            LHS += coeff_matrices[e].T @ coeff_matrices[e]

        RHS = np.zeros((dim_g, N))
        for e in range(E):
            At_e = coeff_matrices[e].T
            perm = epoch_perms[e]
            for k in range(N):
                RHS[:, perm[k]] += At_e @ batch_gradients[e][k]

        # Use lstsq for robustness when LHS may be rank-deficient
        S = np.linalg.lstsq(LHS, RHS, rcond=None)[0].T

        if true_snapshots is not None:
            acc = self._matching_accuracy(S, true_snapshots)
            logger.info("B=1 direct solve: Matching Acc = %s", f"{acc:.1%}")

        return S

    @staticmethod
    def _assign_partition_swap(
        S: np.ndarray,
        batch_means_epoch: np.ndarray,
        B: int,
        e_bar: np.ndarray,
        max_swap_rounds: int = 5,
    ) -> list[list[int]]:
        """Find a partition of snapshots into batches via Hungarian + swaps.

        Assigns N snapshots to K batches of size B by:
        1. Computing a cost matrix based on centered inner products between
           candidate snapshots and observed batch means.
        2. Solving the expanded (N×N) linear assignment problem.
        3. Refining via pairwise swap local search: repeatedly swap pairs
           of snapshots between different batches if it reduces the total
           squared residual between predicted and observed batch means.

        Args:
            S: Current snapshot estimates, shape ``(N, dim_g)``.
            batch_means_epoch: Observed batch means for one epoch,
                shape ``(K, dim_g)``.
            B: Batch size.
            e_bar: Dataset mean snapshot, shape ``(dim_g,)``.
            max_swap_rounds: Maximum number of full sweep rounds for
                swap refinement.

        Returns:
            List of K lists, each containing B snapshot indices.
        """
        N = S.shape[0]
        K = N // B
        dim_g = S.shape[1]

        # --- Hungarian initialization ---
        S_c = S - e_bar
        M_c = batch_means_epoch - e_bar
        sim = S_c @ M_c.T  # (N, K)

        # Expand to (N, N) bipartite matching: B slots per batch
        cost_expanded = np.empty((N, N))
        for k in range(K):
            cost_expanded[:, k * B:(k + 1) * B] = (-sim[:, k])[:, None]

        row_ind, col_ind = linear_sum_assignment(cost_expanded)
        perm = np.empty(N, dtype=int)
        perm[col_ind] = row_ind

        # Build assignment array: assignment[i] = batch index for snapshot i
        assignment = np.empty(N, dtype=int)
        for k in range(K):
            for b_pos in range(B):
                assignment[perm[k * B + b_pos]] = k

        # Compute batch sums
        batch_sums = np.zeros((K, dim_g))
        for i in range(N):
            batch_sums[assignment[i]] += S[i]

        # Per-batch residual: ||sum_k/B - m_k||²
        batch_res = np.sum(
            (batch_sums / B - batch_means_epoch) ** 2, axis=1,
        )

        # --- Swap refinement ---
        for _ in range(max_swap_rounds):
            improved = False
            for i in range(N):
                ki = assignment[i]
                sum_ki_without_i = batch_sums[ki] - S[i]
                best_delta = 0.0
                best_j = -1

                for j in range(N):
                    kj = assignment[j]
                    if ki == kj:
                        continue

                    new_sum_ki = sum_ki_without_i + S[j]
                    new_sum_kj = batch_sums[kj] - S[j] + S[i]

                    new_res_ki = np.sum(
                        (new_sum_ki / B - batch_means_epoch[ki]) ** 2
                    )
                    new_res_kj = np.sum(
                        (new_sum_kj / B - batch_means_epoch[kj]) ** 2
                    )
                    delta = (new_res_ki + new_res_kj) - (
                        batch_res[ki] + batch_res[kj]
                    )

                    if delta < best_delta - 1e-15:
                        best_delta = delta
                        best_j = j

                if best_j >= 0:
                    j = best_j
                    kj = assignment[j]
                    batch_sums[ki] = batch_sums[ki] - S[i] + S[j]
                    batch_sums[kj] = batch_sums[kj] - S[j] + S[i]
                    assignment[i], assignment[j] = kj, ki
                    batch_res[ki] = np.sum(
                        (batch_sums[ki] / B - batch_means_epoch[ki]) ** 2
                    )
                    batch_res[kj] = np.sum(
                        (batch_sums[kj] / B - batch_means_epoch[kj]) ** 2
                    )
                    improved = True

            if not improved:
                break

        # Convert to list-of-lists
        epoch_assign: list[list[int]] = [[] for _ in range(K)]
        for i in range(N):
            epoch_assign[assignment[i]].append(i)
        return epoch_assign

    def _assign_all_epochs(
        self,
        S: np.ndarray,
        batch_means: np.ndarray,
        B: int,
        e_bar: np.ndarray,
        max_swap_rounds: int = 5,
    ) -> np.ndarray:
        """Assign snapshots to batches for all epochs (vectorized).

        For each epoch, performs Hungarian initialization followed by
        pairwise swap refinement. The swap inner loop is vectorized:
        for each snapshot *i*, the cost delta for swapping with ALL
        snapshots *j* in different batches is computed via broadcasting.

        Args:
            S: Current snapshot estimates, shape ``(N, dim_g)``.
            batch_means: Observed batch means, shape ``(E, K, dim_g)``.
            B: Batch size.
            e_bar: Dataset mean snapshot, shape ``(dim_g,)``.
            max_swap_rounds: Maximum swap refinement rounds per epoch.

        Returns:
            Assignment array of shape ``(E, N)`` where
            ``result[e, i]`` is the batch index for snapshot *i* in epoch *e*.
        """
        N = S.shape[0]
        K = N // B
        E = batch_means.shape[0]
        dim_g = S.shape[1]

        assign_arr = np.empty((E, N), dtype=int)

        # Precompute centered snapshots for Hungarian step
        S_c = S - e_bar  # (N, dim_g)

        for e in range(E):
            bm_e = batch_means[e]  # (K, dim_g)

            # --- Hungarian initialization ---
            M_c = bm_e - e_bar  # (K, dim_g)
            sim = S_c @ M_c.T  # (N, K)

            # Expand to (N, N) bipartite: B slots per batch
            cost_expanded = np.empty((N, N))
            for k in range(K):
                cost_expanded[:, k * B:(k + 1) * B] = (-sim[:, k])[:, None]

            row_ind, col_ind = linear_sum_assignment(cost_expanded)
            perm = np.empty(N, dtype=int)
            perm[col_ind] = row_ind

            assignment = np.empty(N, dtype=int)
            for k in range(K):
                for b_pos in range(B):
                    assignment[perm[k * B + b_pos]] = k

            # --- Vectorized swap refinement ---
            # Precompute batch sums and targets
            batch_sums = np.zeros((K, dim_g))
            for i in range(N):
                batch_sums[assignment[i]] += S[i]

            targets = bm_e * B  # (K, dim_g) — scaled targets

            batch_res = np.sum(
                (batch_sums - targets) ** 2, axis=1,
            )  # (K,)

            for _ in range(max_swap_rounds):
                improved = False
                for i in range(N):
                    ki = assignment[i]
                    si = S[i]  # (dim_g,)

                    # Vectorized: compute delta for swapping i with every j
                    # where assignment[j] != ki
                    diff_mask = assignment != ki  # (N,) bool
                    if not diff_mask.any():
                        continue

                    j_indices = np.where(diff_mask)[0]
                    kj_arr = assignment[j_indices]  # batch indices of j's

                    # New sums after swap i<->j:
                    # batch ki: batch_sums[ki] - S[i] + S[j]
                    # batch kj: batch_sums[kj] - S[j] + S[i]
                    sj = S[j_indices]  # (len_j, dim_g)

                    new_sum_ki = batch_sums[ki] - si + sj  # (len_j, dim_g)
                    new_sum_kj = (
                        batch_sums[kj_arr] - sj + si
                    )  # (len_j, dim_g)

                    new_res_ki = np.sum(
                        (new_sum_ki - targets[ki]) ** 2, axis=1,
                    )
                    new_res_kj = np.sum(
                        (new_sum_kj - targets[kj_arr]) ** 2, axis=1,
                    )

                    old_res = batch_res[ki] + batch_res[kj_arr]
                    delta = (new_res_ki + new_res_kj) - old_res

                    best_idx = np.argmin(delta)
                    if delta[best_idx] < -1e-15:
                        j = j_indices[best_idx]
                        kj = assignment[j]
                        batch_sums[ki] = batch_sums[ki] - si + S[j]
                        batch_sums[kj] = batch_sums[kj] - S[j] + si
                        assignment[i], assignment[j] = kj, ki
                        batch_res[ki] = np.sum(
                            (batch_sums[ki] - targets[ki]) ** 2,
                        )
                        batch_res[kj] = np.sum(
                            (batch_sums[kj] - targets[kj]) ** 2,
                        )
                        improved = True

                if not improved:
                    break

            assign_arr[e] = assignment

        return assign_arr


    def level3_invert(
        self,
        snapshots: np.ndarray,
        surrogate: "SurrogateQFL",
        device: str | torch.device | None = None,
    ) -> np.ndarray:
        """Level 3: Invert snapshots to input space via batch Adam + L-BFGS.

        For each recovered snapshot ``s_i``, solves:

            min_x ||cos(W_enc @ x + b_enc) - s_i||^2

        Strategy: run Adam on a batch of random starting points in parallel
        (all as rows of a single tensor), then polish the best result with
        L-BFGS.  Inputs are clamped to [0, 1] after each step.

        Args:
            snapshots: Recovered snapshot matrix of shape ``(N, dim_g)``.
            surrogate: :class:`SurrogateQFL` instance that provides the
                ``encode`` method and the encoding parameters ``W_enc``
                and ``b_enc``.
            device: Torch device for computation (e.g. ``"mps"``,
                ``"cuda"``, ``"cpu"``).  If ``None``, uses CPU.

        Returns:
            Reconstructed input vectors of shape ``(N, d)`` as a NumPy array.
        """
        N = snapshots.shape[0]
        d = surrogate.W_enc.shape[1]  # input dimension

        dev = torch.device(device if device is not None else "cpu")

        W_enc = surrogate.W_enc.detach().to(dev)
        b_enc = surrogate.b_enc.detach().to(dev)
        dim_g = W_enc.shape[0]

        try:
            from tqdm import tqdm as _tqdm
        except ImportError:
            _tqdm = None

        reconstructed = np.empty((N, d), dtype=np.float64)

        # Scale parallel starts based on problem size to manage memory.
        # For small d (< 500), use 500 starts. For large d, reduce.
        n_batch = max(50, min(500, 500_000 // max(d, dim_g)))
        adam_steps = 2000
        adam_lr = 0.01

        snap_range = range(N)
        if _tqdm is not None:
            snap_range = _tqdm(snap_range, desc="Level 3", unit="snap",
                               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

        for i in snap_range:
            target = torch.tensor(snapshots[i], dtype=W_enc.dtype, device=dev)

            # --- Phase 1: Batch Adam on many random starts ---
            restart_rng = torch.Generator().manual_seed(42 + i)

            # Initialize: gray image (0.5) + lstsq seeds + N(0,0.1) random
            extra_inits: list[torch.Tensor] = []
            extra_inits.append(torch.full((d,), 0.5, dtype=W_enc.dtype))

            with torch.no_grad():
                # lstsq seeds must be computed on CPU (MPS doesn't support lstsq)
                W_cpu = surrogate.W_enc.detach()
                s_clamped = torch.clamp(
                    torch.tensor(snapshots[i], dtype=W_cpu.dtype),
                    -1.0 + 1e-7, 1.0 - 1e-7,
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
                    extra_inits.append(sol.clamp(0.0, 1.0))

            n_extra = len(extra_inits)
            n_random = n_batch - n_extra

            # Random starts: uniform in [0, 1]^d
            random_starts = torch.rand(
                n_random, d, dtype=W_enc.dtype, generator=restart_rng
            )
            X = torch.cat(
                [torch.stack(extra_inits), random_starts], dim=0
            ).to(dev)  # (n_batch, d)
            X = X.detach().clone().requires_grad_(True)

            adam = torch.optim.Adam([X], lr=adam_lr)
            for step in range(adam_steps):
                adam.zero_grad()
                snaps = torch.cos(X @ W_enc.T + b_enc)  # (n_batch, dim_g)
                recon_loss = ((snaps - target) ** 2).sum(dim=1)  # (n_batch,)
                loss = recon_loss.sum()
                loss.backward()
                adam.step()
                with torch.no_grad():
                    X.clamp_(0.0, 1.0)
                if recon_loss.min().item() < 1e-8:
                    break

            # --- Phase 2: L-BFGS polish on the best candidate ---
            with torch.no_grad():
                snaps = torch.cos(X @ W_enc.T + b_enc)
                losses = ((snaps - target) ** 2).sum(dim=1)
                best_idx = int(losses.argmin())

            # L-BFGS on CPU (more numerically stable, MPS L-BFGS can be flaky)
            W_lbfgs = surrogate.W_enc.detach()
            b_lbfgs = surrogate.b_enc.detach()
            target_cpu = torch.tensor(snapshots[i], dtype=W_lbfgs.dtype)

            x = X[best_idx].detach().cpu().clone().requires_grad_(True)
            optimizer = torch.optim.LBFGS(
                [x],
                max_iter=500,
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
                snap_x = torch.cos(W_lbfgs @ x + b_lbfgs)
                best_residual = float(((snap_x - target_cpu) ** 2).sum())

            best_x = x.detach().clone()

            if best_residual > 1e-4:
                logger.warning(
                    "L-BFGS did not converge for snapshot %d: "
                    "residual = %.2e",
                    i,
                    best_residual,
                )

            reconstructed[i] = best_x.numpy().astype(np.float64)

        logger.info(
            "Level 3 inversion complete: %d snapshots inverted to "
            "input dimension %d",
            N,
            d,
        )

        return reconstructed


    @staticmethod
    def _matching_accuracy(
        recovered: np.ndarray, true: np.ndarray
    ) -> float:
        """Compute matching accuracy between recovered and true snapshots.

        Uses the Hungarian algorithm on the pairwise squared-distance matrix
        to find the optimal one-to-one assignment, then reports the fraction
        of matched pairs whose distance is below a threshold (1% of mean
        pairwise distance).

        Args:
            recovered: Recovered snapshot matrix ``(N, dim_g)``.
            true: Ground-truth snapshot matrix ``(N, dim_g)``.

        Returns:
            Fraction of correctly matched samples in [0, 1].
        """
        r_sq = np.sum(recovered ** 2, axis=1, keepdims=True)
        t_sq = np.sum(true ** 2, axis=1, keepdims=True)
        dist_matrix = r_sq + t_sq.T - 2.0 * recovered @ true.T
        np.maximum(dist_matrix, 0.0, out=dist_matrix)

        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        matched_dists = dist_matrix[row_ind, col_ind]

        mean_dist = float(np.mean(dist_matrix))
        threshold = 0.01 * mean_dist if mean_dist > 0 else 1e-12

        n_correct = int(np.sum(matched_dists < threshold))
        return float(n_correct / len(row_ind))
