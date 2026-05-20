"""LASA-QTERM production attack — terminal snapshot recovery for QFL.

Method name: **LASA-QTERM** (LASA linear terminal snapshot recovery).
Alias: **Q-SNAP-T** (quantum snapshot from terminals).

Wraps the best *honest* tier-specific estimators from Round 03 without
mean-imputed partial rows or cross-tier headline conflation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from terminal_attacks import (
    active_probe_graph_terminal,
    b1_budget_disaggregate,
    cross_epoch_consistency_terminal,
    graph_rank_terminal,
    graph_term_map,
    partial_epoch_gradients,
    partial_honest_disaggregate,
    passive_mean_broadcast,
)


class QtermTier(str, Enum):
    """Observation tier for LASA-QTERM."""

    T1 = "T1"  # epoch-averaged terminal only (impossibility for individuals)
    T1P = "T1p"  # honest partial terminal (last p minibatch rows / epoch)
    T1B = "T1b"  # B=1 per-client terminal rows (no within-epoch steps)


# Default hyperparameters from Round-03 smooth ablation winners.
DEFAULT_GRAPH_LAMBDA = 10.0
DEFAULT_SPREAD_SCALE = 0.5
DEFAULT_PARTIAL_ROWS = 7
DEFAULT_GRAPH_LAMBDA_PARTIAL = 0.35


@dataclass
class QtermResult:
    """Recovery output and observation accounting."""

    snapshots: np.ndarray
    tier: str
    method: str
    observed_terminal_gradient_rows: int
    observed_intermediate_batch_gradients: int
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class QtermConfig:
    """LASA-QTERM attack configuration."""

    tier: QtermTier = QtermTier.T1P
    n_samples: int = 32
    batch_size: int = 4
    dim_g: int = 32
    true_rank: int = 4
    graph_lambda: float = DEFAULT_GRAPH_LAMBDA
    spread_scale: float = DEFAULT_SPREAD_SCALE
    partial_rows_per_epoch: int = DEFAULT_PARTIAL_ROWS
    partial_graph_lambda: float = DEFAULT_GRAPH_LAMBDA_PARTIAL
    partial_max_iter: int = 80
    gradient_row_budget: int | None = None  # T1b: subsample to this many rows
    random_seed: int = 0


class QtermAttack:
    """Production LASA-QTERM attacker for SurrogateQFL / SHARD-compatible inputs."""

    def __init__(self, config: QtermConfig | None = None) -> None:
        self.config = config or QtermConfig()

    def recover(
        self,
        e_bar: np.ndarray,
        coeff_matrices: list[np.ndarray],
        *,
        terminal_gradients: list[np.ndarray] | None = None,
        batch_gradients: list[list[np.ndarray]] | None = None,
        b1_gradients: list[list[np.ndarray]] | None = None,
    ) -> QtermResult:
        """Recover individual snapshot matrix ``S`` from tier-appropriate observations.

        Parameters
        ----------
        e_bar
            Level-1 mean snapshot (``ShardAttacker.level1_mean_recovery``).
        coeff_matrices
            Per-epoch LASA coefficient matrices ``A^(e)``.
        terminal_gradients
            T1: one epoch-averaged gradient per epoch (length E).
        batch_gradients
            T1p: full or partial minibatch rows per epoch (B>1).
        b1_gradients
            T1b: per-client B=1 rows per epoch.
        """
        cfg = self.config
        n = cfg.n_samples
        tier = cfg.tier

        if tier == QtermTier.T1:
            return self._recover_t1(e_bar, coeff_matrices, terminal_gradients, n)
        if tier == QtermTier.T1P:
            if batch_gradients is None:
                raise ValueError("T1p requires batch_gradients")
            return self._recover_t1p(e_bar, coeff_matrices, batch_gradients, n)
        if tier == QtermTier.T1B:
            if b1_gradients is None:
                raise ValueError("T1b requires b1_gradients")
            return self._recover_t1b(e_bar, coeff_matrices, b1_gradients, n)
        raise ValueError(f"Unknown tier: {tier}")

    def _recover_t1(
        self,
        e_bar: np.ndarray,
        coeff_matrices: list[np.ndarray],
        terminal_gradients: list[np.ndarray] | None,
        n_samples: int,
    ) -> QtermResult:
        """Best honest T1 stack: graph terminal + cross-epoch tie-break."""
        cfg = self.config
        s_graph = graph_term_map(
            e_bar,
            n_samples,
            graph_lambda=cfg.graph_lambda,
            spread_scale=cfg.spread_scale,
        )
        if terminal_gradients is not None and len(terminal_gradients) > 0:
            s_cross = cross_epoch_consistency_terminal(
                e_bar,
                coeff_matrices,
                terminal_gradients,
                n_samples,
                rank=cfg.true_rank,
                graph_lambda=min(cfg.graph_lambda, 0.35),
            )
            s_active = active_probe_graph_terminal(
                e_bar,
                coeff_matrices,
                terminal_gradients,
                n_samples,
                graph_lambda=cfg.graph_lambda,
                spread_scale=cfg.spread_scale,
            )
            candidates = [
                ("graph_term_map", s_graph),
                ("cross_epoch_consistency", s_cross),
                ("active_probe_graph", s_active),
            ]
        else:
            candidates = [("graph_term_map", s_graph)]

        # Production pick: graph_term_map (Round-03 best T1 on smooth/MNIST).
        method, snapshots = candidates[0]
        n_epochs = len(coeff_matrices) if coeff_matrices else 0
        return QtermResult(
            snapshots=snapshots,
            tier=QtermTier.T1.value,
            method=f"lasa_qterm_{method}",
            observed_terminal_gradient_rows=n_epochs,
            observed_intermediate_batch_gradients=0,
            meta={
                "graph_lambda": cfg.graph_lambda,
                "spread_scale": cfg.spread_scale,
                "impossibility_note": (
                    "T1 identifies s_bar only; see paper/impossibility_t1.md"
                ),
            },
        )

    def _recover_t1p(
        self,
        e_bar: np.ndarray,
        coeff_matrices: list[np.ndarray],
        batch_gradients: list[list[np.ndarray]],
        n_samples: int,
    ) -> QtermResult:
        """Imputation-free partial terminal (LASA-QTERM partial mode)."""
        cfg = self.config
        p = cfg.partial_rows_per_epoch
        partial = partial_epoch_gradients(batch_gradients, rows_per_epoch=p)
        n_obs = sum(len(g) for g in partial)
        rng = np.random.default_rng(cfg.random_seed + 9200 + p)
        snapshots = partial_honest_disaggregate(
            e_bar,
            coeff_matrices,
            partial,
            n_samples,
            cfg.batch_size,
            graph_lambda=cfg.partial_graph_lambda,
            max_iter=cfg.partial_max_iter,
            rng=rng,
        )
        return QtermResult(
            snapshots=snapshots,
            tier=QtermTier.T1P.value,
            method="lasa_qterm_partial_honest",
            observed_terminal_gradient_rows=n_obs,
            observed_intermediate_batch_gradients=0,
            meta={
                "rows_per_epoch": p,
                "imputation_free": True,
                "partial_graph_lambda": cfg.partial_graph_lambda,
            },
        )

    def _recover_t1b(
        self,
        e_bar: np.ndarray,
        coeff_matrices: list[np.ndarray],
        b1_gradients: list[list[np.ndarray]],
        n_samples: int,
    ) -> QtermResult:
        """B=1 per-client terminals; optional budget subsample."""
        cfg = self.config
        coeff = coeff_matrices
        grads = b1_gradients
        budget = cfg.gradient_row_budget
        if budget is not None:
            from terminal_attacks import subsample_gradient_rows

            coeff, grads, used = subsample_gradient_rows(
                coeff_matrices, b1_gradients, max_rows=budget
            )
            n_rows = used
            method = "lasa_qterm_b1_budget"
        else:
            n_rows = sum(len(g) for g in b1_gradients)
            method = "lasa_qterm_b1_full"

        snapshots = b1_budget_disaggregate(
            e_bar, coeff, grads, n_samples, cfg.dim_g
        )
        return QtermResult(
            snapshots=snapshots,
            tier=QtermTier.T1B.value,
            method=method,
            observed_terminal_gradient_rows=n_rows,
            observed_intermediate_batch_gradients=0,
            meta={
                "gradient_row_budget": budget,
                "non_acceptance_for_primary_t1": budget is None
                or n_rows > 10,
            },
        )


def recover_passive_baseline(e_bar: np.ndarray, n_samples: int) -> np.ndarray:
    """T1 passive lower bound (mean broadcast)."""
    return passive_mean_broadcast(e_bar, n_samples)


def recover_graph_rank_baseline(
    e_bar: np.ndarray, n_samples: int, dim_g: int, rank: int = 4
) -> np.ndarray:
    """T1 graph low-rank ablation baseline."""
    return graph_rank_terminal(e_bar, n_samples, dim_g, rank=rank)
