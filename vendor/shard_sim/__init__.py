"""SHARD Attack Simulation Package.

Public API:
    FederatedDataLoader - Dataset loading and batching
    SurrogateQFL - Classical surrogate model for gradient generation
    ShardAttacker - 3-level SHARD attack pipeline
    compute_matching_accuracy - Hungarian matching accuracy
    compute_reconstruction_mse - Reconstruction error metric
    plot_reconstruction_grid - Visualization of results
    run_shard_pipeline - End-to-end pipeline execution
"""

from .data_loader import FederatedDataLoader
from .surrogate_model import SurrogateQFL
from .attacker import ShardAttacker
from .metrics import compute_matching_accuracy, compute_reconstruction_mse
from .visualization import plot_reconstruction_grid
from .pipeline import run_shard_pipeline
