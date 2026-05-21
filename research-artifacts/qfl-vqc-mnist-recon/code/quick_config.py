"""Fast-iteration run configuration for QFL-VQC MNIST reconstruction.

Set ``QFL_QUICK=1`` (default for dev) or ``QFL_FULL=1`` for acceptance-grade 28×28 runs.

Quick mode targets **~5–15 minutes** per benchmark (not hours).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from l3_budget import L3Profile, shard_n_batch_capped


@dataclass(frozen=True)
class RunConfig:
    mode: str
    resize: int
    n_samples: int
    batch_size: int
    n_epochs: int
    seeds: tuple[int, ...]
    dim_g_list: tuple[int, ...]
    snapshot_paths: tuple[str, ...]
    l3_profile: L3Profile
    shard_max_iter: int
    partial_rows: int
    grid_seed: int
    estimate_label: str

    @property
    def d(self) -> int:
        return self.resize * self.resize


def is_quick_mode() -> bool:
    if os.environ.get("QFL_FULL") == "1":
        return False
    return os.environ.get("QFL_QUICK", "1") != "0"


def get_run_config(*, research_round: int = 0) -> RunConfig:
    if is_quick_mode():
        d = 14 * 14
        profile = L3Profile(
            name="quick_iter",
            adam_steps=120,
            lbfgs_iter=80,
            n_batch_cap=48,
            include_lapin=False,
            lapin_only_if_d_le=0,
            adam_grid=(120,),
            parallel_seed_workers=1,
            log_every_image=True,
        )
        return RunConfig(
            mode="quick",
            resize=14,
            n_samples=12,
            batch_size=4,
            n_epochs=5,
            seeds=(3, 7),
            dim_g_list=(100,),
            snapshot_paths=("lasa_qterm_T1p_graph", "shard_oracle"),
            l3_profile=profile,
            shard_max_iter=80,
            partial_rows=6,
            grid_seed=3,
            estimate_label="~5-12 min/benchmark (12 clients, 2 seeds, 14×14, 2 paths)",
        )
    d = 28 * 28
    profile = L3Profile(
        name="full_mnist28",
        adam_steps=250,
        lbfgs_iter=200,
        n_batch_cap=128,
        include_lapin=False,
        lapin_only_if_d_le=128,
        adam_grid=(250,),
        parallel_seed_workers=1,
        log_every_image=True,
    )
    dim_list = (160,) if research_round >= 5 and os.environ.get("QFL_FULL_DIM_SWEEP") != "1" else (100, 160, 256)
    paths = ("lasa_qterm_T1p", "gard_sparse_oracle", "shard_oracle")
    return RunConfig(
        mode="full",
        resize=28,
        n_samples=32,
        batch_size=4,
        n_epochs=10,
        seeds=(3, 7, 11),
        dim_g_list=dim_list,
        snapshot_paths=paths,
        l3_profile=profile,
        shard_max_iter=200,
        partial_rows=8,
        grid_seed=7,
        estimate_label="hours (see paper/root_cause_l3_stall.md); use QFL_FULL=1 only for acceptance",
    )


def patch_module_globals(mod, *, research_round: int = 0) -> RunConfig | None:
    """Overwrite benchmark module knobs when ``QFL_QUICK=1`` (not ``QFL_FULL=1``)."""
    if not is_quick_mode():
        return None
    cfg = get_run_config(research_round=research_round)
    prof = cfg.l3_profile
    mapping = {
        "RESIZE": cfg.resize,
        "N_SAMPLES": cfg.n_samples,
        "BATCH_SIZE": cfg.batch_size,
        "N_EPOCHS": cfg.n_epochs,
        "SEEDS": list(cfg.seeds),
        "PARTIAL_ROWS": cfg.partial_rows,
        "SHARD_MAX_ITER": cfg.shard_max_iter,
        "GRID_SEED": cfg.grid_seed,
    }
    for name, val in mapping.items():
        if hasattr(mod, name):
            setattr(mod, name, val)
    if hasattr(mod, "RESIZE_28"):
        mod.RESIZE_28 = cfg.resize if cfg.mode == "full" else 28
        mod.RESIZE_14 = cfg.resize
    if hasattr(mod, "DIM_G"):
        mod.DIM_G = cfg.dim_g_list[0]
    if hasattr(mod, "DIM_G_PRIMARY"):
        mod.DIM_G_PRIMARY = cfg.dim_g_list[0]
    if hasattr(mod, "DIM_G_ALT"):
        mod.DIM_G_ALT = cfg.dim_g_list[0]
    if hasattr(mod, "ADAM_STEPS_GRID"):
        mod.ADAM_STEPS_GRID = prof.adam_grid
    if hasattr(mod, "LBFGS_ITER"):
        mod.LBFGS_ITER = prof.lbfgs_iter
    if hasattr(mod, "JOLI_ADAM_STEPS"):
        mod.JOLI_ADAM_STEPS = prof.adam_steps
    if hasattr(mod, "JOLI_LBFGS_ITER"):
        mod.JOLI_LBFGS_ITER = prof.lbfgs_iter
    if hasattr(mod, "JOLI_N_BATCH_CAP"):
        mod.JOLI_N_BATCH_CAP = prof.n_batch_cap
    if hasattr(mod, "L3_INVERTERS") and not prof.include_lapin:
        mod.L3_INVERTERS = tuple(
            x for x in getattr(mod, "L3_INVERTERS", ()) if "lapin" not in x
        ) or ("joli_l3",)
    if hasattr(mod, "SNAPSHOT_PATHS"):
        mod.SNAPSHOT_PATHS = cfg.snapshot_paths
    if hasattr(mod, "PASS_MIN_SEEDS"):
        mod.PASS_MIN_SEEDS = min(2, len(cfg.seeds))
    return cfg


def estimate_benchmark_seconds(cfg: RunConfig) -> float:
    """Rough wall-clock for one unified quick/full benchmark."""
    n_l3 = (
        len(cfg.seeds)
        * len(cfg.dim_g_list)
        * len(cfg.snapshot_paths)
        * cfg.n_samples
        * cfg.l3_profile.adam_steps
        * 0.002
    )
    snap = len(cfg.seeds) * len(cfg.dim_g_list) * cfg.n_epochs * 2.0
    return snap + n_l3
