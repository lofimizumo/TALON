"""Central L3 compute budgets for QFL-VQC MNIST reconstruction benchmarks.

Root cause (Round 03 stall): LAPIN at d=784 with n_random=48, max_gn=25 costs
~112 s/image (53 GN solves with 784×784 normal equations). Round 03 also ran
JOLI with n_batch=500 and adam_steps∈{2000,4000,6000} sequentially on 32 images
×3 seeds in parallel → multi-hour wall clock with no progress logs after snapshot stage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class L3Profile:
    name: str
    adam_steps: int
    lbfgs_iter: int
    n_batch_cap: int
    include_lapin: bool
    lapin_only_if_d_le: int | None
    adam_grid: tuple[int, ...]
    parallel_seed_workers: int
    log_every_image: bool


def shard_n_batch_capped(d: int, dim_g: int, cap: int) -> int:
    raw = max(50, min(500, 500_000 // max(d, dim_g)))
    return min(raw, cap)


def joli_profile(d: int, *, research_round: int) -> L3Profile:
    """Profiles keyed by experiment round after root-cause fix."""
    import os

    if os.environ.get("QFL_QUICK", "1") != "0" and os.environ.get("QFL_FULL") != "1":
        from quick_config import get_run_config

        return get_run_config(research_round=research_round).l3_profile
    if d <= 64:
        return L3Profile(
            name="small_image",
            adam_steps=2000,
            lbfgs_iter=500,
            n_batch_cap=500,
            include_lapin=True,
            lapin_only_if_d_le=9999,
            adam_grid=(2000, 4000),
            parallel_seed_workers=2,
            log_every_image=True,
        )
    if research_round >= 5:
        # Profiled ~50-90 s/image at d=784, dim_g=160, adam=250, n_batch=128 on CPU.
        return L3Profile(
            name="mnist28_quality",
            adam_steps=250,
            lbfgs_iter=200,
            n_batch_cap=128,
            include_lapin=False,
            lapin_only_if_d_le=128,
            adam_grid=(250,),
            parallel_seed_workers=1,
            log_every_image=True,
        )
    # Rounds 3–4 resume profile
    return L3Profile(
        name="mnist28_resume",
        adam_steps=250,
        lbfgs_iter=150,
        n_batch_cap=128,
        include_lapin=True,
        lapin_only_if_d_le=128,
        adam_grid=(250,),
        parallel_seed_workers=1,
        log_every_image=True,
    )


def estimate_lapin_seconds(d: int, n_images: int, dim_g: int = 100) -> float:
    """Empirical fit from profiling 2026-05-21."""
    if d <= 128:
        per = 0.1
    elif d <= 400:
        per = 2.0
    else:
        per = 25.0  # after lapin_budget() reduction from ~112s
    return per * n_images


def estimate_joli_seconds(
    d: int, dim_g: int, n_images: int, adam_steps: int, n_batch_cap: int
) -> float:
    # Empirical: ~0.2–2 s/img @ 14×14 quick; ~50–90 s/img @ 28×28 adam=250 n_batch=128
    if d <= 256:
        per_image = 0.15 * (adam_steps / 120.0) * (d / 196.0) * (dim_g / 100.0)
    else:
        ref = 87.0
        n_batch = shard_n_batch_capped(d, dim_g, n_batch_cap)
        per_image = ref * (adam_steps / 250.0) * (n_batch / 128.0) * (d / 784.0) * (dim_g / 160.0)
    return max(5.0, per_image * n_images)
