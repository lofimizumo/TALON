# Round 03 Supervisor Review (post-stall recovery)

## Verdict

**REVISE_MAJOR** (experiments incomplete at review time; resume in progress)

---

## Executive summary

Round 03’s **original** run stalled for **>5 hours** after snapshot recovery. Root cause is **not** a deadlock: processes were CPU-saturated on **LAPIN** and **heavy JOLI** at **d=784**. Documented in `paper/root_cause_l3_stall.md`. Stalled PIDs were **killed**; benchmark patched with `l3_budget.py` and dimension-aware `lapin_budget()`.

Snapshot stage **succeeded** before stall (T1p snap MSE ~0.02–0.07 on seeds 3,7,11). Image L3 metrics were **never written** to `round03_metrics.json` in the aborted run.

---

## Root cause audit (verified by profiling)

| Cause | Evidence | Fix |
|-------|----------|-----|
| LAPIN **O(d³)** GN solves | **~112 s/image** at d=784, default `n_random=48`, `max_gn=25` | `lapin_budget()` → ~25 s/image; **skip LAPIN on 28×28 primary grid** |
| JOLI **500×6000** Adam grid | 3 adam budgets × 32 sequential images × 3 parallel seeds | `adam_grid=(250,)`, `n_batch_cap=128`, **sequential seeds** |
| No progress logs | Parent log silent during L3 | Per-image `joli image i/N` logging |
| Misleading parallelism | 3× RAM/CPU contention | `parallel_seed_workers=1` for 28×28 |

---

## Round 03 science (partial)

- SHARD B=1 oracle snapshots: excellent (snap MSE ≈ 0–0.004).
- T1p snapshots: good (≈ 0.02–0.07) — **snapshot path viable** for L3.
- Image targets (MSE≤0.05, PSNR≥18): **pending** resumed benchmark.

---

## Actionable demands (Round 04–05)

1. Complete `benchmark_round03.py` with fixed profile → `artifacts/round03_metrics.json`.
2. Complete `benchmark_round04.py` with per-image logging (same stall risk on 3 paths × 3 seeds).
3. Report **per-seed** pass rates, not mean-only.
4. Primary weak path: **T1p + JOLI-R3** on 28×28 `dim_g=160`.
5. Do **not** re-enable full LAPIN grid on d=784 without budget flag.

**NO ACCEPT** until 28×28 image metrics exist and ≥2/3 seeds pass OR honest 14×14 fallback documented.
