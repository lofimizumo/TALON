# Root cause: Round 03–04 experiment stall (2026-05-21)

## Symptoms

- `benchmark_round03.py` ran **>5 hours** with no log lines after snapshot recovery.
- Stray `lapin_invert(32×784)` test ran **~6 hours** at ~68% CPU.
- `benchmark_round04.py` stuck at startup line for **~50+ minutes** on first `dim_g=160` block.

## Root cause 1 — LAPIN normal equations scale as **O(d³)**

`lapin_invert_single` builds `JTJ` of shape `(d, d)` each Gauss–Newton step.

| Setting | d | Time / image (measured) |
|---------|---|-------------------------|
| Default `n_random=48`, `max_gn=25` | 784 | **~112 s** |
| Same | 64 | **~0.08 s** |
| Budget `n_random=4`, `max_gn=8` | 784 | **~25 s** (still heavy) |

For **N=32** clients: one LAPIN pass ≈ **32 × 112 s ≈ 59 min** (default) or **~13 min** (budget).

Round 03 called LAPIN once per seed (cached across adam grid), but **blocked the whole seed worker** before any JOLI results were logged.

**Fix:** `lapin_budget(d)` in `code/lapin_invert.py`; skip LAPIN on 28×28 for primary grid or use budget; optional diagnostic-only LAPIN on 8×8.

## Root cause 2 — JOLI batch Adam is **O(N × n_batch × adam_steps × d × dim_g)**

Round 03 used `n_batch = shard_n_batch(784, 100) ≈ 500` and `adam_steps ∈ {2000, 4000, 6000}` on **32 images sequentially**.

Per seed: **3** full JOLI grids × 32 images ≈ **96** inversions, each up to 500×6000 matmuls → **hours** on CPU.

Round 04 used lighter `adam=250`, `lbfgs=150`, `n_batch cap 500` but **3 snapshot paths × 3 seeds** without snapshot caching across paths → repeated expensive L3 on similar recoveries.

**Fix:** `l3_budget.joli_profile()` caps `n_batch` at 128, single adam budget for resume, `adam_grid=(250,)` for completion; Round 5+ quality profile with `adam=400` and no LAPIN on 784-d.

## Root cause 3 — **Parallel seed workers** hid progress

`ProcessPoolExecutor(max_workers=3)` ran three full stacks concurrently:

- Triple RAM (~750 MB+ each)
- CPU contention
- Parent log only showed snapshot lines; per-seed logs not flushed during L3

**Fix:** `parallel_seed_workers=1` for 28×28; flush logging; log **per image** index during L3.

## Root cause 4 — No **time budget / estimate** before long jobs

Benchmarks started multi-hour grids without printing estimated cost.

**Fix:** `estimate_lapin_seconds` / `estimate_joli_seconds` logged at seed start; refuse LAPIN on d>128 unless `--allow-slow-lapin`.

## Not the root cause

- Deadlock (processes were CPU-bound at 58–68%)
- MNIST download (snapshots completed in minutes)
- Incorrect math in snapshot recovery (T1p snap MSE was good)

## Acceptance path after fix

1. Complete Round 03–04 with **resume profile** (JOLI-only primary on 28×28).
2. Round 5+: experiment-focused improvements (dim_g sweep, L3 quality, recon grids).
3. Targets: input MSE ≤ 0.05, PSNR ≥ 18 on ≥2/3 seeds — still hard at d=784; may need 14×14 fallback or stronger snapshots.
