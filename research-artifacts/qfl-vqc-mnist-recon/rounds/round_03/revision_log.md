# Round 03 revision log

**Date:** 2026-05-20  
**Mode:** experiment-only (no supervisor review)

## Changes

| File | Description |
|------|-------------|
| `code/benchmark_round03.py` | 28×28 T1p L3 grid: `adam_steps` {2k,4k,6k}, `n_batch` max, `lbfgs` 1000; JOLI + LAPIN; B=1 SHARD oracle via `level2_disaggregate_b1`; per-seed pass table; recon grids. |

## Design notes

- Parent `code/benchmark_round03.py` Pareto informed `tv_lbfgs=0.005` for compressive JOLI.
- `level2_disaggregate_b1` did not exist in vendor; implemented on `ShardAttackerR03` as B=1 wrapper.
- LAPIN ignores Adam grid (GN + L-BFGS polish); executed once per seed, replicated to all `adam_steps` keys (`adam_independent: true`).
- Seeds run in parallel (`ProcessPoolExecutor`, up to `os.cpu_count()` workers) for wall-clock tractability on CPU.

## Outputs

- `artifacts/round03_metrics.json`
- `artifacts/round03_recon_grid_*.png`
- `logs/experiment_round03.log`
