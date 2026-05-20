# Round 02 — revision log

## Summary

Primary **28×28 MNIST** benchmark with `dim_g` sweep, SHARD L2 graph anchor, T1p terminal tuning, comparison grids, and per-seed pass rates.

## Changes

| Area | Action |
|------|--------|
| `code/benchmark_round02.py` | 28×28 E2E; `ShardAttackerR02` graph L2 init; T1p `partial_rows=8`, `partial_max_iter=150`; sweep `dim_g` 100/160/256; pass-rate aggregation; per-dim grids. |
| `rounds/round_02/` | Researcher proposal + this log (no `supervisor_review.md`). |

## Rationale (from Round 01 metrics)

- Oracle SHARD L2 often failed to converge at `max_iter=200` (seeds 7, 11).
- T1p lagged T1b on input MSE; improve terminal observation without B=1 row budget in headline grid.
- Config requires 28×28; dim_g sweep locates compressive regime for JOLI.

## Pivot vs persist

**Persist** SurrogateQFL + SHARD + LASA-QTERM + JOLI; scale resolution and tune snapshot stage before accepting.

## Post-run notes (Round 02 experiments)

- Full **28×28** sweep completed (~79 min CPU): `artifacts/round02_metrics.json`, `logs/experiment_round02.log`.
- **No seed** met both targets (MSE ≤ 0.05, PSNR ≥ 18) at any `dim_g`; best compressive path **T1b + JOLI @ dim_g=256** (mean input MSE 0.081, PSNR 10.9 dB).
- **T1p** (80 terminal rows/epoch, no B=1 flood) beats **oracle** on mean input MSE with JOLI at all swept `dim_g`; e.g. dim_g=160: T1p 0.140 vs oracle 0.201.
- SHARD L2 still non-converges on hard shuffles (seed 7, 11) despite graph init + 500 iter.
- Grid: `artifacts/round02_recon_grid_dim160.png` (seed 7, oracle / T1p / T1b, JOLI L3).
