# Round 02 — Revision log

## Context

Supervisor Round 01 verdict: **REVISE_MAJOR**. Ten mandatory fixes in `rounds/round_01/supervisor_review.md` (assignment/graph stress, honest threat reduction, mean-anchor ablation, baselines, ASSIGN-LOCK stub, SHIELD grid, LEAK-CERT park).

## What we did

- Added `code/benchmark_round02.py` (ported assignment/graph stress from parent `code/benchmark_round05.py`).
- Re-ran full benchmark → `artifacts/round02_metrics.json`, `logs/experiment_round02.log`.
- Wrote `rounds/round_02/researcher_proposal.md` with pass/fail vs config A/B/C.

## 85% claim — **withdrawn**

| Statistic | Round 01 (inflated) | Round 02 (parent-aligned) |
|-----------|--------------------:|-------------------------:|
| Gate | any seed @ MSE ≤ 0.10 | **mean MSE ≤ 0.10** across seeds at minimum rows |
| Min rows (GARD oracle graph) | 9/60 | **15/60** |
| Row reduction | **85%** | **75%** |

`claim_withdrawal.round01_headline_85pct_withdrawn: true` in `round02_metrics.json`. The 85% figure remains only as a **diagnostic** at MSE ≤ 0.15 (mean gate), not the headline.

## Criterion A (pre-registered gate: `parent_mean_mse_at_minimum_rows`)

| Assignment | @ MSE 0.10 | ≥25% reduction? |
|------------|------------|-----------------|
| oracle | 15 rows, **75%** | **Yes** (conditional replication) |
| wrong20 | does not reach target | **No** |
| wrong40 | does not reach target | **No** |
| unknown_random | does not reach target | **No** |

**Criterion A does not hold under wrong assignment** for any tested regime at MSE 0.10.

## Other lanes

- **QFL-SHIELD:** rank×σ grid (20 configs × 5 seeds); **0** cells meet ≤10% normalized utility; kill criteria met → deprioritized.
- **LEAK-CERT:** parked to Round 03+ (tier-specific T1p bound at 7 rows/epoch).
- **ASSIGN-LOCK:** stub implemented; permuted slot order raises LS MSE ~1.2–1.4 vs oracle GARD ~0.6–0.8 at 15 rows.

## Pivot vs persist

- **Persist (narrowed):** GARD-SPARSE as **conditional** Stage-2 replication under oracle assignment + graph side information; focus Round 03 on ASSIGN-LOCK + robust graph under wrong incidence.
- **Deprioritize:** QFL-SHIELD until a feasible utility-feasible cell exists.
- **Park:** LEAK-CERT.

## Next round (preview)

- Harden ASSIGN-LOCK (permutation + incidence hiding jointly).
- Co-occurrence / LS-kNN graph under wrong20 (partial recovery).
- Optional: prefix-epoch protocol from parent Round 05.
