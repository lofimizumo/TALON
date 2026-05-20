# Round 06 Researcher Proposal

**Date:** 2026-05-20  
**Primary path:** Path 2 — relaxed row-reduction milestone (`wrong40` @ 50% rows)  
**Secondary path:** Path 3 — assignment barrier theorem (`ABT-1`)  
**Killed:** Snapshot-DP (Round 05)

## Pre-registered success (Path 2)

| Field | Value |
|-------|-------|
| Assignment | `wrong40` |
| Anchor | `level1_estimate` |
| Row fraction | **0.50** (30 / 60 rows) |
| Attackers | GARD co-occurrence **and** JASPER-Q (best-of) |
| JASPER fix | T1p warm-start **disabled** on `oracle` assignment |
| Gate | Mean snapshot MSE **≤ 0.15** on **≥ 4 / 5** seeds |
| Export | `A_wrong40_mse_0.15_at_50pct_rows` |

## Path 3 (theorem)

| Field | Value |
|-------|-------|
| Document | `paper/assignment_barrier_theorem.md` |
| ID | `ABT-1` |
| Link | Wrong incidence → rank deficiency → MSE floor **≥ 0.15** @ 25% rows (falsifiable) |

## Cross-cutting

- T1p audit: `ShardAttacker.level1_mean_recovery` (Round 05 fix; not primary gate)
- Mandatory: **used-vs-true** observed-batch residual MSE
- `config.json`: criterion B prose aligned to “attack MSE increase ≥ 50%”

## Methods

`code/benchmark_round06.py` — no Snapshot-DP grid.

## Deliverables

- `code/benchmark_round06.py`
- `artifacts/round06_metrics.json`
- `logs/experiment_round06.log`
- `rounds/round_06/revision_log.md`
- `paper/assignment_barrier_theorem.md`

**No** `supervisor_review.md` this round.
