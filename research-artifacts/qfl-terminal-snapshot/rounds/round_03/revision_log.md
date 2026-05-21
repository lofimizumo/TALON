# Round 03 — Revision log

## Status

Response to `rounds/round_02/supervisor_review.md` (**REVISE_MAJOR**). No `supervisor_review.md` written (per mission).

## Changes from Round 02

### Code

- `code/terminal_attacks.py` — `graph_lambda` wired; `partial_honest_disaggregate`, `b1_budget_disaggregate`, `active_probe_graph_terminal`, `cross_epoch_consistency_terminal`, `subsample_gradient_rows`.
- `code/benchmark_round03.py` — budget-80 SHARD vs B=1, honest vs padded partial, tier headlines, new T1 attacks.
- `config.json` — `current_round: 3`.

### Paper

- `paper/impossibility_t1.md` — formal T1 impossibility under stacked \(c^{(e)} = A^{(e)}\bar{s}\).

### Artifacts

- `artifacts/round03_metrics.json` — `headline_by_tier`, smooth + mnist.
- `logs/experiment_round03.log`.

### Literature

- `literature/qfl_terminal_snapshot.md` — Round 03 impossibility, budget-80, honest partial.

## Supervisor issue mapping

| Issue | Round 03 response |
|-------|-------------------|
| Headline B=1 conflation | `headline_by_tier`; no global `best_terminal_method` |
| Partial imputation dishonest | `partial_honest_*` without padding; padded labeled upper bound |
| Comparable budget ignored | `budget80_matched` block (80 vs 80 rows) |
| `graph_lambda` unused | Ridge scaling; best λ now varies (mean 10.0 on smooth) |
| Formal T1 impossibility | `paper/impossibility_t1.md` |
| SHARD matching 0% | Documented in T2 tier; Hungarian MSE still primary metric |

## Pivot vs persist

**Persist** individual-snapshot goal for **T1p / T1b** tiers. **Accept impossibility** for strict **T1** (10 epoch terminals). Do not claim config acceptance from B=1 or padded partial alone.

## Next round

- JASPER-TERMINAL on honest partial rows without full K imputation.
- Partial budget-80 honest tier (equal rows to SHARD without per-client channel).
- SHARD matching audit / iteration budget.
