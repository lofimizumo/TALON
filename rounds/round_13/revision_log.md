# Round 13 — Revision log

## Supervisor Round 12 requirements → actions

| Requirement | Action | Outcome |
|---|---|---|
| Fix `passive_mb_scale_only` | Added `passive_mb_scale_only_estimate_sums` (R11 path) | Mean **0.151** on primary — matches R11 ~0.15 |
| Correct `paper/proofs.md` drift | 0.42 → **0.112**; drift2 superiority removed | Doc aligned with `lemma_mb_b_empirical` |
| New fundamental directions (≥2) | JOINT, COUPLED, DOPT, trajectory midpoint | **All fail** vs TANGO-MB on primary; honestly reported |
| Keep minibatch_sgd primary + median/IQR | Unchanged scenario; `aggregate_robust` retained | Gate still **met** |
| No `supervisor_review.md` | Not created | — |

## Code changes

| File | Change |
|---|---|
| `code/benchmark_round13.py` | New benchmark (4 estimators + honest ablations) |
| `paper/proofs.md` | Drift fix, Lemma MB-JOINT, implementation table |
| `paper/draft.md` | Round 13 results + honest ~14× scaling narrative |
| `config.json` | `current_round: 13`, `PHASE2_ROUND13_NEW_DIRECTIONS` |

## Pivot / naming

- **TANGO-JOINT** — kept name; negative result documented (not promoted to primary).
- **TANGO-COUPLED** — 3-iter fixed point; does not beat MB.
- Primary remains **`tango_mb`** for Phase-2 reporting.

## Still open for Round 14+

- SHARD `level3_invert` matched terminal observation comparison
- General FL / representation drift beyond frozen features
- Approximate tier gap (0.011 vs 1.5e-4 full-batch)
