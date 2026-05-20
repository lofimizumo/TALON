# Round 04 — Revision log

## Context

No `rounds/round_03/supervisor_review.md`; pivoted from Round 03 proposal primary lanes (ASSIGN-LOCK, JASPER-style soft assignment, T1p track, LEAK-CERT).

## What we did

- Added `code/benchmark_round04.py` with JASPER-Q, ASSIGN-LOCK v2, LEAK-CERT-T1p.
- Re-ran benchmark → `artifacts/round04_metrics.json`, `logs/experiment_round04.log`.
- Wrote `rounds/round_04/researcher_proposal.md`.
- Set `config.json` `current_round` → 4.

## Parent / sibling inputs

- Parent Round 06 JASPER negative: entropy regularization + damped Sinkhorn retained; added uniform entropy blend.
- `qfl-terminal-snapshot` Round 04: T1p **70 rows**, mean MSE **0.476** used as external reference in LEAK-CERT aggregate.

## Criterion snapshot

| Criterion | Result |
|-----------|--------|
| A (25% rows @ MSE 0.10, JASPER-Q) | **Fail** |
| B (defense 50% @ 10% utility) | **N/A** |
| C (cert 2× tighter than naive) | **Fail** (1.89×; 1/5 seeds cert < empirical) |

## Mechanism outcomes

- **JASPER-Q:** Large wrong-H gain vs GARD; not criterion-A grade; oracle path regresses.
- **ASSIGN-LOCK v2:** Hungarian recovery **ineffective** (~13% perm accuracy).
- **LEAK-CERT-T1p:** Implemented tier-specific bound; **does not** yet satisfy 2× tightening with full empirical coverage.

## Next round (preview)

- Conditional T1p warm-start (wrong-H only).
- LEAK-CERT: worst-case seed coverage or tier gate in config.
- ASSIGN-LOCK: incidence hiding + secure permutation (not gradient matching alone).
