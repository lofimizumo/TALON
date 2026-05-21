# Round 01 — Revision log

## Context

First round of `qfl-privacy-breakthrough` (10-round plan). No prior `supervisor_review.md`.

## What we did

- Surveyed five lanes from `literature/prior_findings_bridge.md`; implemented three in `code/benchmark_round01.py`.
- Reused `vendor/shard_sim` via `code/_paths.py` and sibling `qfl-terminal-snapshot/code/qterm_attack.py`.
- Ran benchmark → `artifacts/round01_metrics.json`, `logs/experiment_round01.log`.

## Pivot vs persist

- **Persist:** GARD-SPARSE — reproduces parent sparse-row win; meets config criterion (A).
- **Pivot (design, not abandon):** QFL-SHIELD — first config harmed privacy; keep lane for R02 with tuned rank/noise and utility-normalized metric.
- **Pivot:** LEAK-CERT — dof heuristic insufficient at T1p row counts; need rank-aware or sensitivity certificate.

## Deferred

- ASSIGN-LOCK, PROBE-DETECT: survey-only until Round 02.

## Next round (preview)

- Joint sparse Stage-2 + assignment stress (`wrong20`, co-occurrence graph).
- Shield ablation grid with ≤10% utility constraint.
- Certificate tied to observation tier (T1 vs T1p row budget).
