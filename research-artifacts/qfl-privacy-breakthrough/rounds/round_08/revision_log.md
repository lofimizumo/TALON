# Round 08 Revision Log

## Integration (QFL-PRIVACY-MAP)

- Added `paper/privacy_map.md` — defender/auditor roles + tier table.
- Added `code/qprivacy_map_core.py` — tier audits (T1, T1p, S2-oracle, S2-wrong40).
- Added `code/qprivacy_audit.py` — CLI over `round08_metrics.json`.
- Added `code/benchmark_round08.py` — reproducible audit bundle (~10s).

## LEAK-CERT honesty fix

- Criterion **C** now uses `cert_naive_broadcast` only (Round 04 honest naive).
- Trace-inflated naive retained in JSON for gaming detection vs Round 07.
- Documented in `privacy_map.md` and `leak_cert_fix` block in metrics JSON.

## Theorem / threat links

- **ABT-1:** `paper/assignment_barrier_theorem.md` — wrong40 barrier tier.
- **GARD-SPARSE 75%:** Round 01 oracle @ fraction 0.25 (15/60 rows).
- **T1:** `qfl-terminal-snapshot/paper/impossibility_t1.md`.

## Scientist-only

No `supervisor_review.md` for Round 08.

## Command

```bash
python3 research-artifacts/qfl-privacy-breakthrough/code/benchmark_round08.py
```
