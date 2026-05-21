# Round 08 — Researcher Proposal: QFL-PRIVACY-MAP Integration

## Goal

Consolidate Rounds 01–07 into **QFL-PRIVACY-MAP**, a defender/auditor framework with a single tier table, reproducible audit bundle, and CLI — without relitigating T1 impossibility or claiming wrong40 breakthrough.

## Deliverables

| Artifact | Path |
|----------|------|
| Privacy map (paper) | `paper/privacy_map.md` |
| Tier audit core | `code/qprivacy_map_core.py` |
| Auditor CLI | `code/qprivacy_audit.py` |
| Reproducible bundle | `code/benchmark_round08.py` → `artifacts/round08_metrics.json` |

**No** `supervisor_review.md` (Round 08 scientist-only per charter).

## Tier table

| Tier | Status | Round-08 headline |
|------|--------|-------------------|
| **T1** | IMPOSSIBLE | Terminal-only MSE ≈ 1.0 |
| **T1p** | CERT + honest C **fail** | LEAK-CERT v2 covers emp; naive/cert ≈ 0.93× (not 2×) |
| **S2-oracle** | CONDITIONAL THREAT | GARD-SPARSE **75%** row reduction @ MSE ≤ 0.10 |
| **S2-wrong40** | BARRIER (ABT-1) | Mean MSE ≥ 0.15 @ 25–50% rows |

## LEAK-CERT fix (Round 07 gaming)

Round 07 criterion **C** passed only with a **trace-inflated** naive baseline (~16× vs cert). Round 08 uses **honest naive**: broadcast epoch-mean gradient to every snapshot (`cert_naive_broadcast`). Under honest naive, **C fails** while certificate **coverage** still holds — the cert is valid; the 2× tightening claim was inflated.

## Config gates (honest)

- **A:** Pass on oracle GARD-SPARSE (parent mean @ 0.10, 15/60 rows).
- **B:** Fail (unchanged; Snapshot-DP killed R05).
- **C:** Fail under honest naive; pass retained only for diagnostic trace baseline.
- **ABT-1:** Pass on wrong40 @ 25% and 50% fractions.

## Reproduction

```bash
cd research-artifacts/qfl-privacy-breakthrough
python3 code/benchmark_round08.py
python3 code/qprivacy_audit.py --bundle artifacts/round08_metrics.json
```
