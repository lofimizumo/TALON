# QFL-PRIVACY-MAP — 10-round privacy breakthrough run

**Parent:** TALON (`/workspace`) · **Sibling:** LASA-QTERM (`../qfl-terminal-snapshot/`)

## Status

**ACCEPT** (Round 10 supervisor, 2026-05-20) — integrated **defender/auditor** framework after 10 strict rounds (`forbid_accept_before_round: 8` enforced).

## Goal

Breakthrough **QFL privacy-preserving** research: SHARD Stage-2 improvements or other measurable privacy wins—not limits-only essays.

## Two breakthrough signals (accepted scope)

1. **Privacy floor:** Assignment barrier theorem (**ABT-1**) + LASA-QTERM **T1 impossibility** (epoch terminals leak mean only).
2. **Conditional threat:** **GARD-SPARSE** — **75%** fewer observed batch rows vs full SHARD LS at MSE ≤ 0.10 when assignment + graph are oracle-aligned (held-out seeds replicated).

## What failed (kill list)

| Lane | Outcome |
|------|---------|
| GARD under wrong40/unknown | MSE floor ~0.9–1.0 — barrier, not fixable by JASPER-Q |
| Snapshot-DP / SHIELD | 0/420 cells @ ≤10% utility |
| ASSIGN-LOCK | Permutation recovery ~13% |
| Trace-inflated LEAK-CERT | Gaming rejected (Round 07) |
| PROBE-RAND | Does not break JASPER (~12% worse only) |

## Reproduce

```bash
cd /workspace/research-artifacts/qfl-privacy-breakthrough
python3 code/benchmark_round10.py
python3 code/qprivacy_audit.py --bundle artifacts/round08_metrics.json
```

## Key docs

| Path | Content |
|------|---------|
| `paper/final_privacy_breakthrough.md` | Final synthesis |
| `paper/privacy_map.md` | Tier audit map |
| `paper/assignment_barrier_theorem.md` | ABT-1 |
| `tutorial/tutorial.md` | Practitioner guide |

## Round verdicts

| Round | Verdict |
|-------|---------|
| 1–5 | REVISE_MAJOR |
| 6–9 | REVISE_MAJOR |
| 10 | **ACCEPT** |
