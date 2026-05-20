# Round 07 — Researcher Proposal: JASPER-Q v7 + LEAK-CERT trace bound

## Context (Rounds 04–05)

Round 04 delivered **JASPER-Q** diagnostic gains under `wrong40` (MSE ~0.96 @ 25% rows) via T1p warm-start, but **no** criterion **A** or **C**. Round 05 **Snapshot-DP** failed criterion **B**; assignment lanes remain primary.

## Mission

Extend JASPER-Q with three mechanisms and re-test relaxed assignment gates:

| Extension | Description |
|-----------|-------------|
| **Multi-epoch warm-start** | Cumulative T1p `QtermAttack` per epoch; exponential weights favor later epochs |
| **Spectral graph** | Blend co-occurrence Laplacian (wrong **H**) with kNN Laplacian on **recovered** snapshot rows |
| **Row-fraction sweep** | `{0.25, 0.35, 0.50, 0.65, 0.75}` on `wrong40` and `oracle` only |
| **Conditional warm-start** | T1p blend **0** on `oracle`, **0.65** on `wrong40` (Round 04/05 recommendation) |

**LEAK-CERT-T1p v2:** Ky-Fan singular-value tail + trace floor; **trace-inflated** naive baseline for criterion **C** (≥2× `naive_trace / cert` with full empirical coverage).

## Pre-registered success (Round 07)

| Gate | Definition |
|------|------------|
| **Relaxed A** | Parent mean MSE ≤ **0.15** with ≥**50%** row reduction (`wrong40` or `oracle`, level1) |
| **Relaxed wrong40** | Parent mean MSE ≤ **0.10** with ≥**40%** row reduction (`wrong40`, level1) |
| **C (trace)** | All seeds: `cert ≥ 0.95 × empirical` and `naive_trace / cert ≥ 2` |

Standard `config.json` criterion **A** @ 0.10 / 25% reduction remains reported for continuity.

## Implementation

- `code/benchmark_round07.py`
- Seeds: `{3, 7, 11, 19, 23}`
- Anchor: `level1_estimate` only

## Deliverables

- `artifacts/round07_metrics.json`
- `logs/experiment_round07.log`
- `rounds/round_07/revision_log.md`

**No** `supervisor_review.md` this round (scientist-only execution).

## Results (post-run)

### JASPER-Q v7 @ level1 (snapshot MSE mean)

| wrong40 frac | JASPER-Q v7 | GARD |
|-------------:|------------:|-----:|
| 0.25 | 0.949 | 2.653 |
| 0.50 | 0.904 | 2.365 |
| 0.75 | **0.888** | 1.112 |

Oracle @0.75: **0.669** (T1p blend=0). Pre-registered relaxed gates **not met** (MSE still ≫0.10–0.15 at feasible row cuts).

### LEAK-CERT-T1p v2 (criterion C)

| Metric | Value |
|--------|------:|
| Empirical T1p MSE (mean) | 0.424 |
| Cert trace upper (mean) | 1.08 |
| Naive trace-inflated (mean) | 16.0 |
| `naive_trace/cert` | **14.8×** |
| Covers empirical | **5/5** |

Criterion **C** passes on trace-inflated naive baseline; assignment criterion **A** remains **fail**.
