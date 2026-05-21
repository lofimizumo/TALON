# Round 04 — Researcher Proposal: JASPER-Q, ASSIGN-LOCK v2, LEAK-CERT-T1p

## Context (Round 03 → 04)

Round 03 had no `supervisor_review.md`; this round follows `rounds/round_03/researcher_proposal.md` recommendations:

1. Soft assignment (parent TALON Round 06 JASPER negative → entropy reg + terminal warm-start)
2. ASSIGN-LOCK beyond permutation stub
3. Tier-specific LEAK-CERT at **70-row T1p** budget (`qfl-terminal-snapshot` LASA-QTERM mean MSE ≈ **0.476**)

## Methods implemented (`code/benchmark_round04.py`)

| # | Mechanism | Description |
|---|-----------|-------------|
| 1 | **JASPER-Q** | Joint Sinkhorn soft incidence + co-occurrence GARD; entropy blend toward uniform; **T1p warm-start** (65% blend of LASA-QTERM snapshots/mean); N=32 aligned with QFL track |
| 2 | **ASSIGN-LOCK v2** | Server permutes published batch-row order; attacker runs iterative **Hungarian** alignment of `m_pub` to identity slots, then LS + GARD |
| 3 | **LEAK-CERT-T1p** | Analytic upper bound at 7 rows/epoch × 10 epochs = **70** terminal rows; compared to naive mean broadcast and empirical T1p attack |

**Seeds:** 3, 7, 11, 19, 23. **Fractions:** 0.25, 0.15. **Assignments:** wrong20, wrong40, unknown_random, oracle. **Anchors:** level1_estimate, oracle_true.

## Pass/fail vs `config.json`

| Criterion | Config definition | Round 04 result | Pass? |
|-----------|-------------------|-----------------|-------|
| **A** | ≥25% row reduction @ equal snapshot MSE (parent mean gate @ 0.10) | JASPER-Q: oracle and wrong40 **do not** reach MSE ≤ 0.10 at any fraction | **Fail** |
| **B** | ≥50% attack MSE cut @ ≤10% utility | Not re-run (SHIELD kill Round 02) | **Fail / N/A** |
| **C** | Certificate ≥2× tighter than naive | Mean `naive/cert` ≈ **1.89** (<2×); seed 19 attack MSE **0.569** > cert **0.529** | **Fail** |

`forbid_accept_before_round: 8` — no acceptance claim.

## Numeric results (honest)

### JASPER-Q @ fraction 0.25, level1 anchor (snapshot MSE mean)

| Assignment | GARD co-occurrence | JASPER-Q | Hard overlap |
|------------|-------------------:|---------:|-------------:|
| wrong20 | 1.334 | **0.900** | 0.745 |
| wrong40 | 2.653 | **0.955** | 0.488 |
| unknown_random | 1.334 | **0.990** | 0.145 |
| oracle | 0.402 | 0.853 | 1.000 |

JASPER-Q **cuts** wrong-assignment MSE vs fixed-H GARD (e.g. wrong40: 2.65 → 0.96) but remains **far above** MSE ≤ 0.10. Under **oracle** assignment, T1p warm-start **hurts** vs GARD (~0.85 vs ~0.40) because terminal prior mismatches full Stage-2 incidence.

### ASSIGN-LOCK v2 @ fraction 0.25 (mean over seeds)

| Metric | Value |
|--------|------:|
| Permutation recovery accuracy | **12.9%** |
| v1 broken LS MSE | 1.428 |
| v2 recovered LS MSE | 1.428 (no gain) |
| Oracle GARD upper bound | 0.332 |

Hungarian row alignment **does not** break the assignment barrier without extra side information (same MSE as broken permutation).

### LEAK-CERT-T1p (70 rows)

| Metric | Mean |
|--------|-----:|
| Empirical T1p attack MSE | 0.439 |
| Naive mean broadcast | 1.000 |
| Cert tight upper bound | 0.529 |
| `naive/cert` ratio | **1.89×** (not 2×) |
| Cert covers empirical (per seed) | **4/5** seeds |

Reference: `qfl-terminal-snapshot` aggregate T1p MSE **0.476** @ 70 rows.

## Breakthrough status

```json
"assignment_barrier_broken": false
```

| Signal | Met? |
|--------|------|
| Criterion A (wrong40 / oracle, JASPER-Q) | **No** |
| Criterion B | **No / N/A** |
| Criterion C (2× tighter + valid upper bound) | **No** |
| JASPER-Q beats GARD wrong40 @ 0.25 | **Yes** (diagnostic; not criterion A) |
| ASSIGN-LOCK v2 recovers permutation | **No** |

## Primary lane recommendation (Round 05)

1. **Decouple T1p warm-start from oracle Stage-2** — use terminal channel only under wrong assignment; keep oracle path on level1/GARD only.
2. **ASSIGN-LOCK** needs cryptographic permutation + metadata hiding, not gradient-only Hungarian recovery.
3. **LEAK-CERT** — tighten bound to cover worst-seed T1p (≥0.57) or admit criterion C requires tier-specific acceptance table.
4. Continue **co-occurrence GARD** as upper bound; JASPER-Q as wrong-H diagnostic module.

## Artifacts

- `artifacts/round04_metrics.json`
- `logs/experiment_round04.log`
- `code/benchmark_round04.py`
- `rounds/round_04/revision_log.md`

## Answers

1. **Does JASPER-Q break the assignment barrier?** **No** for criterion A — MSE still ≫ 0.10 under wrong40/unknown; large improvement vs GARD is not enough.
2. **Does T1p warm-start help?** **Yes under wrong20/40** (large MSE drop vs GARD); **no under oracle** (warm-start degrades).
3. **Does ASSIGN-LOCK v2 recover permutation?** **No** (~13% accuracy; no MSE improvement).
4. **Does LEAK-CERT-T1p meet criterion C?** **No** — ratio 1.89× and cert fails to upper-bound seed 19 attack.
