# Round 03 — Researcher Proposal: Assignment-Barrier Mechanisms

## Problem & goal

Round 02 showed **75% row reduction @ MSE ≤ 0.10** only under **oracle assignment + oracle mean anchor**; **wrong20/40/unknown_random** fail the parent mean-MSE gate. Round 03 tests four mechanisms aimed at breaking that barrier without oracle side channels.

## Methods implemented (`code/benchmark_round03.py`)

| # | Mechanism | Description |
|---|-----------|-------------|
| 1 | **INCIDENCE-REFINE** | Alternate GARD+co-occurrence solve and greedy top-B row relabeling from batch means (10 iters; no true H in refine loop) |
| 2 | **HYBRID T1p + sparse GARD** | LASA-QTERM T1p (7 partial rows/epoch, true batch membership) + 25% sparse Stage-2 rows with **wrong H**; blend grid + T1p-mean anchor for sparse GARD |
| 3 | **DP-mean anchor** | Laplace noise on Level-1 batch-mean average (ε=1.0, ℓ₂ sensitivity 0.15) vs `level1_estimate` / `oracle_true` |
| 4 | **Co-occurrence under wrong H** | `gard_cooccurrence_wrong_h` vs `gard_oracle_chain_wrong_h` on same corrupted incidence |

**Seeds:** 3, 7, 11, 19, 23. **Barrier fractions:** 0.25, 0.15. **Assignments:** wrong20, wrong40, unknown_random (+ oracle track for criterion A with level1 anchor).

## Pass/fail vs `config.json`

| Criterion | Config definition | Round 03 result | Pass? |
|-----------|-------------------|-----------------|-------|
| **A** | ≥25% row reduction @ equal snapshot MSE | Oracle + level1 + co-occurrence: **fail** (mean MSE ≫ 0.10 at all budgets). Wrong40 + incidence-refine: **fail** | **Fail** |
| **B** | ≥50% attack MSE cut @ ≤10% utility | Not re-run (Round 02 kill) | **Fail / N/A** |
| **C** | Certificate ≥2× tighter than naive | Not implemented | **Fail / N/A** |

`forbid_accept_before_round: 8` — no acceptance claim.

## Numeric results (honest)

### Assignment barrier @ fraction 0.25, level1 anchor (snapshot MSE mean)

| Assignment | Co-occurrence | Oracle chain (wrong H) | Incidence-refine | LS wrong-H |
|------------|--------------:|-----------------------:|-----------------:|-----------:|
| wrong20 | 0.960 | 0.672 | 0.971 | ~1.0 |
| wrong40 | 1.025 | 0.892 | 1.037 | ~1.0 |
| unknown_random | 1.053 | 1.098 | 1.151 | ~1.0 |

Co-occurrence **does not beat** oracle chain under wrong20/40; only **unknown_random** is within noise (Δ ≈ −0.045). INCIDENCE-REFINE does **not** improve over co-occurrence alone.

### DP-mean anchor @ wrong40, fraction 0.25

| Anchor | Co-occurrence MSE | Refine MSE |
|--------|------------------:|-----------:|
| oracle_true | 1.023 | 1.003 |
| level1_estimate | 1.025 | 1.037 |
| dp_mean (ε=1) | 1.031 | 1.037 |

DP-mean is **within noise** of level1; neither approaches Round-02 oracle-anchor **0.13** MSE at 0.15 fraction.

### Hybrid T1p + sparse GARD (Hungarian MSE mean)

| Assignment | T1p alone | Sparse GARD (wrong H) | Best blend | T1p-mean anchor GARD |
|------------|----------:|----------------------:|-----------:|---------------------:|
| oracle | 0.432 | 0.292 | 0.292 (α→0) | 0.292 |
| wrong20 | 0.419 | 0.708 | 0.396 | 0.708 |
| wrong40 | 0.419 | 0.691 | 0.399 | 0.927 |

Under wrong assignment, **sparse GARD hurts**; optimal blend **favors T1p** (mean α≈0.85 wrong40). Modest hybrid gains on wrong20 (2/5 seeds) do not reach MSE ≤ 0.10.

### Threat reduction (parent mean-MSE gate @ 0.10)

- **wrong40 + incidence-refine + level1:** `reaches_target: false` at all fractions (mean MSE ≈ 1.04 @ 15 rows).
- **Oracle + level1 + co-occurrence:** also **false** — removing oracle mean anchor blocks criterion A even with true H.

### Upper bound reminder

With **true H** but same anchor as experiment, GARD oracle-chain @ wrong40 fraction 0.25 ≈ **0.59** MSE (oracle_true anchor) — still above 0.10; assignment error in **H** was not the only Round-02 enabler (mean anchor mattered).

## Breakthrough status vs config

```json
"assignment_barrier_broken": false
```

| Signal | Met? |
|--------|------|
| Criterion A under wrong assignment | **No** |
| Criterion A oracle with realistic (level1/DP) anchor | **No** |
| INCIDENCE-REFINE reaches MSE ≤ 0.10 @ 25% rows wrong40 | **No** |
| Co-occurrence beats mismatched chain prior wrong40 | **No** (worse by +0.14 MSE) |
| Hybrid beats T1p on wrong40 (majority seeds) | **No** (40% seeds; mean hybrid ≈ T1p) |

Round 02 **conditional** result stands: **75% row reduction** remains valid only for **oracle assignment + oracle mean anchor**.

## Primary lane recommendation (Round 04)

1. **ASSIGN-LOCK / epoch metadata** — Round 02 stub shows permutation breaks LS; pursue hiding before incidence refinement.
2. **Soft assignment (JASPER-style)** from parent TALON Round 06 — hard greedy refine is insufficient.
3. **Keep T1p track** for terminal observations; do not rely on sparse Stage-2 GARD when H is wrong.
4. Park **DP-mean-only** anchor unless tied to a proved DP-SGD release mechanism.

## Artifacts

- `artifacts/round03_metrics.json`
- `logs/experiment_round03.log`
- `code/benchmark_round03.py`

## Answers

1. **Did we break the assignment barrier?** **No** — no config breakthrough criterion met under wrong/unknown assignment or realistic anchors.
2. **Did INCIDENCE-REFINE help?** **No** — MSE ≈ 1.0 vs co-occurrence; overlap metric improves (~0.24) but snapshot MSE does not.
3. **Does co-occurrence fix wrong assignment?** **No** under wrong20/40; marginal under unknown_random only.
4. **Does hybrid T1p+GARD help?** **Marginally** on wrong20; **T1p dominates** under wrong40; not criterion-A grade.
