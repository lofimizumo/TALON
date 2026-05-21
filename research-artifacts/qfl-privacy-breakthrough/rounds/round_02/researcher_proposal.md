# Round 02 — Researcher Proposal: Honest Threat Reduction & Assignment Stress

## Problem & goal

Round 01 overclaimed **85% row reduction** via a per-seed minimum-row statistic under oracle assignment, oracle graph, and oracle mean anchor. Round 02 implements supervisor demands: parent-aligned metrics, assignment/graph stress, baselines, mean-anchor ablation, ASSIGN-LOCK stub, and a utility-constrained SHIELD grid—then re-evaluates config criteria A/B/C honestly.

## Pre-registered criterion A gate

**Gate:** `parent_mean_mse_at_minimum_rows` — among row budgets, take the **minimum observed rows** such that **mean snapshot MSE across seeds** ≤ target (same semantics as parent TALON `benchmark_round05.py`). Round-01’s “any single seed passes” statistic is reported only as a diagnostic.

## Method (implemented)

### Stage-2 stress (`code/benchmark_round02.py`)

- **Assignment** (observations from true H, solve with assumed H): `oracle`, `wrong20`, `wrong40`, `unknown_random`.
- **Graph priors:** `gard_oracle_graph`, `gard_noisy_graph`, `gard_wrong_graph`, `gard_cooccurrence_graph`, `gard_lsknn_graph`.
- **Baselines:** `shard_style_ls`, `ridge_ls`, `low_rank_pca`.
- **Mean anchor ablation** @ fractions 0.15/0.25, oracle assignment: `no_anchor`, `level1_estimate`, `noisy_mean`, `oracle_true`.
- **MSE targets:** 0.05, 0.10, 0.15 in `threat_reduction_by_assignment`.
- **Seeds:** 3, 7, 11, 19, 23. **Fractions:** 1.0, 0.6, 0.4, 0.25, 0.15.

### ASSIGN-LOCK (stub)

Server permutes published batch-row order; attacker solves with identity slot-to-incidence map. Compared to oracle-assignment GARD upper bound at 15 rows.

### QFL-SHIELD (grid, deprioritized if kill criteria met)

Rank ∈ {1,2,4,8}, σ ∈ {0, 0.05, 0.10, 0.15, 0.25}. Utility: batch-gradient MSE / undefended gradient energy; constraint ≤10%. Attacks: LASA-QTERM T1p and SHARD Stage-2 on same track.

### LEAK-CERT

**Parked** Round 03+ — tier-specific rank/sensitivity bound at T1p **7 rows/epoch** not implemented.

## Pass/fail vs `config.json`

| Criterion | Config definition | Round 02 result | Pass? |
|-----------|-------------------|-----------------|-------|
| **A** | ≥25% row reduction vs full observation @ equal snapshot MSE | Oracle + parent gate @ 0.10: **75%** (15/60 rows). **wrong20/40/unknown:** no budget reaches mean MSE ≤ 0.10 | **Conditional only** — not under wrong assignment |
| **B** | ≥50% attack MSE cut @ ≤10% utility | 0/100 grid cells feasible on normalized utility; best feasible N/A | **Fail** |
| **C** | Certificate ≥2× tighter than naive | Parked | **Fail / N/A** |

Policy: `forbid_accept_before_round: 8` — acceptance not sought this round.

## Numeric results (parent-aligned, GARD oracle graph)

### Threat reduction — oracle assignment

| Target MSE | Min rows | Reduction | Mean MSE @ budget |
|---:|---:|---:|---:|
| 0.05 | 15 | 75% | 0.049 |
| 0.10 | 15 | **75%** | 0.049 |
| 0.15 | 9 | 85%* | 0.130 |

\*85% only at the looser 0.15 target; **withdrawn as headline** for 0.10.

### Threat reduction — wrong assignment (@ MSE 0.10, parent gate)

| Regime | Reaches target? | Best reduction |
|--------|-----------------|----------------|
| wrong20 | No | — |
| wrong40 | No | — |
| unknown_random | No | — |

**Best honest row reduction @ MSE 0.10 (parent gate):** **75%**, assignment **oracle**, method **gard_oracle_graph**, 15/60 rows. No non-oracle regime qualifies.

### Mean-anchor ablation (GARD oracle graph, fraction 0.15, oracle assignment)

| Anchor | Mean MSE |
|--------|--------:|
| oracle_true | **0.130** |
| noisy_mean | 0.145 |
| no_anchor | 0.194 |
| level1_estimate | 0.362 |

Oracle mean remains strongest; level1-only anchor is inadequate at this sparsity.

### ASSIGN-LOCK stub (mean over seeds)

| Metric | ~Value |
|--------|-------:|
| Lock-broken LS MSE | 1.26 |
| Oracle-assignment GARD MSE | 0.67 |
| Slot identity overlap | 1.0 (permutation breaks semantics, not membership overlap metric) |

Permutation raises error to near-random-LS scale; does not approach oracle GARD bound without unpermute.

### QFL-SHIELD grid

- **Feasible cells (utility ≤10%):** 0 / 100.
- **Kill criteria:** no feasible cell improves T1p attack MSE → **deprioritized** next rounds.

## Claim change vs Round 01

| Claim | Round 01 | Round 02 |
|-------|----------|----------|
| Row reduction @ MSE 0.10 | 85% (9 rows) | **75% (15 rows)** — parent-aligned |
| Criterion A under wrong assignment | not tested | **fails** |
| Breakthrough | labeled “met” | **conditional replication only** |

## Primary lane recommendation

**GARD-SPARSE + ASSIGN-LOCK** — continue Stage-2 graph work only as an **upper bound**; invest in assignment hiding and graph inference under `wrong20`/`unknown_random`. **Drop SHIELD** from primary plan until utility-feasible configs exist.

## Artifacts

- `artifacts/round02_metrics.json`
- `logs/experiment_round02.log`
- Code: `code/benchmark_round02.py`

## Answers (supervisor sign-off preview)

1. **Does criterion A hold under wrong assignment?** **No** — at MSE ≤ 0.10 with the pre-registered parent mean gate, GARD oracle graph never reaches the target under `wrong20`, `wrong40`, or `unknown_random`.
2. **Best honest row reduction %?** **75%** at MSE 0.10 (oracle assignment, 15/60 rows). The former **85%** claim is **withdrawn** for MSE 0.10.
