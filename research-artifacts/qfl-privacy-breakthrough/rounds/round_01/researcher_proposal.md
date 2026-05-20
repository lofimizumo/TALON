# Round 01 — Researcher Proposal: QFL Privacy Breakthrough Survey

## Problem & goal

LASA-QTERM established that **T1** (epoch-terminal-only) individual recovery is impossible, while **T1p/T1b** tiers leak strongly at realistic row budgets. Parent TALON Round 05 showed **GARD** can cut Stage-2 batch-gradient observations by **75%** only under **oracle assignment** and an aligned graph. This run asks: where can we get **measurable** privacy wins—fewer attacker rows, client defenses, or certifiable leak bounds—without relitigating T1 impossibility?

## Hypothesis / claim

1. **GARD-SPARSE:** Replicating parent GARD under sparse Stage-2 observations yields **≥25% row reduction** at matched snapshot MSE (config criterion A).
2. **QFL-SHIELD:** Client-side rank-2 subspace masking plus light noise before gradient publish cuts LASA-QTERM **T1p** attacker MSE by **≥50%** at **≤10%** utility loss (criterion B).
3. **LEAK-CERT:** A tier-conditioned degrees-of-freedom bound is **≥2× tighter** than naive mean-only broadcast (criterion C).

Round 01 tests all three; only (1) is supported empirically.

## Related work

- **SHARD** (vendor `shard_sim`): three-level batched-gradient attack; Stage-2 needs assignment + many rows.
- **LASA-QTERM** (`qfl-terminal-snapshot`): T1 impossibility; T1p ~0.43 MSE vs SHARD ~0.48 on smooth track (Round 04).
- **GARD / JASPER** (parent `rounds/round_05–06`): graph prior helps sparse rows with oracle assignment; unknown assignment is a hard barrier.
- **Prior bridge:** `literature/prior_findings_bridge.md`.

## Method

### Lane 1 — GARD-SPARSE (implemented)

Synthetic Stage-2 proxy (parent Round 05): \(N{=}48\), \(B{=}4\), \(E{=}5\), 60 batch-mean rows, smooth low-rank snapshots. Random row fractions \(\{1.0, 0.6, 0.4, 0.25, 0.15\}\), **oracle assignment**, validation-gated graph Laplacian (chain prior). Baseline: SHARD-style LS with mean anchor.

### Lane 2 — QFL-SHIELD (implemented)

End-to-end `SurrogateQFL` + `ShardAttacker` + production `QtermAttack` (T1p, 7 rows/epoch). Defense: project client snapshots to rank-2 SVD subspace, add \(\sigma{=}0.15\) isotropic noise, then publish gradients. Attack: same LASA-QTERM T1p on defended gradients. Utility: mean batch-gradient MSE vs undefended.

### Lane 3 — LEAK-CERT (implemented, heuristic)

- **Naive:** passive mean-broadcast MSE (T1 honest floor).
- **DoF:** residual variance when \(n_{\text{obs}} \geq N\) (here 70 terminal rows \(\Rightarrow\) dof term 0).
- Compare to empirical T1 / T1p MSE.

### Survey only (Round 02+)

- **ASSIGN-LOCK:** permuted secure aggregation on batch means.
- **PROBE-DETECT:** server active probes.

## Experiment plan

| Lane | Metric | Baseline | Failure mode |
|------|--------|----------|--------------|
| GARD-SPARSE | Snapshot MSE @ target 0.10; min observed rows | SHARD LS | Wrong assignment (not tested R01) |
| QFL-SHIELD | T1p attacker MSE; gradient utility | Undefended T1p | Defense helps attacker (rank prior aligns with graph attack) |
| LEAK-CERT | cert / naive; cert vs empirical T1p | Mean broadcast | Over-observed tier collapses dof term |

**Seeds:** 3, 7, 11, 19, 23. **Code:** `code/benchmark_round01.py`. **Artifacts:** `artifacts/round01_metrics.json`, `logs/experiment_round01.log`.

## Numeric results (Round 01)

### GARD-SPARSE vs SHARD LS (oracle assignment)

| Observed rows | SHARD LS MSE (mean) | GARD-SPARSE MSE (mean) | LS / GARD factor |
|---:|---:|---:|---:|
| 60 | 0.0020 | 0.0020 | 1.0 |
| 36 | 0.220 | **0.0106** | 20.6× |
| 24 | 0.494 | **0.0226** | 21.9× |
| 15 | 0.656 | **0.0499** | 13.1× |
| 9 | 0.790 | **0.130** | 6.1× |

**Threat reduction @ MSE ≤ 0.10:** SHARD LS needs **60/60** rows; GARD-SPARSE needs **9/60** → **85% row reduction** (criterion **A: met**).

### QFL-SHIELD vs LASA-QTERM T1p (undefended)

| Quantity | Mean |
|----------|-----:|
| SHARD Stage-2 oracle MSE | 0.582 |
| T1p attack MSE (undefended) | **0.446** |
| T1p attack MSE (defended r=2, σ=0.15) | 0.555 |
| Attack MSE reduction | **−27%** (worse) |
| Utility loss (grad MSE / ‖S‖²) | >>10% |

Criterion **B: not met** — default shield **increased** leak on 4/5 seeds; utility far above 10%.

### LEAK-CERT

| Quantity | Mean |
|----------|-----:|
| Empirical T1p MSE | 0.427 |
| Cert tight (naive ∪ dof) | 1.000 |
| Tight / naive | 1.0× |
| Cert / T1p (loose upper) | 2.39× |

Criterion **C: not met** — with 70 observed terminal rows, dof term is zero; certificate does not beat naive by 2×.

## Conclusion & primary lane

**Primary lane: GARD-SPARSE** (extend parent Round 05 sparse Stage-2 under oracle assignment, then fold in assignment-robustness in Round 02).

**Deprioritize for R02 unless redesigned:** QFL-SHIELD (needs utility-aware noise calibration, possibly adversarial training against Qterm); LEAK-CERT (needs tier-specific rank bounds, not dof-only).

**Best numeric win vs baseline:** At snapshot MSE ≤ 0.10, **85% fewer observed batch rows** (9 vs 60) vs SHARD-style LS — **breakthrough A**.

## Open risks

- GARD gains may not transfer to **wrong/unknown assignment** (parent Round 05 negative result).
- End-to-end SHARD Stage-2 on smooth track did not converge reliably (warnings in log).
- Shield utility metric may need scale normalization; current gradient MSE is O(10–13).

## Artifacts

- `code/benchmark_round01.py`
- `artifacts/round01_metrics.json`
- `logs/experiment_round01.log`
- `literature/round01_notes.md`
