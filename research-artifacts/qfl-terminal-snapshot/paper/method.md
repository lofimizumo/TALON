# LASA-QTERM: Terminal Snapshot Recovery for Quantum Federated Learning

**Method name:** **LASA-QTERM** (LASA linear **Q**uantum **TERM**inal snapshot recovery).  
**Alias:** **Q-SNAP-T** (quantum snapshot from terminals).

**Production entry point:** `code/qterm_attack.py` (`QtermAttack`).

## Problem

In quantum federated learning with LASA-encoded snapshots, SHARD Stage 2 recovers individual snapshots \(s_i \in \mathbb{R}^d\) from **intermediate** minibatch gradients \(g^{(e,k)} = A^{(e)} \bar{s}^{(e,k)}\). Deployments that publish only **terminal** observations—epoch summaries or weaker partial trajectories—need an honest attack stack that does not smuggle within-epoch SGD steps or mean-imputed fake rows.

## Observation model

| Symbol | Meaning |
|--------|---------|
| \(S \in \mathbb{R}^{N \times d}\) | Per-sample snapshot matrix |
| \(A^{(e)} \in \mathbb{R}^{D \times d}\) | SurrogateQFL coefficient draw per round |
| \(c^{(e)}\) | Epoch terminal (mean of batch gradients that round) |
| \(g^{(e,k)}\) | Minibatch gradient row \(k\) within epoch \(e\) |

Linear LASA structure: \(g = A \bar{s}\) for batch mean \(\bar{s}\).

## LASA-QTERM tiers (honest methods only)

LASA-QTERM is **not** one estimator—it is a **tier-conditioned** package:

| Tier | `QtermTier` | Production method | Rows (default sim) |
|------|-------------|-------------------|-------------------|
| **T1** | `T1` | `graph_term_map` + mean anchor | 10 epoch terminals |
| **T1p** | `T1p` | `partial_honest_disaggregate` | \(10p\) (default \(p{=}7\) → 70) |
| **T1b** | `T1b` | `b1_budget_disaggregate` | 80 (budget) or 320 (full) |

**Excluded from production:** `pad_partial_for_shard` (synthetic early rows), and any headline that compares T1b MSE to SHARD without row-budget labels.

### T1 — graph terminal MAP

1. Recover \(\bar{s}\) from stacked \(c^{(e)} = A^{(e)}\bar{s}\) (Level-1, same as SHARD).
2. Apply chain-graph Fiedler spread around \(\bar{s}\) with ridge \(\lambda\) (`graph_lambda`); spread scales as \(\lambda/(1+\lambda)\).

Round 03 best on smooth: \(\lambda=10\), `spread_scale=0.5`. **Does not identify** \(s_i - \bar{s}\) from data alone (see `paper/impossibility_t1.md`).

### T1p — honest partial terminal

- Observes only the **last** \(p\) of \(K\) minibatch rows per epoch.
- Alternating assignment on observed batch means + graph anchor; **no** imputation of missing early rows.
- Default \(p=7\) of \(K=8\) on \(N=32, B=4\).

### T1b — per-client B=1 terminals

- One gradient \(g^{(e,i)} = A^{(e)} s_i\) per sample per epoch (no within-epoch steps).
- Budget mode: first 80 rows in epoch-major order for fair comparison to 80-row SHARD.

## Baselines (benchmark only, not production)

- **Passive:** broadcast \(\bar{s}\) to all rows.
- **T2 SHARD:** `ShardAttacker.level2_disaggregate` on all intermediate rows (oracle).

## Metrics

- **Primary:** Hungarian-aligned snapshot MSE (permutation-invariant).
- **Diagnostics:** fixed-order MSE, `snapshot_row_std`, row/intermediate counts.
- **Acceptance:** tier-specific; T1 success path is **impossibility + honest negatives**, not ≤2× SHARD at 10 rows.

## Reproducibility

```bash
cd research-artifacts/qfl-terminal-snapshot
python code/benchmark_round04.py
```

Artifacts: `artifacts/round04_metrics.json`, `logs/experiment_round04.log`.

## Relation to parent TALON

Parent work (`/workspace`, PHASE2_SCOPED_ACCEPT) showed terminal probes recover **class aggregates**, not individuals. LASA-QTERM targets **per-sample LASA snapshots** with SHARD as oracle; see `literature/parent_talon_bridge.md` and `tutorial/tutorial.md`.

## References

- Formal T1 impossibility: `paper/impossibility_t1.md`
- Tier scope table: `paper/scope.md`
- Vendored simulator: `vendor/shard_sim` via `code/_paths.py`
