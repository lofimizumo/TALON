# Round 05 — Researcher Proposal: Stress-Tested Stage-2 Graph Recovery

## Goal

Round 05 tests whether the Round-04 GARD idea survives the two missing pieces called out by the supervisor: non-oracle assignment and non-oracle graph side information. The result is mixed and mostly negative for a full SHARD replacement: graph regularization can substantially reduce the number of observed batch gradients, but only when assignment is essentially correct and the graph is strongly aligned with the true snapshot geometry.

## Method under test

I kept GARD as the main candidate but changed its role from a proposed attack to a validation-gated prior:

\[
\min_S \|H_{\Omega}S-M_{\Omega}\|_F^2
  + \lambda \operatorname{Tr}(S^\top L S)
  + \rho\left\|\frac{1}{N}\mathbf{1}^\top S-\bar{s}\right\|_2^2 .
\]

The benchmark now uses a held-out subset of observed batch rows to select \(\lambda\). Snapshot MSE is never used for hyperparameter selection. If the assumed incidence plus mean anchor is full-rank, the graph prior is gated to \(\lambda=0\), because Round 04 already showed that exact LS is better when the linear system is identifiable.

## Round-05 benchmark

Implemented in `code/benchmark_round05.py`.

### Simulator

- \(N=48\), batch size \(B=4\), \(E=5\), total batch-gradient rows \(60\), snapshot dimension \(32\).
- True snapshots are smooth low-rank graph signals with additive noise, matching Round 04 for comparability.
- Observed batch means are always generated from the true hidden incidence.
- The attacker solves with one of four assignment regimes:
  - `oracle`: exact true batch membership;
  - `wrong20`: approximately one member per batch row replaced;
  - `wrong40`: approximately two members per batch row replaced;
  - `unknown_random`: batch membership guessed at random.
- Observation protocols:
  - random fractions: 1.00, 0.80, 0.60, 0.40, 0.25, 0.15;
  - prefix epochs: 5, 4, 3, 2, 1 observed epochs.
- Seeds: 3, 7, 11, 19, 23, 29, 31, 37.

### Baselines and graph regimes

- `shard_style_ls`: partial-observation SHARD fixed-assignment LS with mean anchor.
- `ridge_ls`: ridge LS, ridge weight selected on validation rows.
- `low_rank_pca`: ridge LS followed by validation-selected low-rank PCA projection.
- `gard_oracle_graph`: chain graph aligned with the synthetic snapshot order.
- `gard_noisy_graph`: correct chain plus random noisy edges.
- `gard_wrong_graph`: permuted chain graph.
- `gard_cooccurrence_graph`: graph inferred from observed attacker-assumed batch co-occurrence.
- `gard_lsknn_graph`: graph inferred from SHARD-style LS recovered snapshots.
- `wrong_graph_forced_lam0.3`: explicit wrong-graph MAP stress test without validation gating.

## Numeric results

### Oracle assignment, random observed rows

Mean snapshot MSE over 8 seeds:

| Observed rows | SHARD LS | Ridge | Low-rank | GARD oracle | GARD noisy | GARD wrong | GARD cooccurrence | GARD LS-kNN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 60/60 | 0.0021 | 0.0036 | 0.0097 | 0.0021 | 0.0021 | 0.0021 | 0.0021 | 0.0021 |
| 48/60 | 0.0544 | 0.0544 | 0.0586 | 0.0544 | 0.0544 | 0.0544 | 0.0544 | 0.0544 |
| 36/60 | 0.2288 | 0.2472 | 0.2498 | **0.0107** | 0.0651 | 0.2288 | 0.3023 | 0.1429 |
| 24/60 | 0.4825 | 0.4985 | 0.4998 | **0.0236** | 0.1765 | 0.4825 | 0.5818 | 0.4514 |
| 15/60 | 0.6763 | 0.7011 | 0.7152 | **0.0539** | 0.3628 | 0.7045 | 0.7248 | 0.6850 |
| 9/60 | 0.8018 | 0.8700 | 0.8717 | **0.1502** | 0.5886 | 0.9168 | 0.8088 | 0.8000 |

Interpretation: with correct assignment and the oracle graph, GARD is still a strong nullspace filler. At 40% observed rows it improves over SHARD-style LS by about 20.5x. The noisy graph helps only moderately. Wrong, co-occurrence, and LS-kNN graphs mostly fail to give a reliable advantage.

### Assignment stress at 40% observed rows

Mean snapshot MSE:

| Assignment regime | SHARD LS | Ridge | Low-rank | GARD oracle graph | GARD noisy graph | GARD wrong graph |
|---|---:|---:|---:|---:|---:|---:|
| oracle | 0.4825 | 0.4985 | 0.4998 | **0.0236** | 0.1765 | 0.4825 |
| wrong20 | 0.9423 | 0.7803 | 0.7811 | **0.3625** | 0.5800 | 0.9791 |
| wrong40 | 1.6200 | 0.9385 | 0.9474 | **0.8294** | 0.8613 | 0.9817 |
| unknown_random | 2.1510 | 1.0300 | 1.0740 | 1.0060 | **0.9974** | 1.0210 |

This is the most important negative result. Even an oracle graph cannot rescue substantially wrong assignment. With fully unknown/random assignment, all methods are near MSE 1.0 or worse, which is essentially unusable in this normalized simulator.

### Threat-model reduction

Using fixed target snapshot MSE \(\le 0.10\):

- Random-row observation with oracle assignment:
  - SHARD LS, ridge LS, and low-rank PCA require 48/60 observed rows.
  - GARD with oracle graph requires 15/60 observed rows, a 75% reduction.
  - GARD with noisy graph requires 36/60 observed rows, a 40% reduction.
  - Wrong graph and LS-kNN graph require 48/60 rows, no meaningful gain over SHARD-style LS.
- Prefix-epoch observation with oracle assignment:
  - SHARD LS and ridge LS require 4/5 epochs.
  - GARD with oracle graph requires 1/5 epoch.
  - GARD with noisy graph requires 3/5 epochs.
- Under `wrong20`, `wrong40`, and `unknown_random`, no tested method reaches MSE \(\le 0.10\) at any random-row budget.

## Conclusion

The Round-04 optimism should be narrowed. GARD is not a standalone solution to SHARD Stage 2 under unknown assignments. It is a conditional regularizer that can reduce the batch-gradient observation burden if:

1. the attacker has nearly correct batch membership,
2. the incidence design is rank-deficient due to missing rows,
3. a graph aligned with true snapshot geometry is available, and
4. regularization is selected on held-out observed rows.

The graph prior collapses under wrong graphs unless validation gates it off, and validation does not reliably protect against assignment error because the held-out rows share the same wrong incidence model. Co-occurrence-inferred and LS-kNN-inferred graphs are not strong enough in this simulator.

## Proposed next direction

The stronger direction is no longer "GARD alone." The next method should target assignment directly:

- joint assignment-and-snapshot MAP with graph smoothness only as a secondary prior;
- soft batch-incidence inference from multiple epochs with consistency constraints;
- validation based on cross-epoch cycle consistency or held-out known audit batches, not only observed residual under assumed incidence;
- graph learning from public metadata or recovered snapshots only after proving that the graph source is available without reintroducing SHARD's full-gradient requirement.

Until that exists, the honest claim is: **GARD can reduce Stage-2 observation requirements by 75% in the favorable correct-assignment/oracle-graph setting, but it does not yet solve SHARD's unknown-assignment bottleneck.**

## Artifacts

- `code/benchmark_round05.py`
- `artifacts/round05_metrics.json`
- `artifacts/round05_random_oracle_assignment.png`
- `artifacts/round05_assignment_stress.png`
- `logs/experiment_round05.log`
