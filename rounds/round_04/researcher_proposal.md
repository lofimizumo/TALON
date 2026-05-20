# Round 04 — Researcher Proposal: Stage-2 Recovery With Missing Batch Gradients

## Goal correction

Round 04 pivots away from Stage-3 polish. The updated goal is to find a method that is significantly better than SHARD under the same broad honest-but-curious gradient-leakage setting, with emphasis on Stage 2 and on weakening SHARD's requirement that the adversary observe every intermediate local-round batch gradient.

## Diagnosis: SHARD's fundamental bottlenecks

SHARD's Stage 2 is the central fragility point, not Stage 3.

1. **Complete-observation dependence.** SHARD first left-inverts every observed batch gradient into a batch mean snapshot, then uses all epoch/batch rows to build an incidence design. The paper's uniqueness theorem assumes the realized batch-incidence matrix has full column rank and that each per-epoch gradient map is left-invertible. With missing batch gradients, the incidence rank can collapse below \(N\), leaving an unobserved nullspace even if the batch assignments were known.

2. **Over-strong threat model.** The adversary needs access to local intermediate gradients for every mini-batch across local epochs. Many FL deployments expose only round deltas, sparse telemetry, compressed gradients, partial logs, or sampled updates. This makes SHARD's current observation model stronger than its honest-but-curious framing suggests.

3. **Combinatorial assignment plus linear identifiability are entangled.** SHARD's alternating minimization has two hard problems: identifying hidden batch assignments and solving for per-sample snapshots from batch averages. The current guarantee only works after assuming enough complete rows to make the incidence design identifiable and unique.

4. **Minimum-norm least squares is a poor missing-row estimator.** When rows are missing, SHARD's fixed-assignment update can fit observed batch means nearly perfectly while placing the missing nullspace at an arbitrary minimum-norm solution. This can have low observed residual but high per-sample snapshot error.

## Candidate full-method directions

### Candidate A: GARD — Graph-Augmented Recovery for Disaggregation

Add a Gaussian Markov graph prior to Stage 2:

\[
\min_S \|H_{\Omega}S - M_{\Omega}\|_F^2
  + \lambda \mathrm{Tr}(S^\top L S)
  + \rho \left\|\frac{1}{N}\mathbf{1}^\top S-\bar{s}\right\|_2^2.
\]

Here \(H_{\Omega}\) are the observed batch-incidence rows, \(M_{\Omega}\) are recovered/observed batch means, \(L\) is a public or side-information graph over client samples, and \(\bar{s}\) is the Stage-1 mean anchor. In a full attack, the graph could come from temporal acquisition order, user/session metadata, public class/order metadata, or an inferred co-occurrence graph from a small fully observed prefix. The method is a posterior mean/MAP estimator under Gaussian observation noise and graph-smooth snapshots.

**Why it addresses the bottleneck:** it fills the missing-row nullspace using an explicit structural prior instead of requiring \(H_{\Omega}\) to have full column rank. It should be gated off when incidence is full rank and noise is low, because SHARD's exact LS is then better.

### Candidate B: BLRD — Bayesian Low-Rank Disaggregation

Model snapshots as \(S=\bar{s}+\tilde{U}\tilde{V}^{\top}\), with a low-rank posterior over latent factors. This targets class/manifold structure and can be optimized directly from observed batch means. It is attractive when snapshots are globally low-dimensional but does not by itself decide which nullspace directions are plausible unless the factor posterior is strongly constrained.

**Round-04 pilot outcome:** a quick implementation of factor MAP initialized from SHARD-style LS did not outperform LS in this small masked-incidence setting. I therefore did not keep it as the selected method, but it remains a candidate if combined with graph priors or amortized permutation-invariant set recovery.

### Candidate C: Active/partial observation recovery

If the honest-but-curious participant can influence which local gradients are logged or requested, use incidence-rank-aware active sampling: choose the next batch/epoch observations to maximize the smallest nonzero singular value of the observed incidence design, or maximize posterior variance reduction under a graph prior. This directly reduces the number of gradients needed, but it changes the adversary capability more than GARD and needs a separate threat-model justification.

## Selected direction

I selected **GARD** for the minimal benchmark because it directly attacks the missing-gradient assumption while staying close to SHARD's Stage-2 linear algebra. It can be interpreted as SHARD's fixed-assignment update plus a Bayesian graph prior and mean anchor. This is not yet a complete replacement for SHARD's unknown-assignment loop; it is a focused identifiability experiment.

## Minimal benchmark

Implemented in `code/benchmark_round04.py`.

### Simulator

- \(N=48\), batch size \(B=4\), \(K=12\), \(E=5\), so the full run has 60 batch-gradient rows.
- Snapshot dimension \(\dim(\mathfrak{g})=32\).
- True snapshots are smooth low-rank graph signals with small noise.
- Observations are \(M_{\Omega}=H_{\Omega}S+\epsilon\), with noise std 0.01.
- Observation fractions: 1.0, 0.6, 0.4, 0.25, 0.15.
- Seeds: 3, 7, 11, 19, 23.

### Baseline

`shard_style_ls`: the fixed-assignment SHARD update, solving least squares with the Stage-1 mean anchor. This is favorable to SHARD because it gives the baseline the true incidence rows, avoiding the assignment problem.

### Method

`gard_graph_map`: solves the graph-regularized posterior/MAP linear system with \(\lambda=0.35\) and the same mean anchor.

## Numeric results

Snapshot MSE means over 5 seeds:

| Observed fraction | Rows | Rank with mean | SHARD-style LS | GARD | GARD gain |
|---:|---:|---:|---:|---:|---:|
| 1.00 | 60/60 | 48 | 0.00197 | 0.01973 | 0.10x |
| 0.60 | 36/60 | 37 | 0.25272 | 0.03423 | 7.42x |
| 0.40 | 24/60 | 25 | 0.48769 | 0.07110 | 7.47x |
| 0.25 | 15/60 | 16 | 0.67938 | 0.13252 | 5.43x |
| 0.15 | 9/60 | 10 | 0.79260 | 0.28797 | 2.93x |

Interpretation:

- With complete full-rank observations, SHARD-style LS is best. This is expected; the graph prior biases an already identifiable system.
- Once observations are missing and the incidence-plus-mean rank is below \(N\), GARD substantially reduces snapshot MSE.
- SHARD-style LS has near-zero observed-batch residual in missing regimes, but high snapshot MSE. That demonstrates the nullspace/identifiability failure: fitting observed rows is not enough.

## Threat-model implication

GARD is a plausible route to reducing the "all local gradients" requirement, but only when a defensible sample graph or smoothness prior exists. It should be framed as a **conditional Stage-2 replacement**:

- use vanilla SHARD/LS when batch observations are complete and incidence is full rank;
- use GARD when observations are missing, rank deficient, and a public or inferable graph prior is available.

The next round should remove the oracle-incidence simplification by combining GARD with SHARD's assignment loop or with a soft permutation-invariant assignment layer.

## Artifacts

- `code/benchmark_round04.py`
- `artifacts/round04_metrics.json`
- `artifacts/round04_stage2_missing_observations.png`
- `logs/experiment_round04.log`
