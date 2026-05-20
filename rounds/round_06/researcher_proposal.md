# Round 06 — Researcher Proposal: JASPER for Assignment-First Stage 2

## Goal

Round 05 showed that GARD is only a conditional nullspace prior: it helps when batch incidence is already correct, but it does not solve the missing/corrupted assignment bottleneck. Round 06 therefore pivots to an assignment-first Stage-2 method and tests whether joint soft incidence recovery can replace SHARD's fixed-incidence Stage 2 when exact batch composition and full intermediate rows are missing.

The result is mostly negative. The new method improves over naive fixed-incidence LS in several corrupted-assignment regimes, but it does not beat the conditional GARD prior consistently and it fails under fully unknown within-epoch assignment. It is not yet a viable Stage-2 replacement.

## Method: JASPER

I define **JASPER: Joint Assignment Sinkhorn Projection and Estimation Recovery**.

JASPER alternates between soft assignment and snapshot recovery:

\[
\min_{S,A}
  \|(A/B)S - M\|_F^2
  + \lambda_g \operatorname{Tr}(S^\top L S)
  + \lambda_r \|S\|_F^2
  + \text{entropic/KL assignment terms}
\]

subject to:

- each observed batch row has soft membership mass \(B\);
- within each observed epoch, each sample has capacity at most one;
- optional incidence priors can bias but not hard-fix the assignment;
- graph regularization is weak and secondary, after assignment uncertainty is represented.

For fixed \(A\), JASPER solves a ridge/graph MAP system for snapshots \(S\). For fixed \(S\), it updates each epoch's soft incidence using an entropic Sinkhorn-style projection with row batch-size constraints and per-sample epoch capacities. The mean anchor is estimated from observed batch averages; the method is not given the true snapshot mean.

## Benchmark

Implemented in `code/benchmark_round06.py`.

Synthetic Stage-2 setup:

- \(N=36\) samples, batch size \(B=4\), \(E=5\) epochs, 45 full batch rows, snapshot dimension 24.
- True snapshots are latent smooth low-rank vectors.
- Observations are only noisy batch-average snapshots for selected rows.
- Observation protocols:
  - random row fractions: 1.00, 0.80, 0.60, 0.40, 0.25;
  - prefix epochs: 5, 4, 3, 2, 1.
- Assignment regimes:
  - `oracle`: true observed incidence;
  - `corrupt25`: one member per batch row replaced;
  - `corrupt50`: two members per batch row replaced;
  - `unknown_epoch`: no sample-level assignment, only observed epoch/row counts.
- Seeds: 3, 7, 11, 19, 23, 29.

Compared methods:

- `shard_fixed_incidence_ls`: SHARD-style fixed-incidence LS using the available assumed incidence.
- `gard_conditional_prior`: fixed-incidence ridge + chain graph prior, retained as an ablation/prior.
- `jasper_joint_soft_incidence`: the new joint soft-incidence method.

## Numeric Results

Mean snapshot MSE at 40% random observed rows:

| Assignment regime | SHARD fixed LS | GARD conditional | JASPER joint soft | JASPER hard overlap |
|---|---:|---:|---:|---:|
| oracle | 0.6101 | **0.2970** | 0.3707 | 1.000 |
| corrupt25 | 1.0720 | **0.5419** | 0.6147 | 0.718 |
| corrupt50 | 1.3834 | **0.7432** | 0.8017 | 0.438 |
| unknown_epoch | 1.7061 | **1.0621** | 1.1095 | 0.132 |

Mean snapshot MSE at 100% random rows:

| Assignment regime | SHARD fixed LS | GARD conditional | JASPER joint soft | JASPER hard overlap |
|---|---:|---:|---:|---:|
| oracle | **0.0027** | 0.0719 | 0.0667 | 1.000 |
| corrupt25 | 1.6805 | **0.2992** | 0.3175 | 0.759 |
| corrupt50 | 2.7689 | **0.5622** | 0.5683 | 0.492 |
| unknown_epoch | 6.2400 | **1.1932** | 1.7446 | 0.094 |

JASPER does reduce catastrophic fixed-incidence LS failure under corrupted assignment. For example, at full observation under `corrupt25`, fixed LS has MSE 1.6805 while JASPER has 0.3175. But this is not enough: the conditional GARD prior is still slightly better in that regime, and all non-oracle regimes remain far above the target MSE of 0.10.

## Threat-Model Reduction

Target: mean snapshot MSE \(\le 0.10\).

| Protocol | Assignment | SHARD rows/epochs | GARD rows/epochs | JASPER rows/epochs |
|---|---|---:|---:|---:|
| random rows | oracle | 45/45 | 45/45 | 45/45 |
| random rows | corrupt25 | not reached | not reached | not reached |
| random rows | corrupt50 | not reached | not reached | not reached |
| random rows | unknown_epoch | not reached | not reached | not reached |
| prefix epochs | oracle | 4/5 epochs | 5/5 epochs | 5/5 epochs |
| prefix epochs | corrupt25 | not reached | not reached | not reached |
| prefix epochs | corrupt50 | not reached | not reached | not reached |
| prefix epochs | unknown_epoch | not reached | not reached | not reached |

Unlike Round 05's oracle-graph GARD result, Round 06 shows no meaningful reduction in required observed rows once oracle graph/mean advantages are removed and assignment uncertainty is modeled.

## Interpretation

JASPER is the right kind of pivot, but this implementation fails the success threshold. The core issue is identifiability: when labels are unknown within an epoch, many soft assignments can fit observed batch averages while having almost random true membership overlap. At 40% rows under `unknown_epoch`, JASPER's hard overlap is only 0.132 and snapshot MSE is 1.1095.

The Sinkhorn assignment update also uses a weak linear proxy cost based on distance between individual snapshot estimates and batch means. That proxy is not the true combinatorial objective of matching a batch average, so it can reduce fitted residual without recovering the true batch composition.

## Viability Verdict

**Not yet viable as a Stage-2 replacement.**

Round 06 satisfies the requested pivot by implementing and testing an assignment-first joint soft method, but the empirical result is negative. JASPER is useful as a diagnostic prototype and possibly as a module when moderate assignment priors exist, but it does not remove SHARD's dependence on accurate batch membership or sufficient observed rows.

The next viable direction should either add genuinely informative side information for assignment, such as partial audit rows or public per-example metadata, or change the threat model to recover snapshots only up to permutation/cluster equivalence rather than per-sample identity.

## Artifacts

- `code/benchmark_round06.py`
- `artifacts/round06_metrics.json`
- `artifacts/round06_mse_oracle.svg`
- `artifacts/round06_mse_corrupt25.svg`
- `artifacts/round06_mse_corrupt50.svg`
- `artifacts/round06_mse_unknown_epoch.svg`
- `logs/experiment_round06.log`
