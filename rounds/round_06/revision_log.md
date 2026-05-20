# Round 06 Revision Log

## Supervisor-Driven Pivot

Round 05 received `REVISE_MAJOR / PIVOT_REQUIRED` because GARD did not solve the unknown-assignment bottleneck. Round 06 therefore changed the main method from graph-regularized fixed-incidence recovery to assignment-first joint soft-incidence recovery.

## Changes Made

1. Defined a new method, **JASPER: Joint Assignment Sinkhorn Projection and Estimation Recovery**.
   - Objective jointly optimizes snapshots and soft incidence.
   - Assignment constraints enforce row batch size and per-epoch sample capacity.
   - Entropic Sinkhorn projection handles unknown/corrupted incidence.
   - Graph smoothing is retained only as a weak secondary regularizer.

2. Implemented a new runnable benchmark in `code/benchmark_round06.py`.
   - True snapshots remain latent.
   - Observed data are noisy batch averages for only selected rows/epochs.
   - The true snapshot mean is not supplied; methods use the observed batch-average mean.
   - Assignment regimes now include exact, partially corrupted, and unknown-within-epoch cases.

3. Added direct comparisons against:
   - `shard_fixed_incidence_ls`;
   - `gard_conditional_prior`;
   - `jasper_joint_soft_incidence`.

4. Added assignment metrics and threat-model metrics.
   - Snapshot MSE.
   - True observed batch residual.
   - Used/assumed observed batch residual.
   - Hard top-\(B\) assignment overlap.
   - Soft true-member mass.
   - Required observed rows/epochs to reach MSE \(\le 0.10\).

5. Added dependency-free SVG plots.
   - The available system `python3` had no NumPy.
   - The existing project virtualenv had NumPy but no Matplotlib.
   - The benchmark therefore writes SVG plots without external plotting dependencies.

## Run Details

Command used:

```bash
"/Users/yetao/Documents/06.My Papers/CQU/2026/01.SHARD/source/experiments/vqc_pennylane/.venv/bin/python" "/Users/yetao/Documents/06.My Papers/CQU/2026/01.SHARD/research-artifacts/novel-inversion-vs-shard/code/benchmark_round06.py"
```

Runtime: 12.13 seconds.

Outputs:

- `artifacts/round06_metrics.json`
- `artifacts/round06_mse_oracle.svg`
- `artifacts/round06_mse_corrupt25.svg`
- `artifacts/round06_mse_corrupt50.svg`
- `artifacts/round06_mse_unknown_epoch.svg`
- `logs/experiment_round06.log`

## Key Results

At 40% random observed rows:

| Assignment | SHARD fixed LS | GARD conditional | JASPER joint soft |
|---|---:|---:|---:|
| oracle | 0.6101 | **0.2970** | 0.3707 |
| corrupt25 | 1.0720 | **0.5419** | 0.6147 |
| corrupt50 | 1.3834 | **0.7432** | 0.8017 |
| unknown_epoch | 1.7061 | **1.0621** | 1.1095 |

At 100% random observed rows:

| Assignment | SHARD fixed LS | GARD conditional | JASPER joint soft |
|---|---:|---:|---:|
| oracle | **0.0027** | 0.0719 | 0.0667 |
| corrupt25 | 1.6805 | **0.2992** | 0.3175 |
| corrupt50 | 2.7689 | **0.5622** | 0.5683 |
| unknown_epoch | 6.2400 | **1.1932** | 1.7446 |

Target MSE \(\le 0.10\):

- Random-row protocol: only oracle assignment reaches target, and only with all 45/45 rows.
- Prefix-epoch protocol: oracle SHARD reaches target at 4/5 epochs; oracle GARD and JASPER require 5/5 epochs.
- No corrupted or unknown assignment regime reaches target for any method.

## Honest Assessment

JASPER is not a successful Stage-2 replacement in this form. It is assignment-aware and can reduce the worst fixed-incidence LS failures under corrupted priors, but it does not recover enough true assignment mass to reach useful snapshot MSE. Under fully unknown within-epoch assignment, it mostly fits arbitrary soft mixtures: the fitted residual can be low while true assignment overlap remains near random.

The Round 06 conclusion is therefore negative but informative: Stage 2 cannot be fixed by a weak entropic assignment prior and graph smoothing alone. A viable replacement needs stronger assignment information, stronger cross-epoch identifiability, or a revised claim that does not require per-sample snapshot identity under fully unknown incidence.
