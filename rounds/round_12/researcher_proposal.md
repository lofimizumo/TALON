# Round 12 — Phase 2 revision (TANGO-MB theory, counts, differentiated methods)

## Problem & goal

Round 11 fixed the **deterministic minibatch divisor bug** (TANGO-MB) and won primary prototype MSE vs passive, but supervisor **REVISE_MAJOR** blocked ACCEPT: flawed Remark MB-A algebra, no drift bound, count MAE regression, STORM duplicate, unquantified active vs scaling, narrow scenarios.

**Goal this round:** Close theory gaps, restore dual-metric honesty (count + prototype), differentiate methods beyond rescaling, broaden evaluation — while keeping **minibatch_sgd** as Phase-2 primary with prototype win vs passive.

## Hypothesis / claims

| ID | Claim | Status |
|---|---|---|
| H1 | Correct Lemma MB-A proof resolves algebra criticism | **Supported** (`paper/proofs.md`) |
| H2 | Empirical within-step drift explains \(\sim 0.01\) MSE floor | **Supported** (mean drift **0.112**, Lemma MB-B) |
| H3 | Principled MB count estimator can beat passive on primary | **Supported** (0.100 vs 0.295 MAE) |
| H4 | Active probing dominates scaling-only passive+MB | **Supported** (~26× prototype gain) |
| H5 | Drift2 / stack-ridge differ numerically from TANGO-MB | **Supported** (drift2 **0.025**; stack on passive) |

## Method (Round 12)

### Lemma MB-A (corrected)

Per-minibatch sum at \(W_0=0\): \(\sum_b \partial \mathcal{L}_b / \partial b_c = (N/B)(N p_c - n_c)\). Terminal scale \(T_{\mathrm{eff}} = T(N/B)\). Implementation: `effective_gradient_steps`, `scale_terminal_deltas`.

### Lemma MB-B (drift)

Bound \(\|W_M-W_0\|_F \le \eta (N/B) G_{\max}\). Code: `within_step_weight_drift`. **TANGO-MB-drift2** inflates \(T_{\mathrm{eff}}\) using measured drift before second inversion.

### Lemma MB-Iter (\(W_0 \neq 0\))

One Jacobian step on bias gradients at estimated \(\hat W\) with blend \(\alpha=0.15\). Enabled when `init_weight_scale > 0`.

### Count estimator (`estimate_counts_mb`)

Lemma B with \(T_{\mathrm{eff}}\)-scaled bias gradients, using the probe round **closest to uniform** (minimizes bias-moment error from aggressive active probes). Prototypes still from full active stack + scaled `tango_estimate_sums`.

### Methods vs Round 11

| Method | Role |
|---|---|
| `tango_mb` | Primary Phase-2 estimator |
| `tango_mb_iter` | Nonzero-init Jacobian correction |
| `tango_mb_drift2` | Drift-adjusted second pass (replaces STORM) |
| `stack_mb_ridge` | Ridge stacked moments on passive probes |
| `passive_mb` | Scaling-only ablation |
| `tango_vanilla` | Failure control |

**STORM removed** (was identical to TANGO-MB at ridge=0).

## Experiment plan

**Script:** `code/benchmark_round12.py`  
**Seeds:** 8 per scenario (unchanged)  
**Aggregates:** mean, std, **median**, **IQR**

### Scenarios

| Scenario | Purpose |
|---|---|
| `minibatch_sgd` | **Phase-2 primary** |
| `minibatch_nonzero_init` | \(W_0 \neq 0\) + iter |
| `minibatch_label_noise` | 8% label flips |
| `minibatch_terminal_noise` | \(\sigma=0.002\) on deltas |
| `frozen_mlp_minibatch` | Nonlinear \(\phi(x)\), head-only |
| `balanced_clean` | Full-batch regression |
| `imbalanced_clean` | Class skew |

## Results (computed — `artifacts/round12_metrics.json`)

### Primary `minibatch_sgd`

| Method | Proto MSE mean | Proto MSE median | Count MAE mean |
|---|---:|---:|---:|
| passive_multi_round | 0.753 | 0.751 | 0.295 |
| passive_mb | 0.284 | 0.282 | — |
| tango_vanilla | 6.588 | — | — |
| **tango_mb** | **0.011** | **0.011** | **0.100** |
| tango_mb_drift2 | 0.025 | — | — |
| tango_mb_iter | 0.011 | — | (same as MB at \(W_0=0\)) |

- **phase2_primary_win:** true (proto)
- **count_mae_beats_passive:** true
- **active_probe_gain_over_scaling_only_x:** ~25.7×
- **tango_mb_vs_passive_gain_x:** ~68.3×

### `minibatch_nonzero_init`

| Method | Proto MSE mean |
|---|---:|
| tango_vanilla | 10.078 |
| tango_mb | 0.017 |
| **tango_mb_iter** | **0.013** |
| passive_multi_round | 4.493 |

### Lemma MB-B empirical

- `within_step_weight_drift_mean`: **0.1115**
- `effective_gradient_steps`: **18.0**

## Changes this round (from Round 11)

See `revision_log.md`.

## Limitations

1. Still **synthetic** simulator; not CNN FedAvg.
2. **SHARD** baseline not re-run (project gate open).
3. Drift2 **worse** than MB on primary mean MSE — secondary method only.
4. Count estimator relies on **near-uniform probe round** among active biases — principled but not full joint ML.
5. Global Phase-2 ACCEPT remains for supervisor.

## Artifacts

- `artifacts/round12_metrics.json`
- `artifacts/round12_minibatch_methods.svg`
- `logs/experiment_round12.log`

## Reproduce

```bash
python3 code/benchmark_round12.py
```
