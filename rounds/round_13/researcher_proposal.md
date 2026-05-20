# Round 13 — New minibatch directions (joint, coupled, honest ablation)

## Problem & goal

Round 12 **REVISE_MAJOR** persists: \(T_{\mathrm{eff}}\) scaling is bookkeeping at \(W_0 \approx 0\), not a general minibatch-FL fix; `passive_mb` was coupled to `estimate_counts_mb`; `paper/proofs.md` had drift **0.42** vs measured **0.112**; drift2 falsely claimed to beat TANGO-MB.

**Goal:** Discover and **test** at least two non-trivial directions beyond rescaling, keep **`minibatch_sgd`** primary, beat passive on prototype MSE, report **median + IQR**, fix proof errors, restore honest scaling ablation.

## Hypotheses

| ID | Claim | Status |
|---|---|---|
| H1 | Joint moment inversion over all probe rounds improves counts/prototypes | **Rejected** — JOINT mean **0.283** vs TANGO-MB **0.011** |
| H2 | Coupled \((W, S, n)\) fixed-point iteration reduces drift bias | **Rejected** — COUPLED mean **0.272** |
| H3 | Terminal trajectory / midpoint bias correction helps | **Rejected** — midpoint mean **0.272** |
| H4 | D-opt round weighting in joint LS beats uniform-round pick | **Rejected** — DOPT = JOINT (**0.283**) |
| H5 | `passive_mb_scale_only` restores R11 scaling-only (~0.15) | **Supported** — mean **0.151**, median **0.140** |
| H6 | TANGO-MB remains primary winner vs passive | **Supported** — unchanged vs R12 |

## Methods implemented

| Method | Description |
|---|---|
| **TANGO-JOINT** | Weighted LS on bias moments for \(n_c\), then joint weight LS for class sums |
| **TANGO-COUPLED** | 3 iterations: JOINT → \(\hat W\) → Jacobian bias correction → JOINT |
| **TANGO-DOPT** | JOINT with \(w_r \propto 1/\|p^{(r)}-\mathbf{1}/C\|\) |
| **TANGO-trajectory-midpoint** | \(W_{\mathrm{mid}}=(W_0+W_M)/2\) bias correction + JOINT |
| **passive_mb_scale_only** | R11: `scale_terminal_deltas` + `tango_estimate_sums` |
| **passive_mb_r12_coupled** | R12 degraded path (for audit; equals scale-only on passive probes) |
| **tango_mb** | Round-12 primary (retained) |

## Experiment plan

- **Script:** `code/benchmark_round13.py`
- **Seeds:** 8 per scenario; aggregates: mean, std, **median**, **IQR**
- **Primary:** `minibatch_sgd` (B=8, T=6, \(T_{\mathrm{eff}}=18\))

## Results — primary `minibatch_sgd` (`artifacts/round13_metrics.json`)

| Method | Proto MSE mean | Proto MSE median | Proto MSE IQR | Count MAE mean |
|---|---:|---:|---:|---:|
| passive_multi_round | 0.753 | 0.751 | — | 0.295 |
| **passive_mb_scale_only** | **0.151** | **0.140** | 0.032 | — |
| passive_mb_r12_coupled | 0.151 | 0.140 | — | — |
| **tango_mb** | **0.011** | **0.011** | 0.005 | **0.100** |
| tango_joint | 0.283 | 0.280 | 0.038 | 0.195 |
| tango_coupled | 0.272 | 0.276 | — | — |
| tango_trajectory_midpoint | 0.272 | — | — | — |

**Phase-2 gate:** `tango_mb` **0.011** < passive **0.753** — **met** (unchanged from Round 12).

**Active vs scaling (honest):** active / R11 scaling-only ≈ **0.151 / 0.011 ≈ 13.7×** (not R12’s inflated ~26× using coupled passive_mb **0.284**).

**New directions vs TANGO-MB:** None beat **0.011**; least bad among failures = **tango_coupled** (**0.272**).

## Comparison to Round 12

| Metric | Round 12 | Round 13 |
|---|---:|---:|
| tango_mb proto MSE mean | 0.0110 | **0.0110** (identical) |
| tango_mb proto MSE median | 0.0114 | **0.0114** |
| passive proto MSE mean | 0.753 | 0.753 |
| scaling-only proto MSE | ~0.150 (stack_mb_ridge) / 0.284 (broken passive_mb) | **0.151** (`passive_mb_scale_only`) |
| New best method | drift2 0.025 (worse than MB) | coupled 0.272 (worse than MB) |

## Theory / documentation fixes

- `paper/proofs.md`: drift mean **0.112** (not 0.42); drift2 demoted; **Lemma MB-JOINT** documents negative joint result.
- `paper/draft.md`: Round 13 table + honest scaling ratio.

## Honest conclusions

1. **Bookkeeping win stands:** TANGO-MB at \(W_0=0\) is still the only strong estimator on primary.
2. **Joint / coupled / trajectory** do not close the stochastic within-step gap; aggressive rounds break shared-\(n_c\) bias linearization.
3. **Methodological win:** `passive_mb_scale_only` fixes the scaling-vs-active narrative for Phase-2 ACCEPT prose.
4. **Still open:** SHARD `level3_invert` cross, general FL scope, approximate tier gap (~73× vs full-batch).

## Artifacts

- `artifacts/round13_metrics.json`
- `artifacts/round13_minibatch_methods.svg`
- `logs/experiment_round13.log`
