# Round 14 — Phase-2 scoped closure (REVISE_MINOR from Round 13)

## Problem & goal

Round 13 supervisor **REVISE_MINOR** left four items before scoped Phase-2 ACCEPT:

1. SHARD `level3_invert` baseline cross (config gate)
2. Fix **TANGO-JOINT ≡ TANGO-DOPT** duplicate in code
3. Explicit **scoped acceptance package** (`paper/phase2_scope.md`)
4. `benchmark_round14.py` with frozen MLP + minibatch, label noise, median/IQR, SHARD row

## Hypotheses

| ID | Claim | Test |
|---|---|---|
| H1 | Uniform-weight JOINT differs from D-opt DOPT on primary | Compare `tango_joint` vs `tango_dopt` medians |
| H2 | SHARD tier cross is documentable without MNIST | Synthetic `level3_invert` via `shard_cross_round14.py` |
| H3 | TANGO-MB primary win persists after JOINT fix | `phase2_primary_win` on `minibatch_sgd` |
| H4 | Scoped ACCEPT is supportable with explicit fences | `paper/phase2_scope.md` + JSON `phase2_scoped_accept` |

## Methods

| Change | Implementation |
|---|---|
| JOINT fix | `tango_joint_estimate_sums` uses `round_weights=None` (uniform) |
| DOPT | Unchanged D-opt weights in `benchmark_round13.tango_dopt_estimate_sums` |
| SHARD cross | `code/shard_cross_round14.py` → `shard_baseline_cross` in metrics |
| Benchmark | `code/benchmark_round14.py` |

## Experiment plan

```bash
cd /workspace
python3 code/benchmark_round14.py
```

- **Primary:** `minibatch_sgd` (8 seeds, median/IQR)
- **Stress:** `frozen_mlp_minibatch`, `minibatch_label_noise`, `minibatch_terminal_noise`
- **Reference:** `balanced_clean` exact tier

## Expected deliverables

- `artifacts/round14_metrics.json`, `artifacts/round14_minibatch_methods.svg`
- `logs/experiment_round14.log`
- `paper/phase2_scope.md`
- Updated `paper/proofs.md`, `paper/draft.md`, `tutorial/tutorial.md`

## Success criteria

- `joint_vs_dopt_identical` = 0 on primary (methods measurably distinct)
- `shard_baseline_cross.require_baseline_comparison_vs_shard` addressed
- `phase2_primary_win` = 1
- Scoped accept block in metrics JSON aligns with `phase2_scope.md`
