# Round 09 Revision Log

## Supervisor Gaps Addressed (from Round 08 review)

| Round 08 requirement | Round 09 action |
|---|---|
| Minibatch SGD stress test | Added `minibatch_sgd` scenario; documented failure |
| Nonzero initial head-weight stress | Added `nonzero_init_head` scenario |
| Larger class count and dimension | Added `large_10class_30dim` (10 classes, 30 dims) |
| Passive multi-round baseline | `passive_multi_round` with repeated neutral bias |
| Public/statistical prototype prior | `public_prior` (crude single-round mean prior) |
| Oracle aggregate upper bound | `oracle_aggregate` separated from TANGO |
| Count vs prototype reporting | Separate `count_mae`, `count_relative_error`, `prototype_mse` |
| Theorem exact vs approximate split | `theorem_scope` in metrics JSON + proposal section |
| Threat model wording | "active terminal probing" throughout |
| Tutorial cleanup | Added Round 09 section; no duplicate JOLI block |

## Code Added

`code/benchmark_round09.py`:

- Extends Round 08 simulator with minibatch local training and nonzero `w0`.
- Evaluates four methods per scenario: TANGO active, passive multi-round, public prior, oracle aggregate.
- Writes `artifacts/round09_metrics.json`, `artifacts/round09_baselines.svg`, `artifacts/round09_count_vs_proto.svg`, `logs/experiment_round09.log`.

Command:

```bash
python3 code/benchmark_round09.py
```

Runtime: under 2 seconds.

## Key Numeric Changes

### Balanced clean (8 rounds, 8 seeds)

| Metric | Round 08 | Round 09 |
|---|---:|---:|
| TANGO prototype MSE | `0.000151` | `0.000151` (reproduced) |
| TANGO count MAE | (bundled) | `0.0419` (now explicit) |
| Passive prototype MSE | `0.143141` (1 round headline) | `0.143141` (8 neutral rounds) |
| Public prior prototype MSE | n/a | `0.412694` |
| Oracle prototype MSE | n/a | `0.0` |

### New stress results (TANGO active)

- **minibatch_sgd:** prototype MSE `6.5877`, count MAE `2.5743` -- **negative result** (worse than passive `0.7532`).
- **nonzero_init_head:** prototype MSE `0.0097`, count MAE `0.7428` -- degraded counts, moderate prototype error.
- **large_10class_30dim:** prototype MSE `1.06e-5`, count MAE `0.0294` -- scale-up succeeds.
- **more_local_steps:** prototype MSE `0.0019` (matches Round 08 trend).
- **high_within_class_variance:** prototype MSE `0.000358`, individual MSE `0.9828`.

## Interpretation Changes

1. **Minibatch is a hard limit for the current first-order TANGO estimator.** The paper must classify minibatch FL under the approximate/future-work tier, not the exact theorem.
2. **Oracle aggregate is not TANGO.** It is an upper bound showing perfect moment recovery would yield zero prototype error.
3. **Passive multi-round without probe design is weak** but can accidentally estimate counts better than TANGO in some seeds; prototype recovery remains ~`950x` worse than active TANGO on balanced data.
4. **Count recovery can degrade while prototype shape remains acceptable** (nonzero head case) -- motivates separate reporting in the paper.

## Tutorial and Paper Skeleton

- Updated `tutorial/tutorial.md` with Round 09 reproduction, baseline table, and theorem-scope note.
- Added `paper/outline.md` as a draft section skeleton for manuscript drafting.

## Claims Unchanged (discipline)

Still do **not** claim:

- Individual recovery from terminal-only observations.
- TANGO beats SHARD at SHARD's individual target.
- Passive honest-but-curious observation suffices.
- Exact theorem holds for minibatch SGD without modification.
