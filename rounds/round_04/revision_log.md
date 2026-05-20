# Round 04 — Revision Log

## What changed

- Pivoted from Round-03 Stage-3 polish to Stage-2/snapshot disaggregation, following the updated user correction in `config.json`.
- Diagnosed SHARD's core weakness as an identifiability and observation-model issue: the method assumes complete intermediate batch-gradient observations and a full-rank incidence design.
- Proposed three directions:
  - GARD: graph-regularized Bayesian/MAP disaggregation under missing batch rows.
  - BLRD: low-rank Bayesian factor disaggregation.
  - Active/partial observation recovery by incidence-rank-aware sampling.
- Selected GARD for the minimal runnable benchmark because it directly addresses missing/subsampled gradient observations while staying closest to SHARD's Stage-2 linear system.

## Experimental choices

- Built a synthetic oracle-incidence Stage-2 simulator instead of running the full SHARD pipeline. This isolates the mathematical bottleneck \(M_{\Omega}=H_{\Omega}S+\epsilon\).
- Used the fixed-assignment SHARD least-squares update as the baseline. This is favorable to SHARD because it removes the hard assignment problem and tests only the missing-row identifiability issue.
- Added a graph-smooth synthetic snapshot generator to test the condition under which GARD is supposed to help: observations are rank deficient but a public/side-information graph prior is valid.
- Saved metrics to `artifacts/round04_metrics.json`, a plot to `artifacts/round04_stage2_missing_observations.png`, and the run log to `logs/experiment_round04.log`.

## Honest negative result and pivot

An initial low-rank factor MAP variant (BLRD) was tried during development. It did not improve over SHARD-style least squares in this small benchmark and sometimes trailed it. I therefore pivoted to graph-regularized MAP recovery, which gave clear gains in rank-deficient missing-observation regimes.

## Key result

GARD does **not** beat SHARD-style LS when all 60/60 batch-gradient rows are present and the incidence-plus-mean system has full rank. This is expected and should be handled by a rank-aware gate.

Under missing observations, GARD reduces snapshot MSE versus SHARD-style LS:

- 36/60 rows: 0.03423 vs 0.25272, a 7.42x MSE reduction.
- 24/60 rows: 0.07110 vs 0.48769, a 7.47x MSE reduction.
- 15/60 rows: 0.13252 vs 0.67938, a 5.43x MSE reduction.
- 9/60 rows: 0.28797 vs 0.79260, a 2.93x MSE reduction.

## Limitations

- The benchmark uses oracle incidence. It does not yet solve SHARD's hidden assignment problem.
- The graph prior is only valid if the attacker has defensible public metadata, acquisition order, temporal locality, or an inferred graph.
- `graph_lambda=0.35` is fixed, not tuned on a separate validation split.
- The simulator tests Stage 2 only; it does not propagate recovered snapshots into Stage 3 inversion.

## Next round recommendation

Implement a GARD-regularized alternating SHARD loop: use SHARD's assignment step, but replace the update step with the graph posterior solve and add a rank-aware gate that sets the graph penalty to zero when observations are complete/full-rank. Then test with unknown assignments and a validation-based lambda selection rule.
