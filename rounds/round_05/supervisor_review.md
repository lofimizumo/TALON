# Round 05 Supervisor Review

## Verdict

REVISE_MAJOR / PIVOT_REQUIRED

GARD remains viable only as a conditional Stage-2 regularizer for fixed or near-correct incidence. It is not yet viable as the primary Round 06 method for the updated major goal: significantly better than SHARD, Stage-2/full-method priority, reduced reliance on all intermediate gradients, and not oracle-only. Round 06 should pivot away from "GARD alone" toward joint soft assignment or another assignment-first Stage-2 method, with graph smoothness demoted to a secondary prior.

## Executive Summary

Round 05 is substantially more honest than Round 04. It directly addresses the prior supervisor requirements by adding non-oracle incidence regimes, graph mismatch regimes, ridge/low-rank baselines, validation-selected regularization, and threat-model reduction accounting. The main result is negative in the right way: GARD can reduce observed Stage-2 batch-gradient rows under oracle assignment plus an aligned graph, but it does not solve the unknown-assignment bottleneck.

The reported key numbers are reproduced by the saved JSON aggregates and match the log. At 40% random observed rows with oracle assignment, oracle-graph GARD has mean snapshot MSE 0.0236 versus SHARD-style LS 0.4825. But at the same budget under wrong20, wrong40, and unknown_random assignments, oracle-graph GARD degrades to 0.3625, 0.8294, and 1.0059 respectively; no method reaches the target MSE <= 0.10 under assignment error. This is decisive evidence that assignment error dominates the graph prior.

The central remaining concern is that even the positive result is still oracle-heavy: the benchmark supplies the true snapshot mean as a mean anchor and uses a graph aligned with the synthetic snapshot generator. The graph prior is useful for nullspace filling when the model is already structurally correct, but the method still has not shown a non-oracle path to the graph, the assignments, or a full SHARD replacement.

## Honesty / Reproducibility Audit

I inspected:

- `rounds/round_05/researcher_proposal.md`
- `rounds/round_05/revision_log.md`
- `code/benchmark_round05.py`
- `artifacts/round05_metrics.json`
- `logs/experiment_round05.log`
- `rounds/round_04/supervisor_review.md`

The code appears to compute the reported metrics rather than inject desired outcomes. `benchmark_round05.py` constructs the simulator, incidence corruption regimes, graph regimes, validation split, hyperparameter selection, aggregation, threat-reduction table, plots, and JSON output in a single script. The log records the same seeds, observation budgets, assignment regimes, per-seed overlaps, ranks, and headline method MSEs that appear in the saved aggregate JSON.

I spot-checked the aggregate JSON using a separate standard-library parser. The proposal's headline table is consistent with the saved metrics:

- Oracle assignment, 40% random rows: SHARD LS 0.4825, ridge 0.4985, low-rank 0.4998, GARD oracle graph 0.0236, GARD noisy graph 0.1765, GARD wrong graph 0.4825, co-occurrence graph 0.5818, LS-kNN graph 0.4514.
- Assignment stress at 40% random rows: oracle-graph GARD is 0.0236 under oracle assignment, 0.3625 under wrong20, 0.8294 under wrong40, and 1.0059 under unknown_random.
- Target MSE <= 0.10 under random oracle assignment is reached by SHARD-style LS/ridge/low-rank at 48/60 rows, oracle-graph GARD at 15/60 rows, and noisy-graph GARD at 36/60 rows.
- No random-row method reaches target under wrong20, wrong40, or unknown_random.

I attempted a cheap rerun with the available system `python3`, but it failed immediately because `matplotlib` is unavailable:

```text
ModuleNotFoundError: No module named 'matplotlib'
```

No run-folder virtual environment was present for rerunning the script locally. Therefore this audit verifies reproducibility by code inspection, saved JSON/log consistency, and a failed environment check, not by a fresh full rerun.

## Response to Round 04 Requirements

Round 05 addresses several Round 04 requirements well:

1. It moves beyond oracle incidence by testing `wrong20`, `wrong40`, and `unknown_random`, with observations generated from true hidden incidence but solvers using attacker-assumed incidence.
2. It stress-tests graph mismatch with oracle, noisy, wrong/permuted, co-occurrence, and LS-kNN graph sources.
3. It adds ridge LS and low-rank PCA baselines.
4. It selects ridge and graph regularization using held-out observed rows rather than snapshot MSE.
5. It implements a rank-aware gate: graph lambda is forced to zero when the assumed incidence plus mean anchor is full-rank.
6. It reports threat-model reduction using a fixed MSE target.

The requirements not yet satisfied are the most important ones:

1. There is still no actual unknown-assignment Stage-2 algorithm. The benchmark tests how methods behave when handed wrong/random incidence, but it does not infer incidence.
2. There is still no full-method result. Stage 1 gradient-to-batch-mean recovery and Stage 3 input reconstruction are omitted.
3. The positive graph result still depends on an oracle-aligned graph and a synthetic generator whose smooth sample-index geometry matches the oracle chain.
4. The method still uses an oracle true snapshot mean as the mean anchor. In `run_one`, `mean_snapshot = s_true.mean(axis=0)` is supplied to every solver. This is a favorable side channel unless the paper can justify how the attacker obtains the snapshot mean without reintroducing the all-gradient requirement.
5. The validation criterion is not enough to detect assignment error because it validates residuals under the same wrong incidence model. Round 05 correctly acknowledges this limitation.

## Technical Assessment

The fixed-incidence linear algebra is sound for the synthetic Stage-2 proxy. The MAP solve minimizes observed batch residuals with a mean anchor, optional ridge, and optional graph Laplacian. The validation split is conceptually appropriate: choose regularization on held-out observed batch rows and evaluate snapshot MSE only afterward.

The rank-aware graph gate is a reasonable response to Round 04. When the incidence plus mean anchor is full-rank, the graph prior is disabled. This prevents GARD from damaging fully identified oracle cases, but it also highlights the limited role of GARD: it is not a replacement for SHARD-style identification, only a missing-row nullspace prior.

The assignment corruption regimes are useful but should be interpreted carefully. `wrong20` actually replaces exactly one member in a batch of four, producing about 75% overlap, not 80%. `wrong40` replaces two members but can occasionally reselect a true member because the fill pool excludes only kept members rather than all true members, producing overlap around 0.51-0.55. This does not invalidate the conclusion; if anything, the tested errors are not harsher than their names suggest, and GARD still fails the target.

The strongest result is the negative assignment result. Even with an oracle graph, assignment error pushes snapshot MSE far above the target. This means graph smoothness cannot be the primary mechanism for surpassing SHARD unless the assignment problem is solved first.

## Major Issues

1. Critical: GARD does not solve the unknown-assignment bottleneck.

The benchmark hands every method an assumed incidence matrix. It measures robustness to wrong incidence but does not infer assignments. Under unknown_random, the best methods are around MSE 1.0 even with oracle/noisy graph priors. This fails the "not oracle-only" requirement.

2. Critical: The positive result still relies on oracle graph side information.

Oracle-graph GARD is excellent under oracle assignment, but co-occurrence and LS-kNN graphs do not deliver reliable gains. At 40% oracle assignment, co-occurrence is worse than SHARD LS (0.5818 versus 0.4825), and LS-kNN is only slightly better (0.4514) while still far from the oracle graph. This means the useful graph has not been obtained from allowed attacker information.

3. Major: The benchmark uses an oracle mean anchor.

All methods receive `s_true.mean(axis=0)`. If this mean is unavailable in the real threat model, then both SHARD-style LS and GARD are being evaluated with side information the attacker should not have. Round 06 must either justify this anchor from observable quantities or test sensitivity to estimated/noisy/no-anchor variants.

4. Major: Validation residual under wrong incidence is not a reliable model-selection signal.

The saved metrics show graph lambdas often become large under wrong/random assignments, but the snapshot MSE remains poor. Since validation rows share the same wrong assignment model, validation cannot distinguish "smooth plausible wrong solution" from true recovery.

5. Major: The method is still Stage-2 synthetic only.

The updated goal prioritizes Stage 2 or full method, and Stage 2 is appropriate here. But claims of being significantly better than SHARD require either integration with SHARD's actual assignment/update loop or a full-method proxy showing that Stage-2 gains survive downstream inversion.

6. Major: The synthetic data generator favors the oracle chain.

Snapshots are smooth low-rank signals over sample index, while the oracle graph is a chain over the same index. This is acceptable for diagnosing nullspace filling, but not for claiming realistic side information.

## Rubric Scores

- Novelty: 3/5. GARD itself is standard graph-Tikhonov/MAP disaggregation; the stronger novelty would be joint assignment plus graph-regularized Stage 2, which is not implemented yet.
- Soundness: 3/5. The fixed-assignment experiments are technically sound and honestly negative, but the core method does not address unknown assignment.
- Feasibility: 4/5. The linear solves are cheap, and Round 06 can afford broader algorithmic experiments.
- Impact: 2/5 currently, 4/5 potential. Impact is low if GARD remains oracle-only; it could become high if a joint soft-assignment method reduces intermediate-gradient requirements without oracle priors.
- Evaluability: 4/5 for synthetic Stage 2. The saved metrics/log are detailed and checkable, though the local environment lacks dependencies for rerun.

## Verdict on GARD Viability

GARD should not be the main Round 06 claim. It should be retained as a module or prior inside a new assignment-aware method.

Viable narrow claim:

> With correct batch membership and an aligned graph, graph-regularized Stage-2 disaggregation can fill missing-observation nullspaces and reduce observed batch-gradient rows by up to 75% in this synthetic benchmark.

Not viable claim:

> GARD significantly improves on SHARD under realistic unknown assignments or removes SHARD's reliance on all intermediate gradients.

Round 05 supports the first claim and refutes the second, at least for the current formulation.

## Required Round 06 Directions

1. Pivot to joint soft assignment and snapshot recovery.

Build a method that optimizes over soft incidence matrices and snapshots jointly, with constraints matching batch size, epoch coverage, nonnegativity, and row sums. Use graph smoothness only as one regularizer, not the main source of identifiability.

2. Add assignment metrics and failure criteria.

Report assignment overlap/accuracy, snapshot MSE, observed true residual, assumed residual, convergence, and runtime. Success should require both accurate assignment and snapshot recovery, not only low residual under an assumed incidence.

3. Replace validation with cross-epoch consistency.

Held-out residual under the same wrong incidence is insufficient. Try cross-epoch cycle consistency, agreement of independently recovered soft assignments across epochs, or held-out audit batches whose membership is partially known.

4. Remove or justify oracle side information.

Test no mean anchor, noisy/estimated mean anchor, and attacker-observable mean estimates. For graphs, require a graph source available without all intermediate gradients: public metadata, externally known acquisition order, or a learned graph from partial observations with no hidden labels.

5. Compare against SHARD's actual Stage-2 loop.

Implement the new update inside or beside SHARD's alternating assignment/update procedure on the same partial-observation budget. A ridge/low-rank/graph fixed-incidence benchmark is no longer enough.

6. Keep GARD as an ablation.

Round 06 should include: no graph, oracle graph, noisy graph, wrong graph, learned graph, and graph with lambda forced to zero/full grid. This will show whether the new assignment mechanism or the graph prior is responsible for any gains.

7. Define a stronger success threshold.

The next method should beat ridge/low-rank baselines under non-oracle assignment and reach MSE <= 0.10 at reduced observation budgets. If it cannot, the project should pivot away from GARD-family priors entirely.

## Bottom Line

Round 05 is a useful and credible negative audit of GARD. It narrows the method to a conditional nullspace prior and shows that the real unsolved problem is assignment. Round 06 must therefore be assignment-first: joint soft incidence, cross-epoch validation, no oracle graph/mean assumptions, and direct comparison to SHARD Stage 2 under partial observations.
