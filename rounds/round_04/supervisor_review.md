# Round 04 Supervisor Review

## Verdict

REVISE_MAJOR

## Executive Summary

Round 04 is a real pivot toward the stricter goal: it targets Stage 2 and missing intermediate batch gradients rather than Stage 3 polish. The GARD result is not obviously hardcoded; the metrics are generated from fixed seeds, logged per run, and the reported aggregates match the saved JSON/log. However, the benchmark is intentionally narrow and favorable to GARD: it uses oracle incidence, a chain graph prior, and synthetic snapshots generated from the same smooth ordered structure that the chain Laplacian rewards. This is honest as a bottleneck probe, but it is too toy to claim publication-level progress over SHARD or a meaningful threat-model reduction. GARD currently reduces missing-row ill-posedness only when the attacker already has a strong, correctly aligned sample graph and the hidden batch assignments are known. It does not yet mitigate access to all intermediate gradients in the full SHARD pipeline, because it does not solve missing Stage-2 observations with unknown assignments or from actual gradients. Round 05 must integrate GARD into an unknown-assignment Stage 2 or a whole-pipeline alternative, remove oracle priors, and test prior mismatch and validation-selected hyperparameters. The right framing is "promising conditional regularizer for rank-deficient fixed-incidence disaggregation," not "new attack that weakens SHARD's observation requirement."

## Honesty / Reproducibility Audit

The experiment is mostly honest and reproducible in the limited sense that the code computes the reported values rather than injecting them. `code/benchmark_round04.py:262-318` constructs the configuration, seeds, observation fractions, per-seed runs, aggregates, JSON output, and plot in one script. The log in `logs/experiment_round04.log:1-35` records the same seeds, fractions, and per-seed MSE/gain values as `artifacts/round04_metrics.json:33-133`, and the proposal's table matches those aggregate means. I did not find table-script result injection or a separate plotting script that hardcodes desired numbers.

The main reproducibility weakness is not hardcoding of metrics but experiment design overfitting. The data generator builds snapshots from smooth one-dimensional sinusoidal latent coordinates ordered by sample index (`code/benchmark_round04.py:68-88`), while GARD uses exactly a chain Laplacian over that same index order (`code/benchmark_round04.py:134-159`). This makes the graph prior unusually well specified; no scrambled graph, random graph, noisy metadata graph, block/class graph, or adversarially wrong graph is tested. The chosen `graph_lambda=0.35` is fixed in the config (`code/benchmark_round04.py:42-52`) and acknowledged as not validation-selected in `rounds/round_04/revision_log.md:35-40`; therefore the gain may reflect an unreported development-time tuning choice.

The benchmark also gives both methods the true incidence rows. `make_incidence` samples true batch membership (`code/benchmark_round04.py:91-104`), `subsample_rows` reveals the kept rows (`code/benchmark_round04.py:107-117`), and both `shard_style_ls` and `gard_recover` are called directly with `h_obs` (`code/benchmark_round04.py:175-185`). This is transparent in the proposal and revision log, but it means the experiment does not reproduce SHARD's actual hidden-assignment task.

The baseline is favorable to SHARD in one dimension and unfavorable in another. It is favorable because it gets oracle incidence and avoids assignment error (`rounds/round_04/revision_log.md:13-18`). It is unfavorable because the true data-generating process exactly satisfies GARD's smooth graph prior, while the baseline is a minimum-norm least-squares estimator with no comparable low-rank/smoothness prior. A fairer benchmark needs at least: Tikhonov/ridge LS, low-rank MAP, graph prior with wrong/noisy graph, and SHARD's actual alternating assignment/update loop under missing rows.

## Feasibility & Resources

The current GARD solve is computationally feasible for the pilot. It solves an `N x N` linear system with `dim_g` right-hand sides (`code/benchmark_round04.py:144-159`), which is cheap at `N=48` and should still be tractable for moderate client sizes if implemented with sparse Laplacians or conjugate gradient. The benchmark runs in about 0.03 seconds (`logs/experiment_round04.log:33-35`), so Round 05 can afford a much broader stress test without resource concerns.

Scaling risks remain for realistic full-pipeline integration. The current SHARD Stage 2 assignment loop is already combinatorial/iterative (`supplementary_materials/code/shard_sim/attacker.py:158-366`), and GARD only replaces the fixed-assignment update. Adding graph regularization is feasible, but the hard part is missing-observation assignment: constructing/optimizing over unknown assignments when some epochs or batches are absent. A publication-worthy implementation must show convergence behavior, runtime, and robustness for larger `N`, different batch sizes, and partial observations that remove entire epochs or structured logging windows, not only uniformly sampled rows.

## Theory & Assumptions

The theory is plausible as a regularized inverse problem but not yet a new identifiability result. GARD solves a MAP/posterior mean problem for a Gaussian smoothness prior (`rounds/round_04/researcher_proposal.md:21-33`), and the observed gains are exactly what should happen when the true signal lies near the low-frequency subspace of the graph Laplacian. This supports the claim that priors can fill nullspaces, but not the stronger claim that SHARD's access requirement is mitigated under realistic honest-but-curious observations.

The strongest assumption is the graph/prior oracle. The proposal suggests public metadata, temporal acquisition order, user/session metadata, or inferred co-occurrence graph (`rounds/round_04/researcher_proposal.md:31-33`), but the benchmark does not instantiate or validate any of these. If the graph is inferred from a fully observed prefix, then the attacker may still require exactly the kind of intermediate-gradient access the new method is supposed to reduce. If the graph is public metadata, Round 05 must show the metadata is available before attack time, correlated with snapshots, and not equivalent to private label/order information.

The synthetic Stage-2 benchmark is faithful enough to demonstrate one mathematical failure mode: rank-deficient fixed-incidence least squares can fit observed batch means while failing at snapshot recovery (`artifacts/round04_metrics.json:55-132`). It is not faithful enough to claim progress over SHARD as an attack. It omits gradient-to-batch-mean inversion, assignment uncertainty, coefficient matrix conditioning, Stage 1 mean errors, Stage 3 reconstruction quality, realistic data distributions, and graph construction from observable side information.

## Rubric Scores

- Novelty: 3/5. Graph-regularized disaggregation is a sensible and potentially useful adaptation, but as implemented it is a standard MAP/Tikhonov prior applied to a synthetic inverse problem.
- Soundness: 2/5. The fixed-incidence linear algebra is sound, but the main claim depends on an oracle graph and oracle incidence.
- Feasibility: 4/5. The method is simple to implement and computationally cheap; full SHARD integration is plausible.
- Impact: 3/5. It could address the stricter Stage 2/missing-gradient goal if the graph prior can be justified, but the current evidence is conditional and narrow.
- Evaluability: 3/5. The script is runnable and logs seeds/metrics, but lacks independent validation, ablations, and full-pipeline tests.

## Issues

1. [Critical] Oracle incidence means the benchmark does not test the central Stage-2 problem. The current SHARD Stage 2 has to infer assignments (`supplementary_materials/code/shard_sim/attacker.py:175-183`), but Round 04 gives both GARD and LS the true `h_obs` (`code/benchmark_round04.py:175-185`). This prevents any claim that GARD works in the actual SHARD pipeline.

2. [Critical] The graph prior is effectively oracle-aligned with the synthetic data generator. Smooth sinusoidal latent snapshots over sample index (`code/benchmark_round04.py:68-88`) are recovered with a chain Laplacian over the same order (`code/benchmark_round04.py:134-159`). This is a designed win condition, not evidence that public or inferable metadata suffices.

3. [Major] The method does not yet reduce the threat-model requirement in a realistic way. It reduces the number of observed rows only after true incidence and a correct graph are supplied. It may trade "access to all intermediate gradients" for "access to a high-quality sample graph/prior," which may be equally unrealistic or privacy-sensitive.

4. [Major] Hyperparameter selection is not publication-ready. `graph_lambda=0.35` is fixed (`code/benchmark_round04.py:42-52`) with no validation protocol, sensitivity curve, or train/test split. The revision log admits this (`rounds/round_04/revision_log.md:35-40`).

5. [Major] The synthetic benchmark is too low-diversity. It uses one generator, one graph type, one noise level, one `N/B/E/dim_g` regime, and five seeds (`code/benchmark_round04.py:42-52`, `code/benchmark_round04.py:262-268`). This cannot support robust claims.

6. [Major] Baselines are incomplete. The comparison is only minimum-norm LS versus graph MAP. Missing baselines include ridge/Tikhonov without graph, low-rank/PCA prior, oracle low-rank MAP, wrong/noisy graph MAP, and SHARD alternating minimization under complete and partial observations.

7. [Major] No full-method metric is reported. The config now requires Stage 2 or full-method improvement and threat-model mitigation (`config.json:6-14`). Round 04 is Stage 2 only and does not propagate recovered snapshots into Stage 3 inversion or attack success metrics.

8. [Minor] The complete-observation result is correctly negative for GARD, but the proposed rank-aware gate is not implemented. The proposal says graph regularization should be gated off when incidence is full-rank (`rounds/round_04/researcher_proposal.md:31-33`, `rounds/round_04/revision_log.md:24-27`), but the code always applies `graph_lambda`.

9. [Minor] Uncertainty reporting is limited. Means and standard deviations are saved, but there are only five seeds and no confidence intervals, paired tests, or seed robustness across broader randomization (`artifacts/round04_metrics.json:33-133`).

## Actionable Suggestions

1. Implement a missing-observation Stage-2 algorithm with unknown assignments. Replace SHARD's update step (`supplementary_materials/code/shard_sim/attacker.py:284-308`) with a graph-regularized solve, but retain and stress-test the assignment step under missing rows. Report assignment accuracy, snapshot MSE, observed residual, and convergence.

2. Remove the oracle graph assumption. Test at least four graph regimes: correct chain graph, permuted/wrong chain graph, noisy metadata graph, and graph inferred only from information the attacker is allowed to observe. If GARD only works in the correct-chain case, frame it as a conditional prior experiment, not a threat-model mitigation.

3. Use validation-selected hyperparameters. Choose `lambda` and mean-anchor weight from observed-only validation residual or a held-out subset of observed batch rows, then evaluate on held-out seeds and graph regimes. Do not tune on snapshot MSE.

4. Add strong baselines. Include ridge LS, low-rank MAP, Laplacian smoothing with wrong graph, SHARD's original alternating loop on the same partial observations, and a simple imputation baseline that fills missing rows before SHARD.

5. Make the synthetic benchmark more faithful. Generate snapshots from the existing `shard_sim` surrogate pipeline when possible, recover batch means from coefficient matrices and gradients, inject Stage 1 mean error, and vary `N`, `B`, `E`, `dim_g`, noise, and missingness patterns.

6. Demonstrate actual threat-model reduction. Define observation budgets such as 100%, 60%, 40%, and "one logged epoch plus sparse telemetry"; compare complete SHARD versus partial-observation GARD-SHARD on the same data. The claim should be phrased as "achieves comparable reconstruction with X% of intermediate gradients under stated side information" only if the full pipeline supports it.

7. Add prior-mismatch failure analysis. Publication-worthy work must show when GARD fails, not only when it wins. Report performance as graph correlation with the true snapshot geometry degrades.

8. Connect Stage 2 to Stage 3. Even if Stage 2 MSE improves, show whether the recovered snapshots improve final input reconstruction or privacy leakage. If Stage 3 saturates or is insensitive, the contribution should be framed accordingly.

9. State a precise theorem or proposition. A useful Round 05 theory result would bound GARD error in terms of observed incidence rank, observation noise, graph smoothness energy, and graph mismatch, making clear that identifiability comes from the prior and is conditional.

10. Keep the honest negative results. The complete-observation case where LS beats GARD is important evidence that this is a conditional fallback, not a universal replacement. Implement the rank-aware gate and report both gated and ungated versions.

