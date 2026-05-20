# Round 06 Supervisor Review

## Verdict

REVISE_MAJOR / PIVOT_REQUIRED

JASPER is a real joint soft-incidence prototype, not merely a renamed fixed-incidence smoother. It constructs and iteratively updates a soft assignment matrix, enforces row batch-size and per-epoch capacity constraints, and alternates that assignment with snapshot recovery. However, the empirical and code audit show that the implemented assignment signal is too weak to mitigate the actual threat model. It mostly turns fixed wrong incidence into a smoother soft wrong incidence, often reducing the residual under its own soft assignment while failing to recover true batch membership or per-sample snapshots.

Round 07 should not return to GARD as the main method and should not spend the main round only tuning JASPER. The strongest direction is a fundamental pivot away from per-batch incidence recovery from batch averages alone: test an aggregate-only or server-trajectory/model-update inversion formulation that uses information actually available to an honest-but-curious server across training dynamics, and evaluate whether it can produce useful leakage without solving the label-symmetric within-epoch assignment problem.

## Audit Scope

I inspected:

- `rounds/round_06/researcher_proposal.md`
- `rounds/round_06/revision_log.md`
- `code/benchmark_round06.py`
- `artifacts/round06_metrics.json`
- `logs/experiment_round06.log`
- `rounds/round_05/supervisor_review.md`

The saved proposal, metrics JSON, and experiment log are mutually consistent for the main headline numbers. I did not rerun the benchmark because the requested task was an audit of the saved Round 06 artifacts, not a fresh replication run.

## Executive Summary

Round 06 correctly responds to the Round 05 mandate by pivoting from GARD as the main claim to an assignment-first method. The benchmark removes the most serious Round 05 side channel by replacing the true snapshot mean with an observed batch-average mean estimate. It also adds useful assignment metrics: hard top-B overlap, soft true-member mass, true observed batch residual, used observed batch residual, and target-MSE reachability.

The result is honestly negative. At 40% random observed rows, JASPER improves over naive fixed-incidence LS under corrupted priors, but it remains worse than the GARD conditional smoother and far above the target MSE:

- Oracle: SHARD LS 0.6101, GARD 0.2970, JASPER 0.3707.
- Corrupt25: SHARD LS 1.0720, GARD 0.5419, JASPER 0.6147.
- Corrupt50: SHARD LS 1.3834, GARD 0.7432, JASPER 0.8017.
- Unknown epoch: SHARD LS 1.7061, GARD 1.0621, JASPER 1.1095.

At full random observation, JASPER reaches low MSE only with oracle assignment. Under corrupt25 it is 0.3175, under corrupt50 it is 0.5683, and under unknown epoch it is 1.7446. No non-oracle assignment regime reaches the target MSE <= 0.10 for any method under either random-row or prefix-epoch protocols.

The most important failure is not just high snapshot MSE. Under `unknown_epoch`, JASPER's hard assignment overlap remains near random: 0.132 at 40% random rows and 0.094 at 100% random rows. The method therefore does not solve the assignment bottleneck that Round 05 identified.

## Is JASPER a Real Joint Soft-Incidence Method?

Yes, but it is a shallow one.

Evidence that it is real:

- `run_jasper` initializes from an incidence prior, solves a snapshot MAP system, updates soft assignments, damps the update, and repeats for 16 iterations.
- `sinkhorn_epoch_capacity` enforces nonnegative soft membership with row mass equal to the batch size and per-epoch column capacities capped near one.
- The JASPER snapshot solve uses its current soft incidence, not the original hard prior.
- The metrics evaluate both the soft incidence itself and the recovered snapshots.

Evidence that it is not yet deep enough:

- The assignment update uses a per-sample proxy cost `||s_i - m_b||^2`, where each estimated sample snapshot is compared to a batch mean. That is not the true batch-average objective, which depends on combinations of B samples jointly matching a mean.
- The Sinkhorn projection is local within each epoch and does not impose a cross-epoch identity or trajectory consistency principle strong enough to break label symmetry.
- The method is seeded by an incidence prior and uses regime-specific prior powers: oracle gets power 4.0, corrupt25 gets 1.5, corrupt50 gets 0.75, unknown gets 0.0. This is acceptable for a diagnostic benchmark, but it means the method is partly told how much to trust the prior rather than learning that trust level from attacker-observable evidence.
- JASPER frequently drives the used observed residual very low while the true observed residual and snapshot MSE remain poor. This is classic evidence of fitting a self-consistent wrong mixture.

So the answer is not "superficial" in the sense of no joint optimization. It is superficial in the sense that the soft-assignment subproblem is a proxy relaxation that lacks the information needed to identify true membership.

## Fairness of Comparisons

The comparisons are mostly fair for the narrow question asked in Round 06: can a joint soft-incidence Stage-2 method outperform fixed-incidence LS and a conditional graph smoother when assignments are corrupted or unknown?

Positive fairness points:

- All methods receive the same observed batch averages and the same observed row/epoch metadata.
- The true snapshot mean is not supplied; `estimate_mean_snapshot(m_obs)` is used.
- Observations are generated from the true hidden incidence, while solvers use oracle, corrupted, or unknown priors depending on regime.
- The proposal accurately reports that JASPER fails the target rather than cherry-picking the regimes where it helps fixed LS.
- The log and JSON agree on seeds, budgets, per-seed MSEs, assignment overlaps, and generated artifacts.

Important limitations:

- GARD is not an assignment method, so beating or losing to GARD does not directly answer whether JASPER is solving assignment. The assignment metrics do answer this, and they are negative.
- JASPER and GARD use different graph regularization strengths: JASPER uses 0.03 and GARD uses 0.3. This makes sense because graph smoothing is intended to be secondary in JASPER, but it also means the comparison is not a clean "same prior plus assignment" ablation.
- Hyperparameters are fixed rather than selected using attacker-observable validation. This is acceptable for Round 06 exploration, but Round 07 should avoid method-specific hand tuning unless it is symmetric across baselines.
- The corruption regimes still reveal their regime label to JASPER through `assignment_prior_power`. A realistic attacker may know whether metadata is audited, guessed, or absent, but not necessarily the exact prior quality.
- The benchmark remains synthetic Stage 2 only. It does not test Stage 1 gradient extraction, SHARD's actual assignment/update loop, or downstream input inversion.

These limitations do not invalidate the negative conclusion. If anything, several choices are favorable to JASPER, and it still fails.

## Threat-Model Assessment

JASPER does not materially mitigate the threat model.

The target was snapshot MSE <= 0.10 with reduced reliance on complete intermediate batch gradients and without oracle assignment. The saved threat-reduction table shows:

- Random-row protocol: only oracle assignment reaches target, and only at full 45/45 observed rows for all three methods.
- Prefix-epoch protocol: oracle SHARD reaches target at 4/5 epochs, while oracle GARD and oracle JASPER require 5/5 epochs.
- No corrupted or unknown assignment regime reaches target for any method.

The unknown-assignment setting is decisive. With no within-epoch sample membership, the problem is label-symmetric and underidentified in this Stage-2 formulation. JASPER's low used residual is not evidence of attack success because it measures the residual under the method's own soft assignment. At 40% random rows in `unknown_epoch`, JASPER has used observed batch MSE 0.0102 but true observed batch MSE 0.2185, hard overlap 0.1319, and snapshot MSE 1.1095. The method is fitting an internally plausible mixture, not recovering the true sample snapshots.

This means Round 06 strengthens the Round 05 conclusion: graph priors and weak soft assignment are conditional aids, not a path around missing batch membership.

## Major Issues

1. Critical: unknown within-epoch assignment remains unsolved.

JASPER's overlap under `unknown_epoch` is essentially random. The method does not provide a credible route from batch-average observations and epoch counts to per-sample identity.

2. Critical: the assignment objective is not the true combinatorial batch-average objective.

The cost compares individual sample estimates to batch means, then applies Sinkhorn-style row and capacity constraints. Real batch matching depends on sets of B samples whose average matches the observation. This mismatch can lower residuals while moving away from true membership.

3. Major: used residual can be misleading.

The metrics correctly include both used and true residuals. The discrepancy is large in unknown regimes. Round 07 should treat low self-residual as a diagnostic for overfitting unless it is accompanied by independent consistency checks or downstream inversion success.

4. Major: prior-quality knowledge is baked into the method.

The `assignment_prior_power` schedule gives JASPER different trust in the prior depending on the regime. This is useful for a controlled experiment, but a deployable attack would need an observable way to infer or validate prior reliability.

5. Major: JASPER is usually worse than the conditional smoother.

At 40% random rows, JASPER loses to GARD in every assignment regime despite being the assignment-aware method. At 100% rows, JASPER only slightly beats GARD in oracle assignment and one noisy corrupt50 average in the log/proposal is close; it does not establish a robust advantage.

6. Major: the evaluation is still Stage 2 only.

This is acceptable for diagnosing the bottleneck, but claims against SHARD must eventually test either the actual SHARD pipeline or a replacement attack objective with a comparable privacy endpoint.

## Rubric Scores

- Novelty: 3/5. JASPER is a reasonable soft-assignment relaxation for Stage 2, but Sinkhorn-style soft matching plus ridge/graph MAP recovery is not yet a distinctive attack mechanism.
- Soundness: 3/5. The code and metrics are internally consistent, and the negative interpretation is honest. The main weakness is the proxy assignment objective.
- Fairness: 3/5. Baselines are reasonable and the oracle mean issue was fixed. Remaining concerns are method-specific hyperparameters and regime-specific prior powers.
- Threat-model fit: 2/5. The method addresses the right bottleneck but fails in the setting that matters most.
- Impact: 2/5 currently. Impact would be low if the claim remains per-sample snapshot recovery under unknown assignment; the negative result is useful but not publishable as a positive attack.
- Evaluability: 4/5. The artifacts are detailed, deterministic, and auditable.

## Should Round 07 Improve JASPER, Return to GARD, or Pivot?

Do not return to GARD as the main method. Round 05 and Round 06 both show that GARD is a conditional nullspace prior. It is useful as an ablation and maybe as a regularizer inside another method, but it does not solve the core missing-incidence problem.

Do not make Round 07 primarily "JASPER with better tuning." Better temperatures, more iterations, a different graph lambda, or slightly improved Sinkhorn damping are unlikely to overcome label symmetry. A deeper JASPER variant would need a fundamentally different assignment signal: cross-epoch cycle consistency, partial audited rows, public metadata, or a true set-based matching objective. Without such information, it is still trying to infer identities from an underidentified mixture model.

The strongest Round 07 direction is a pivot to a fundamentally different attack surface:

**Server-trajectory / model-update inversion under aggregate observations.**

Instead of recovering per-batch incidence and per-sample snapshots from intermediate batch means, formulate the attack around the information an honest-but-curious server naturally has: initial model, final model or round updates, optimizer state if visible, global hyperparameters, selected client participation, and possibly a sequence of server checkpoints. The question becomes whether aggregate trajectory constraints leak inputs, prototypes, labels, or class-level representatives without assigning every sample to every hidden minibatch.

Concrete Round 07 benchmark:

- Compare four attack targets: aggregate-only terminal update, multi-checkpoint server trajectory, terminal-gradient approximation, and existing Stage-2 JASPER/GARD baselines.
- Use the same synthetic data first, but make success metrics appropriate to aggregate leakage: reconstruction MSE up to permutation, prototype recovery, label/class leakage, membership/property inference, and downstream inversion quality.
- Avoid oracle incidence entirely in the main setting. Oracle incidence can remain as an upper bound, not a claim.
- If active perturbation is considered, keep it within honest-but-curious constraints: standard server choices such as initialization, learning rate schedule, checkpoint frequency, or benign probe batches only if the protocol would realistically allow them. Do not assume malicious gradient manipulation unless the threat model is explicitly changed.
- Include a hard falsification criterion: if aggregate-only/server-trajectory constraints cannot beat non-inversion baselines or recover useful prototypes without hidden incidence, then the project should pivot from "SHARD replacement" to a negative-identifiability paper.

Why this direction is strongest: Round 06 shows that per-batch assignment is the wrong bottleneck to brute force from batch averages alone. Server trajectory and model-update inversion change the inverse problem so that assignment may not be required, or the target can be weakened to permutation/prototype-level leakage that is actually identifiable.

## Required Round 07 Deliverables

1. A non-incidence main method.

Implement at least one aggregate-only or server-trajectory inversion baseline that does not consume true, corrupted, or random per-batch incidence as a central input.

2. A threat-model table.

State exactly what the honest-but-curious server sees: final update, checkpoint sequence, optimizer state, hyperparameters, batch size, client dataset size, labels, public metadata, and any allowed active choices.

3. A fair baseline suite.

Keep SHARD fixed-incidence, GARD conditional, and JASPER as upper-bound/diagnostic baselines, but do not let oracle-incidence methods define success.

4. Identifiability-aware metrics.

Report per-sample reconstruction only when identity is actually identifiable. Otherwise report permutation-invariant, prototype, distributional, class/property, and downstream inversion metrics.

5. A negative-result path.

If the aggregate/server-trajectory method fails, write it as a theorem-backed or experiment-backed impossibility/identifiability result rather than another local optimizer failure.

## Bottom Line

Round 06 is a useful negative round. It confirms that a real joint soft-incidence implementation does not solve SHARD's hidden-assignment bottleneck under the tested conditions. The project should now stop treating assignment recovery from Stage-2 batch averages as the main path. Round 07 should pivot to aggregate-only or server-trajectory/model-update inversion, with GARD and JASPER retained only as diagnostic baselines.
