# Round 07 Supervisor Review

## Verdict

**ACCEPT_WITH_MAJOR_LIMITATIONS / REFINE_AND_INTEGRATE**

TANGO is the first direction in this run that materially relaxes SHARD's strongest and least realistic Stage-2 assumption. It does not observe intermediate minibatch gradients, batch-average rows, batch order, or a hidden incidence matrix. Under the stated linear-head setting, terminal model deltas from a small number of server-chosen classifier-bias probes identify class sums, class counts, class prototypes, and the dataset mean. The saved metrics support that claim: prototype MSE falls from 0.143141 with one neutral terminal round to 0.000274 with two rounds and 0.000151 with eight rounds.

However, TANGO should not be sold as individual sample recovery. The experiment itself shows individual reconstruction from repeated prototypes saturating at 0.266669, close to the within-class variance floor of 0.253177. That is not a failed optimizer; it is the expected non-identifiability of within-class zero-sum perturbations under aggregate class-sum observations. TANGO is viable as an aggregate/prototype leakage contribution and as an identifiability boundary for terminal-update attacks, not as a direct SHARD replacement if the paper's claim remains per-sample inversion.

Round 08 should therefore **refine TANGO and integrate it into a tiered attack/limits narrative**, with TANGO as the main positive terminal-update result and GARD/JOLI/JASPER retained only as conditional Stage-2 tiers or upper-bound diagnostics. Do not spend the final round trying to force individual recovery from TANGO aggregates alone.

## Audit Scope

I inspected:

- `rounds/round_07/researcher_proposal.md`
- `rounds/round_07/revision_log.md`
- `code/benchmark_round07.py`
- `artifacts/round07_metrics.json`
- `logs/experiment_round07.log`
- `rounds/round_06/supervisor_review.md`

The proposal, revision log, metrics JSON, code, and experiment log agree on the main configuration, seeds, observations, and headline results. I did not rerun the benchmark; this audit is based on the saved Round 07 artifacts.

## Executive Summary

Round 06 asked for a pivot away from hidden per-batch incidence recovery. Round 07 complies. TANGO changes the inverse problem from "recover per-sample snapshots from hidden batch averages" to "recover aggregate class statistics from terminal client updates under active but ordinary server initialization choices."

The central result is strong for the weaker target:

- One terminal round is rank deficient and gives poor prototype recovery: prototype MSE 0.143141, dataset mean MSE 0.143078.
- Two terminal rounds essentially identify the empirical prototypes in this synthetic setting: prototype MSE 0.000274, dataset mean MSE 0.000131.
- Four and eight rounds improve conditioning modestly: best prototype MSE 0.000151 and dataset mean MSE 0.0000717.
- Individual recovery remains poor: best individual MSE 0.266669 versus within-class variance 0.253177.

This is exactly the kind of honest pivot the previous reviews called for. It gives a plausible paper contribution if the paper is reframed around aggregate/prototype leakage under terminal updates, plus a formal negative result for individual identifiability. It is not enough if the intended headline is "we recover private individual samples without SHARD's intermediate gradients."

## Does TANGO Mitigate SHARD's Too-Strong Assumption?

**Yes, for aggregate leakage. No, for SHARD-equivalent individual inversion.**

The improvement over SHARD's observation model is real. TANGO does not require the server to see:

- intermediate minibatch gradients;
- per-batch average rows;
- batch order;
- per-batch incidence;
- per-sample snapshots.

This directly answers the Round 06 supervisor request to test an attack surface based on what an honest-but-curious server naturally observes across training dynamics. The code records `observed_intermediate_batch_gradients = 0`, and the estimator consumes only terminal `delta_w`, terminal `delta_b`, public biases/probabilities, optimizer hyperparameters, and architecture.

The caveat is that the benchmark is still a very clean first setting. It uses a linear softmax head with zero initial weights, full-batch local training, a small learning rate, known client dataset size, known class-output semantics, and repeated server-chosen bias probes over the same synthetic client distribution. These choices are acceptable for a proof-of-concept identifiability experiment, but they are not yet a robust FL privacy attack evaluation.

The paper must state the threat model as active terminal probing, not passive observation of an arbitrary production training run. "The server can choose initial classifier biases across repeated rounds for the same client/data distribution" is weaker than "the server sees intermediate gradients," but it is still an active-probing assumption that needs protocol justification.

## Is Prototype/Aggregate Leakage Enough?

**Potentially yes, but only if the contribution is framed correctly.**

Prototype and dataset-statistic leakage can be a defensible privacy contribution when:

- the data distribution is sensitive at class or cohort level;
- class prototypes correspond to recognizable hidden representations or can be decoded downstream;
- the attack works from terminal updates rather than intermediate gradients;
- the paper proves what is and is not identifiable from the observation model.

Round 07 gives a good minimal version of that story. It also gives the correct negative boundary: terminal first-order class-sum observations are invariant to within-class zero-sum perturbations, so individual samples are not identifiable without extra information. That boundary is scientifically valuable because it prevents another round of optimizer tuning from being mistaken for progress.

But aggregate leakage alone is not automatically a top-tier paper claim. The final paper needs one of the following:

1. A stronger demonstration that recovered hidden prototypes map to meaningful raw-input or semantic leakage.
2. A principled theorem showing exact class-sum/count identifiability from terminal probes and individual non-identifiability under aggregate-only observations.
3. A tiered attack taxonomy showing TANGO as the low-observation aggregate tier, with GARD/JOLI/JASPER as higher-observation conditional tiers.

Without at least one of those additions, TANGO risks reading as a neat linear-head moment recovery exercise rather than a full privacy attack contribution.

## Experiment Honesty and Baseline Fairness

The experiments are honest in the most important sense: they do not hide the failure of individual recovery. The proposal explicitly says TANGO is viable for aggregate/prototype leakage but not for sample recovery, and the metrics include both prototype MSE and individual MSE against the within-class variance floor.

The saved outputs are internally consistent:

- `round07_metrics.json` and `experiment_round07.log` agree on seeds, configuration, and per-round metrics.
- The benchmark records zero observed intermediate gradients.
- The one-round baseline is correctly weak because the system is rank deficient.
- SHARD Stage 2 is marked not applicable under the same observations, which is the right statement: without batch-average rows or incidence, SHARD's Stage-2 machinery has no input.

Fairness limitations remain:

- The main setting is synthetic, low-dimensional, balanced, and class-separated.
- The local client update is full-batch, not minibatch SGD. This removes batch-order complexity and makes the first-order terminal approximation unusually clean.
- The head starts with zero weights, and the server repeats active bias probes. This is a valid diagnostic but still favorable to TANGO.
- Baselines are minimal. "SHARD not applicable" is true, but the paper should also compare against non-inversion aggregate baselines: single-round neutral update, random/public class prototype priors, label-count-only leakage, and possibly passive multi-round terminal updates without designed bias probes.
- The code enforces `S_total = sum_c S_c` with a weighted consistency row even in the one-round case. That is mathematically legitimate, but the paper should explain that one round remains rank deficient because this constraint alone does not resolve absolute class sums.
- The benchmark recovers hidden feature prototypes, not raw inputs. Any claim about input privacy requires a decoder, feature-inversion step, or a clear argument that hidden prototypes themselves are sensitive.

These limitations do not invalidate the Round 07 conclusion. They define the work needed before this becomes a paper-ready claim.

## Major Issues

1. **Major: TANGO is not individual inversion.**

The best individual MSE is essentially the within-class variance floor. The paper should not imply that terminal aggregate updates recover private samples unless additional side information is introduced and tested.

2. **Major: active repeated probing needs protocol justification.**

Choosing classifier biases is described as an ordinary server initialization choice. That is plausible, but repeated probes over the same client distribution with zero initial head weights should be described as an active attack surface. Round 08 must distinguish passive terminal observation from active terminal probing.

3. **Major: the benchmark is too favorable to be final evidence.**

Full-batch local training, linear head, small local step count, balanced classes, known client size, and synthetic separated hidden features make the moment system unusually clean. Round 08 needs stress tests: minibatch local SGD, nonzero head weights, class imbalance, more classes, larger feature dimensions, noisier/less separated classes, more local steps, and multiple learning rates.

4. **Major: the baseline suite is incomplete.**

SHARD's non-applicability is not a performance baseline. TANGO should be compared to single-round neutral update, passive non-designed probes, public/statistical prototype priors, label-count-only leakage, and oracle aggregate moments as an upper bound.

5. **Major: raw-input privacy is not yet demonstrated.**

Recovering class prototypes in a hidden feature space may be privacy-relevant, but the contribution needs either semantic examples, a decoder experiment, or a clear statement that the attacked object is the feature representation.

6. **Minor: count recovery is approximate and should be characterized.**

Count MAE stays small but not zero, and it changes with the number of probes. Round 08 should report when counts are assumed known, estimated, or rounded/projected to integer feasible counts.

## Rubric Scores

- Novelty: **4/5**. The pivot from incidence recovery to terminal aggregate moment inversion is a meaningful change of attack surface and cleaner than the previous Stage-2 variants.
- Soundness: **4/5 for the toy linear-head setting; 3/5 as a general privacy attack**. The derivation and code align, but the setting is narrow.
- Fairness: **3/5**. The same-observation claim against SHARD is fair, but the broader baseline suite is still too thin.
- Threat-model fit: **4/5 for weakening intermediate-gradient assumptions; 2/5 for individual recovery**.
- Impact: **3/5 currently; 4/5 if Round 08 adds theorem, stress tests, and tiered framing**.
- Evaluability: **4/5**. The artifacts are deterministic, readable, and internally consistent.

## Required Round 08 Instructions

Round 08 should not choose between "TANGO only" and "declare impossibility" as mutually exclusive paths. The right final direction is:

**Refine TANGO into a theorem-backed terminal aggregate leakage result, then integrate it with GARD/JOLI/JASPER as a tiered attack/limits framework.**

Concrete deliverables:

1. **Formalize the TANGO theorem.**

State assumptions exactly: linear softmax head, initial weights, known architecture, public hyperparameters, known or estimated client sample count, repeated server-chosen bias probes, and terminal updates. Prove class-sum/count identifiability conditions and rank requirements. Prove non-identifiability of individual samples under within-class zero-sum perturbations.

2. **Separate passive and active variants.**

Run and report:

- passive one-round terminal update;
- passive multi-round natural server states if available;
- active designed bias probes;
- oracle first-gradient aggregate as an upper bound.

3. **Stress-test TANGO.**

Add experiments for minibatch SGD, nonzero initial head weights, more local steps, more classes, class imbalance, larger dimensions, higher within-class variance, less class separation, and learning-rate sensitivity.

4. **Add fair aggregate baselines.**

Compare against neutral single-round estimates, public/statistical prototype priors, label-count-only leakage, passive terminal updates without designed probes, and oracle class-sum moments. Keep SHARD/GARD/JOLI/JASPER as "requires stronger observation" tiers rather than direct competitors under terminal-only observations.

5. **Clarify privacy endpoint.**

Either demonstrate that hidden prototypes are semantically meaningful, add a feature-to-input inversion/decoder experiment, or explicitly define the contribution as hidden-representation aggregate leakage.

6. **Build the tiered final narrative.**

The final paper should present:

- **Tier 0:** terminal-only passive leakage, mostly weak/rank-limited.
- **Tier 1:** TANGO active terminal-probe aggregate leakage, positive for prototypes/counts/means.
- **Tier 2:** GARD/JOLI conditional snapshot improvement when partial batch aggregates or priors exist.
- **Tier 3:** SHARD-style individual recovery only under strong intermediate-gradient/incidence assumptions.
- **Limit:** individual samples are not identifiable from TANGO aggregates alone.

## Bottom Line

TANGO meaningfully mitigates SHARD's too-strong intermediate-gradient assumption, but only by weakening the privacy target from individual sample recovery to aggregate/prototype leakage. That is a legitimate and more honest contribution than another hidden-incidence optimizer, provided Round 08 turns it into a theorem-backed, stress-tested, tiered attack/limits story. The final round should refine TANGO, integrate it with the stronger-observation GARD/JOLI/JASPER line as conditional tiers, and explicitly declare the impossibility of individual recovery from aggregate terminal moments alone.
