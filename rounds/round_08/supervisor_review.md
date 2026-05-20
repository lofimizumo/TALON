# Round 08 Supervisor Review

## Final Verdict

**ACCEPT AS A FINAL PAPER DIRECTION, BUT NOT AS A SHARD-EQUIVALENT ATTACK.**

Round 08 makes the correct final pivot. TALON/TANGO is not a better SHARD method for recovering individual samples from weaker observations. It is a different and weaker leakage contribution: terminal model updates, under active server bias probes, can reveal class-level hidden-feature sums, counts, prototypes, and dataset means without observing intermediate minibatch gradients. That is a real mitigation of SHARD's too-strong Stage-2 observation requirement, but only because the recovered privacy target is changed from individual samples to aggregate/prototype statistics.

The final project is scientifically viable if framed as:

> A tiered leakage and identifiability paper showing that active terminal probes identify aggregate class moments, while individual recovery remains non-identifiable without stronger observations or side information.

It is not viable if framed as:

> A method that significantly outperforms SHARD at SHARD's original individual-reconstruction target under a weaker terminal-only threat model.

## Audit Scope

I audited:

- `rounds/round_08/researcher_proposal.md`
- `rounds/round_08/revision_log.md`
- `code/benchmark_round08.py`
- `artifacts/round08_metrics.json`
- `logs/experiment_round08.log`
- `tutorial/tutorial.md`
- `rounds/round_07/supervisor_review.md`

I did not rerun the benchmark. This review checks internal consistency, claim discipline, threat-model fit, theorem plausibility, and whether the saved metrics honestly support the proposed thesis.

## Executive Assessment Against the Corrected Goal

The user's corrected goal was not "improve Stage 3 polish." It was to find a direction significantly better than SHARD by prioritizing Stage 2 or the full method, mitigating the intermediate-gradient requirement, and verifying the ideas honestly.

Round 08 partially satisfies this goal:

- **Stage-3 overfocus has been fixed.** JOLI is now correctly demoted to background: a Stage-3 polish under oracle snapshots, not a new leakage channel.
- **Stage-2/full-method pressure is addressed honestly.** GARD is retained only as a conditional partial-observation Stage-2 prior. JASPER is retained as a negative assignment result. TANGO becomes the main positive result under a weaker observation model.
- **The too-strong intermediate-gradient requirement is mitigated for aggregate leakage.** TANGO uses terminal deltas and designed classifier-bias probes, not minibatch gradients, batch rows, batch order, incidence, or per-sample snapshots.
- **The original SHARD-level individual target is not achieved.** Terminal-only TANGO does not recover individuals; the experiments and theorem sketch correctly explain why.

Therefore the final project is better than a narrow SHARD Stage-3 tweak, but it is not "better than SHARD" in the same task. It is better as a paper direction because it maps the boundary between what weaker terminal observations can and cannot identify.

## TALON/TANGO: Better Method or Different Leakage?

TALON is a better framework than the previous round-by-round attempts because it stops pretending there is one universal inversion target. It gives a useful observation ladder:

- Tier 0: passive terminal observation, weak/rank-limited aggregate leakage.
- Tier 1: active terminal probing, TANGO aggregate/prototype leakage.
- Tier 2: partial intermediate observations, GARD conditional nullspace filling.
- Tier 3: strong intermediate observations, SHARD/JOLI individual recovery or polish.
- Limit: unknown assignment and terminal aggregate observations do not identify individuals.

TANGO itself is **not** a better SHARD method. It is a different leakage contribution with a weaker target. Under the same terminal-only observations, SHARD Stage 2 is not applicable because there are no batch-average rows or incidence constraints to solve. TANGO's fair claim is not "beats SHARD"; it is "extracts a meaningful aggregate privacy target in an observation regime where SHARD's individual-recovery machinery has no input."

This is still a defensible contribution. Hidden class prototypes can be privacy-relevant, especially if hidden features are semantically meaningful or decodable. But the paper must call the endpoint aggregate hidden-representation leakage unless it adds a decoder or raw-input semantic evidence.

## Threat Model: Honest-But-Curious or Stronger?

TANGO is **not passive honest-but-curious**. It requires active terminal probing:

- the server chooses classifier biases;
- the server repeats probes over the same client or comparable client distribution;
- the server knows optimizer hyperparameters, architecture, local step count, initial model state, and client sample count or can estimate it;
- the server observes terminal client deltas from each probe.

This can fit an **active, protocol-compliant honest-but-curious server** if the FL protocol normally lets the server choose global initialization or round models and clients simply train the supplied model. It does require stronger server control than passive observation of ordinary training. It is still weaker than SHARD's intermediate-gradient assumption because it does not require seeing local minibatch gradients or batch incidence.

The paper should use precise wording:

- Safe: "active server-side terminal probing through chosen classifier biases."
- Safe: "weaker than intermediate-gradient observation, stronger than passive terminal logging."
- Unsafe: "standard honest-but-curious server passively observes terminal updates and recovers prototypes."

## Theorem and Identifiability Claims

The main theorem claims are reasonable under the stated narrow model:

- linear softmax head;
- zero or controlled initial head weights;
- fixed hidden features;
- known labels or known class-output semantics;
- known learning rate, local step count, and client sample count;
- terminal deltas close to first-order full-batch gradients;
- enough linearly independent bias probes for a full-rank linear system.

Under those assumptions, the equation

\[
N g_{r,c}=p^{(r)}_c S-S_c
\]

does identify class sums coordinate-wise once the probe design is full rank with the consistency constraint. Bias gradients can estimate counts. The non-identifiability theorem for individuals is also sound: any within-class zero-sum perturbation preserves class sums, counts, dataset mean, and first-order terminal observations.

The paper must not overextend these theorems. Exact identifiability is a first-order/full-batch/linear-head statement. Multi-step local training, nonzero head weights, minibatch SGD, nonlinear backbones that update during local training, unknown labels, client drift, and secure aggregation all require either approximation bounds or separate experiments. Round 08 includes a "more local steps" stress test, which is helpful, but it does not prove general nonlinear FL identifiability.

## Experiment Honesty and Reproducibility

The Round 08 experiments are honest for the stated toy setting.

The code records `observed_intermediate_batch_gradients = 0.0`, uses terminal weight and bias deltas, and reports both the positive aggregate result and the negative individual result. The metrics and log agree on the main headline:

- balanced clean active probes: prototype MSE `0.0001506`;
- balanced one-round neutral baseline: prototype MSE `0.1431415`;
- active gain: about `950.6x`;
- balanced individual MSE from repeated prototypes: `0.266669`;
- balanced within-class floor: `0.253177`.

The stress tests strengthen the claim modestly:

- imbalance remains strong: prototype MSE `0.0003817`;
- terminal noise `1e-3` remains strong: `0.0002822`;
- weak bias / poorer conditioning remains usable: `0.0003295`;
- more local steps degrades but still works: `0.0019079`;
- high within-class variance preserves prototype recovery but makes individual recovery much worse: individual MSE `0.982833` versus floor `0.836947`.

These are the right qualitative outcomes. Prototype recovery improves dramatically under active probes, while individual recovery stays near the within-class floor. That verifies the central idea honestly: the attack identifies aggregate moments, not individuals.

Reproducibility is good at artifact level. The benchmark is deterministic, uses fixed seeds, writes JSON and logs, and the saved metrics match the proposal and tutorial headlines. The code is readable and self-contained.

Important limitations remain:

- no minibatch SGD stress test;
- no nonzero initial head-weight stress test;
- only 3 classes and 10-dimensional synthetic features;
- no larger class count or larger representation dimension;
- no passive natural multi-round baseline without designed probes;
- no public/statistical prototype prior baseline;
- no oracle aggregate upper-bound baseline separated from TANGO;
- no raw-input or decoder evidence that hidden prototypes are semantically revealing;
- count recovery is approximate and should be reported separately from prototype recovery;
- the tutorial currently contains the new TALON tutorial followed by stale older JOLI tutorial material, which should be cleaned before paper drafting.

These limitations do not invalidate the final thesis, but they constrain how strong the paper can sound.

## Relationship to SHARD, GARD, JASPER, and JOLI

SHARD remains the strong-observation individual-recovery tier. TALON should not claim to dominate it on sample-level reconstruction. Instead, TALON should say that SHARD-style recovery requires intermediate information that may not be present, and that terminal-only observations support a different aggregate target.

GARD is useful but conditional. The Round 05 row-reduction result under oracle assignment and aligned graph priors is relevant as a partial-observation Stage-2 module. It does not solve unknown assignment and should not be advertised as a general replacement for SHARD Stage 2.

JASPER is valuable mainly as negative evidence. Its low residual but poor membership recovery supports the unknown-assignment barrier. This should become a formal non-identifiability or ambiguity claim, not a failed-method section.

JOLI is background. It improves semantic-looking input recovery from oracle snapshots in a compressive Stage-3 setting, but it does not reduce the leakage requirement and should not be part of the headline contribution.

## Supportable Paper Claim

The exact claim supportable now is:

> Under a linear-head terminal-update model, an active server that chooses classifier-bias probes can recover class sums, class counts, dataset means, and class prototypes from terminal client updates without observing intermediate minibatch gradients. These aggregate moments do not identify individual samples; individual recovery requires stronger observations, side information, or priors. SHARD-style individual inversion therefore occupies a stronger-observation tier, while TANGO exposes a weaker but still meaningful aggregate leakage channel.

This is supported by the theorem sketch, code, saved metrics, and stress tests.

## Unsupported or Unsafe Claims

The following claims are not supported:

- TANGO recovers individual samples from terminal-only observations.
- TALON/TANGO is significantly better than SHARD at SHARD's original individual-reconstruction target.
- Active bias probing is passive honest-but-curious observation.
- GARD solves Stage 2 under unknown assignment.
- JOLI is a new leakage channel or a mitigation of SHARD's Stage-2 assumptions.
- Hidden-feature prototypes automatically imply raw-input leakage without decoder or semantic evidence.
- The exact terminal identifiability theorem holds unchanged for arbitrary nonlinear, multi-step, minibatch FL training.

## Required Paper Edits Before Drafting

Before turning this into a manuscript, make these changes:

1. Clean `tutorial/tutorial.md` so it does not append the obsolete JOLI-only tutorial after the new TALON tutorial.
2. Rename threat model language to "active terminal probing" throughout.
3. Separate exact theorem assumptions from approximate multi-step empirical behavior.
4. Add at least one passive multi-round or public-prior baseline if time permits.
5. State clearly that the privacy endpoint is hidden-representation aggregate leakage unless a decoder experiment is added.
6. Keep SHARD/GARD/JASPER/JOLI as observation tiers, not direct terminal-only competitors.

## Final Rubric

- **Novelty:** 4/5 as a tiered leakage and limits framework; 2/5 as a direct SHARD replacement.
- **Soundness:** 4/5 for the linear-head aggregate theorem and saved toy experiments; 3/5 for broad FL generality.
- **Threat-model improvement:** 4/5 for removing intermediate-gradient observation; 2/5 if described as passive honest-but-curious.
- **Stage-2/full-method relevance:** 3.5/5. It addresses the Stage-2 assumption by changing the observation-target pair, not by solving unknown individual disaggregation.
- **Experimental honesty:** 4/5. The metrics expose both success and failure modes.
- **Paper readiness:** 3.5/5. The thesis is now coherent, but the evaluation remains narrow.

## Bottom Line

Round 08 succeeds as the final correction of direction. The project should proceed as TALON: a tiered leakage and limits paper centered on TANGO's active terminal aggregate leakage and the non-identifiability of individuals from aggregate moments.

The concise supported thesis is:

> Active terminal probes can identify class-level hidden prototypes and aggregate moments without intermediate minibatch gradients, but terminal aggregate moments do not identify individual samples. SHARD-equivalent individual recovery requires stronger observations or side information.

