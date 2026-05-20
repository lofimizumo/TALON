# Round 08 -- Final Researcher Proposal: TALON, a Tiered Leakage and Limits Framework

## Final Candidate Thesis

The strongest paper-worthy direction is not a single SHARD replacement that recovers individual samples from weaker observations. The final direction is **TALON: Terminal Aggregate Leakage with Observation tiers and Non-identifiability limits**.

TALON reframes the project as a tiered privacy attack and limits theorem:

- **Strong observation:** SHARD, GARD, and JOLI can recover or polish individual-level objects only when the attacker has strong intermediate information: batch-average rows, near-correct incidence, oracle or high-quality graph priors, or per-sample snapshots.
- **Partial observation:** GARD is a conditional nullspace prior. It can reduce the number of observed Stage-2 batch-gradient rows when batch incidence is essentially correct and the graph is aligned, but it does not solve unknown assignment.
- **Terminal-only observation:** TANGO recovers class sums, class counts, dataset means, and class prototypes from repeated terminal client updates under active server bias probes. It removes SHARD's intermediate-gradient stream but weakens the privacy target from individual samples to aggregate/prototype leakage.
- **Unknown assignment:** JASPER gives a negative result: without additional assignment information, many hidden memberships explain the same batch averages, and joint soft assignment does not identify individual snapshots.

The paper claim should therefore be:

> Terminal model updates can leak statistically precise class-level prototypes without exposing any intermediate minibatch gradients, but individual samples are not identifiable from terminal aggregate moments alone. Stronger individual recovery requires stronger observations or side information, and unknown assignment creates an identifiability barrier for SHARD-style Stage 2.

This is a significantly better scientific direction than claiming SHARD-equivalent individual recovery under a weaker threat model. It directly addresses the user's priority of reducing the intermediate-gradient assumption while avoiding the false claim that terminal-only observations identify individuals.

## Method: TALON Tiers

### Tier 0: Passive Terminal Observation

One neutral terminal update is generally rank deficient for absolute class sums. The Round-07 and Round-08 benchmarks show this clearly: in the balanced setting, one neutral terminal round gives prototype MSE `0.143141`.

This tier is valuable as a baseline and as a warning. A single terminal delta may leak some centered or contrastive aggregate information, but it should not be sold as a complete inversion channel.

### Tier 1: Active Terminal Aggregate Leakage (TANGO)

TANGO assumes the server can choose ordinary initial classifier biases across repeated rounds for the same client or client distribution and observe only terminal model deltas. It does not observe local minibatch gradients, batch order, incidence, batch averages, or per-sample snapshots.

For a linear softmax head with zero initial weights and initial bias \(b^{(r)}\), let
\[
p^{(r)}=\operatorname{softmax}(b^{(r)}), \quad
S_c=\sum_{i:y_i=c}x_i,\quad S=\sum_c S_c.
\]
The first-order full-batch weight gradient for class \(c\) satisfies
\[
N g_{r,c}=p^{(r)}_c S-S_c.
\]
Across at least two linearly independent bias probes, the attacker solves a linear moment system for \(S\) and \(S_c\). Bias updates similarly estimate class counts \(n_c\), giving prototypes \(\mu_c=S_c/n_c\).

TANGO's target is aggregate leakage: class sums, counts, prototypes, and dataset means. These are privacy-relevant when the hidden representation is sensitive or can be decoded, but they are not individual reconstructions.

### Tier 2: Conditional Partial-Observation Stage 2 (GARD)

GARD remains useful as a conditional Stage-2 module:
\[
\min_S \|H_\Omega S-M_\Omega\|_F^2
+\lambda \operatorname{Tr}(S^\top L S)
+\rho\left\|\frac{1}{N}\mathbf{1}^\top S-\bar{s}\right\|_2^2.
\]
Round 05 shows the best narrow result: with oracle assignment and an aligned graph, GARD reaches target snapshot MSE with `15/60` observed rows while SHARD-style LS needs `48/60`, a 75% reduction. But under wrong or unknown assignments, no tested method reaches the target. Thus GARD is a prior for missing-row nullspaces, not a full attack under unknown incidence.

### Tier 3: Strong-Observation Individual Recovery (SHARD/JOLI)

SHARD-style individual recovery remains a strong-observation tier. It requires enough intermediate batch-gradient rows and sufficient incidence information to identify per-sample snapshots. JOLI is a Stage-3 polish that improves input fidelity in compressive snapshot inversion, but it is not a new leakage channel and does not reduce the Stage-2 threat model.

## Theorem and Claim Sketches

### Theorem 1: Terminal Aggregate Identifiability

Assume:

1. A client trains a linear softmax head on fixed hidden features \(x_i\) and labels \(y_i\).
2. The server observes terminal deltas from \(R\) rounds with known initial biases \(b^{(r)}\), known learning rate, local step count, and client sample count \(N\).
3. The first-order terminal approximation is valid, or the terminal update is sufficiently close to the first-order aggregate gradient.
4. The probe probability matrix \(p^{(r)}=\operatorname{softmax}(b^{(r)})\) yields a full-rank linear system after adding the consistency constraint \(S=\sum_c S_c\).

Then class sums \(S_c\), total sum \(S\), and class counts \(n_c\) are identifiable from terminal aggregate updates. Consequently the empirical class prototypes \(\mu_c=S_c/n_c\) and dataset mean \(S/N\) are identifiable.

Proof sketch: for each feature coordinate, stack the linear equations \(N g_{r,c}=p^{(r)}_c S-S_c\) over rounds and classes. Full rank of the resulting design identifies the unknown vector \((S,S_1,\ldots,S_C)\). Bias-gradient equations identify counts by \(N g^b_{r,c}=N p^{(r)}_c-n_c\). Multi-step terminal updates introduce approximation error controlled by step size, local steps, and head nonlinearity.

### Theorem 2: Individual Non-Identifiability From Aggregate Moments

Under the same aggregate observation model, individual samples are not identifiable from class sums and counts alone. For any class \(c\), any set of perturbations \(\delta_i\) satisfying
\[
\sum_{i:y_i=c}\delta_i=0
\]
produces a different individual dataset with the same class sum, class count, total sum, and first-order terminal aggregate updates.

Proof sketch: the observation map depends on \(\{x_i\}\) only through \(S_c\) and \(n_c\). Within-class zero-sum perturbations leave all observed moments invariant. Therefore the equivalence class has dimension at least \(\sum_c (n_c-1)d\), so individual recovery is impossible without additional priors, labels/metadata, multiple nonlinear feature maps, known candidate sets, or other side information.

### Claim 3: Unknown Assignment Barrier

For Stage-2 batch-average observations with hidden incidence, observed row means alone do not identify individual membership in general. Round 06 empirically supports this: under unknown within-epoch assignment, JASPER's hard overlap remains near random and snapshot MSE stays high even when its own soft-assignment residual is low. This should be formalized as a mixture non-identifiability result: different incidence matrices and snapshot sets can induce the same observed batch-average multiset.

## Round 08 Stress Test

Implemented in `code/benchmark_round08.py`.

The benchmark extends Round 07 with six scenarios:

- `balanced_clean`
- `imbalanced_clean`
- `noisy_terminal_1e-3`
- `weak_bias_poor_condition`
- `more_local_steps`
- `high_within_class_variance`

It preserves the terminal-only observation model. For all scenarios, observed intermediate batch gradients remain zero.

### Numeric Headline

Mean over 8 seeds:

| Scenario | Best active rounds | Prototype MSE | One-round prototype MSE | Active gain | Individual MSE | Within-class floor |
|---|---:|---:|---:|---:|---:|---:|
| balanced clean | 8 | `0.0001506` | `0.1431415` | `950.6x` | `0.266669` | `0.253177` |
| imbalanced clean | 8 | `0.0003817` | `0.2792532` | `731.5x` | `0.267185` | `0.252630` |
| terminal noise 1e-3 | 8 | `0.0002822` | `0.1435491` | `508.6x` | `0.266736` | `0.253177` |
| weak bias / poor condition | 4 | `0.0003295` | `0.1431415` | `434.4x` | `0.266801` | `0.253177` |
| more local steps | 8 | `0.0019079` | `0.1444444` | `75.7x` | `0.277133` | `0.253177` |
| high within-class variance | 8 | `0.0003575` | `0.1647381` | `460.8x` | `0.982833` | `0.836947` |

The result strengthens the Round-07 conclusion:

- Multiple active terminal probes remain strong for prototype leakage under imbalance, small terminal noise, poor probe conditioning, more local steps, and higher within-class variance.
- Individual reconstruction remains close to the within-class floor and worsens when within-class variance increases.
- Better prototype recovery does not imply individual recovery.

## Comparison to SHARD

Under the same terminal-only observations, SHARD Stage 2 is not directly applicable because there is no stream of batch-average rows and no incidence matrix to solve against. The fair comparison is not "TANGO beats SHARD at per-sample inversion." The fair claim is:

> TANGO leaks an identifiable aggregate privacy target from terminal observations where SHARD's Stage-2 individual-recovery machinery has no input.

For individual recovery, SHARD still belongs in the stronger-observation tier. TALON should present this as an observation-target trade-off rather than a single leaderboard.

## Paper Structure

1. **Introduction:** SHARD's main weakness is its reliance on intermediate local gradients and hidden incidence. The paper asks what privacy targets remain identifiable as the observation model weakens.
2. **Observation tiers:** define strong, partial, terminal-only, and unknown-assignment settings.
3. **Terminal aggregate theory:** prove TANGO aggregate identifiability and individual non-identifiability.
4. **Algorithms:** give TANGO linear estimator, GARD conditional MAP, and JASPER negative diagnostic.
5. **Experiments:** show TANGO stress tests, GARD conditional row reduction, JASPER assignment failure, and JOLI Stage-3 background.
6. **Limits and ethics:** emphasize that terminal prototypes can be sensitive but are not sample-level reconstructions.

## Final Viability Verdict

**Accept as a final paper direction if the paper is framed as tiered leakage and identifiability limits.**

Do not claim terminal-only individual recovery. The paper-worthy contribution is stronger and more honest:

- TANGO materially relaxes SHARD's intermediate-gradient threat model for aggregate/prototype leakage.
- GARD explains when partial intermediate observations can fill missing nullspaces.
- JASPER supplies the negative assignment barrier.
- JOLI remains a Stage-3 polish background result.
- The combined framework tells readers what is identifiable, under which observations, and where SHARD-style individual recovery fundamentally requires stronger information.

## Artifacts

- `code/benchmark_round08.py`
- `artifacts/round08_metrics.json`
- `artifacts/round08_tango_stress.svg`
- `logs/experiment_round08.log`
- `tutorial/tutorial.md`
