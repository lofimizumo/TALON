# Tutorial: TALON, Tiered Leakage and Limits Beyond SHARD

## 1. Why the Project Changed

The original SHARD-style goal was individual reconstruction from gradient-derived intermediate information. Rounds 04-07 showed that this is not one problem; it is a ladder of observation models with different identifiable privacy targets.

The final project direction is **TALON: Terminal Aggregate Leakage with Observation tiers and Non-identifiability limits**.

TALON's central message is:

> As the attacker observes less, the recoverable target changes. Terminal updates can identify class-level aggregate moments under active probes, but individual samples require stronger observations or additional priors.

This is a better paper direction than claiming that terminal-only observations recover individuals. It reduces SHARD's strongest assumption for a meaningful aggregate privacy target while proving why that weaker observation model cannot support SHARD-equivalent sample recovery.

## 2. Observation Tiers

| Tier | Observation model | Main result | Individual recovery? |
|---|---|---|---|
| Tier 0 | One passive terminal update | Rank-limited aggregate leakage | No |
| Tier 1 | Active terminal bias probes | TANGO recovers class sums, counts, prototypes, and dataset mean | No, not from aggregates alone |
| Tier 2 | Partial intermediate batch rows with good incidence/graph | GARD fills missing-row nullspaces conditionally | Conditional |
| Tier 3 | Strong SHARD-style intermediate gradients and incidence | SHARD/JOLI can recover or polish individual-level objects | Yes, under strong assumptions |
| Limit | Unknown assignment without extra side information | JASPER negative result / identifiability barrier | No robust evidence |

## 3. TANGO in One Page

**TANGO** means **Terminal Aggregate Neural Gradient Observation**.

Threat model:

- The server observes terminal client model deltas across repeated rounds.
- The server knows the architecture, learning rate, local step count, and initial model state.
- The server may choose ordinary initial classifier biases as active probes.
- The server does not observe intermediate minibatch gradients, batch order, batch incidence, batch-average snapshots, or per-sample snapshots.

For a linear softmax head initialized with zero weights and bias \(b^{(r)}\), define
\[
p^{(r)}=\operatorname{softmax}(b^{(r)}).
\]
For hidden features \(x_i\), labels \(y_i\), class sums \(S_c=\sum_{i:y_i=c}x_i\), and total sum \(S=\sum_c S_c\), the first-order full-batch gradient satisfies
\[
N g_{r,c}=p^{(r)}_c S-S_c.
\]

With one neutral round, this system is rank deficient. With multiple distinct bias probes, stack the equations over rounds and classes. If the design is full rank after the consistency constraint \(S=\sum_c S_c\), solve for \(S\) and \(S_c\). Bias deltas estimate counts \(n_c\), and prototypes follow:
\[
\mu_c=\frac{S_c}{n_c}.
\]

The privacy endpoint is therefore **class-level hidden-representation leakage**, not per-sample inversion.

## 4. The Key Theorems

### Terminal Aggregate Identifiability

Under a linear softmax head, known public hyperparameters, known initial biases, repeated terminal updates, and a full-rank probe design, terminal aggregate updates identify:

- class sums \(S_c\);
- class counts \(n_c\);
- class prototypes \(\mu_c\);
- dataset mean \(S/N\).

Proof idea: each feature coordinate gives a linear system in \((S,S_1,\ldots,S_C)\). Distinct probes make the coefficient matrix full rank. Bias updates provide count equations.

### Individual Non-Identifiability

Terminal aggregate moments do not identify individual samples. For any class \(c\), perturb samples by \(\delta_i\) such that
\[
\sum_{i:y_i=c}\delta_i=0.
\]
The class sum, count, prototype, dataset mean, and first-order terminal aggregate updates are unchanged. Therefore infinitely many individual datasets produce the same terminal observations.

This is why individual MSE saturates near the within-class variance floor in Rounds 07-08.

## 5. What the Rounds Established

### JOLI

JOLI is a Stage-3 polish for compressive snapshot inversion. It improves image fidelity from oracle snapshots, but it does not reduce SHARD's Stage-2 observation requirement. It should be background, not the headline.

### GARD

GARD adds a graph/Laplacian prior to Stage-2 disaggregation. It can reduce observed batch-gradient rows when:

- batch incidence is correct or nearly correct;
- the incidence design is rank deficient because rows are missing;
- the graph is aligned with true snapshot geometry;
- regularization is selected without using snapshot MSE.

The strongest narrow result from Round 05 is a 75% row reduction under oracle assignment and an aligned graph: `15/60` rows for GARD versus `48/60` for SHARD-style LS at the target MSE. But GARD fails under unknown assignment, so it is a conditional prior.

### JASPER

JASPER was the joint soft-assignment attempt. It is useful mainly as a negative result: when within-epoch assignment is unknown, soft assignment can fit its own residual while failing to recover true membership or snapshots. This supports an identifiability barrier for SHARD-style Stage 2 without additional assignment information.

### TANGO

TANGO is the main threat-model reduction. It removes intermediate minibatch gradients entirely but changes the target to aggregate/prototype leakage.

## 6. Round 08 Results

Run:

```bash
cd "/Users/yetao/Documents/06.My Papers/CQU/2026/06.TALON"
.venv/bin/python code/benchmark_round08.py
```

Outputs:

- `artifacts/round08_metrics.json`
- `artifacts/round08_tango_stress.svg`
- `logs/experiment_round08.log`

Headline over 8 seeds:

| Scenario | Best active rounds | Prototype MSE | One-round prototype MSE | Active gain | Individual MSE | Within-class floor |
|---|---:|---:|---:|---:|---:|---:|
| balanced clean | 8 | `0.0001506` | `0.1431415` | `950.6x` | `0.266669` | `0.253177` |
| imbalanced clean | 8 | `0.0003817` | `0.2792532` | `731.5x` | `0.267185` | `0.252630` |
| terminal noise 1e-3 | 8 | `0.0002822` | `0.1435491` | `508.6x` | `0.266736` | `0.253177` |
| weak bias / poor condition | 4 | `0.0003295` | `0.1431415` | `434.4x` | `0.266801` | `0.253177` |
| more local steps | 8 | `0.0019079` | `0.1444444` | `75.7x` | `0.277133` | `0.253177` |
| high within-class variance | 8 | `0.0003575` | `0.1647381` | `460.8x` | `0.982833` | `0.836947` |

Takeaway: active terminal probes robustly recover prototypes, but individual reconstruction remains bounded by within-class variance.

## 7. How to Write the Paper

Recommended title shape:

> Beyond SHARD: Tiered Gradient Leakage and Identifiability Limits Under Weaker Federated Observations

Recommended structure:

1. **Motivation:** SHARD's intermediate-gradient assumption is too strong for many FL deployments.
2. **Observation ladder:** define terminal-only, partial intermediate, and strong intermediate settings.
3. **TANGO theory:** prove aggregate identifiability and individual non-identifiability.
4. **Algorithms:** present TANGO, GARD, JASPER diagnostic, and JOLI as background.
5. **Experiments:** show Round-08 TANGO stress tests, Round-05 GARD conditional gains, Round-06 JASPER negative assignment results, and Round-03 JOLI context.
6. **Limits:** make clear that terminal-only attacks recover aggregate hidden prototypes, not individual samples.

## 8. Claim Discipline

Safe claims:

- Terminal active probes can recover hidden class prototypes without observing intermediate minibatch gradients.
- Prototype leakage is identifiable under rank conditions.
- Individual samples are not identifiable from class aggregate moments alone.
- GARD reduces missing-row burden only under correct incidence and useful graph priors.
- Unknown assignment remains a barrier for SHARD-style individual recovery.

Unsafe claims:

- Terminal-only observations recover individual samples.
- TANGO directly beats SHARD at SHARD's own individual-reconstruction target.
- GARD solves missing intermediate gradients in the full unknown-assignment setting.
- JOLI is a new leakage channel.

## 9. Code Map

| File | Role |
|---|---|
| `code/benchmark_round08.py` | Final TANGO stress benchmark |
| `artifacts/round08_metrics.json` | Final stress metrics |
| `artifacts/round08_tango_stress.svg` | Final stress plot |
| `rounds/round_08/researcher_proposal.md` | Final integrated proposal |
| `rounds/round_08/revision_log.md` | Round-08 change log |
| `code/benchmark_round07.py` | Original TANGO proof of concept |
| `code/benchmark_round06.py` | JASPER assignment negative result |
| `code/benchmark_round05.py` | GARD stress test |
| `code/benchmark_round03.py` | JOLI Stage-3 polish background |
