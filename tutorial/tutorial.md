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

Threat model (**active terminal probing**, not passive honest-but-curious):

- The server observes terminal client model deltas across repeated rounds.
- The server knows the architecture, learning rate, local step count, and initial model state.
- The server **actively chooses** initial classifier biases as probes across rounds.
- The server does not observe intermediate minibatch gradients, batch order, batch incidence, batch-average snapshots, or per-sample snapshots.

This is weaker than SHARD's intermediate-gradient assumption but stronger than passively logging a single neutral terminal update.

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

### Terminal Aggregate Identifiability (exact tier)

**Exact theorem assumptions:** linear softmax head, **zero initial head weights**, **full-batch** local training, first-order terminal deltas, known \(\eta,T,N\), known probe biases with full-rank design.

Under those assumptions, terminal aggregate updates identify:

- class sums \(S_c\);
- class counts \(n_c\);
- class prototypes \(\mu_c\);
- dataset mean \(S/N\).

Proof idea: each feature coordinate gives a linear system in \((S,S_1,\ldots,S_C)\). Distinct probes make the coefficient matrix full rank. Bias updates provide count equations.

**Approximate tier (empirical):** minibatch SGD, nonzero initial weights, and many local steps break or degrade the first-order estimator. Round 09 reports these separately from the exact theorem.

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

## 6. Round 09 Results (paper-readiness benchmark)

Run:

```bash
cd /workspace
python3 code/benchmark_round09.py
```

Outputs:

- `artifacts/round09_metrics.json`
- `artifacts/round09_baselines.svg`
- `artifacts/round09_count_vs_proto.svg`
- `logs/experiment_round09.log`

### Baselines on balanced clean (8 active/passive rounds, 8 seeds)

| Method | Prototype MSE | Count MAE | Individual MSE |
|---|---:|---:|---:|
| TANGO active | `0.000151` | `0.0419` | `0.2667` |
| Passive multi-round (neutral bias) | `0.143141` | `0.0152` | `0.4285` |
| Public statistical prior | `0.412694` | n/a | `0.6659` |
| Oracle aggregate (upper bound) | `0.0` | `0.0` | `0.2667` |

TANGO vs passive prototype gain: **950.6x**. Oracle aggregate shows the ceiling if moments were perfect; it is not a runnable attack.

### Extended stress (TANGO active)

| Scenario | Prototype MSE | Count MAE | Notes |
|---|---:|---:|---|
| balanced_clean | `0.000151` | `0.0419` | Reproduces Round 08 |
| imbalanced_clean | `0.000382` | `0.1107` | Count harder under imbalance |
| minibatch_sgd | `6.5877` | `2.5743` | **Fails** — first-order model breaks |
| nonzero_init_head | `0.009720` | `0.7428` | Prototype OK, counts degraded |
| large_10class_30dim | `0.000011` | `0.0294` | Scale-up succeeds |
| more_local_steps | `0.001908` | `0.1495` | Moderate approximation error |
| high_within_class_variance | `0.000358` | `0.0519` | Individual MSE `0.983` near floor |

Takeaways:

- Report **count recovery** and **prototype recovery** separately.
- Active probing remains essential in the exact regime; passive and public priors are much worse.
- Minibatch SGD is an honest negative result and must not be folded into the exact theorem.
- Individual reconstruction does not improve with better prototypes.

Round 08 stress results remain in `artifacts/round08_metrics.json` for reference.

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
| `code/benchmark_round09.py` | Paper-readiness benchmark with baselines and extended stress |
| `artifacts/round09_metrics.json` | Round-09 metrics (count + prototype separate) |
| `artifacts/round09_baselines.svg` | Method comparison plot |
| `artifacts/round09_count_vs_proto.svg` | Count MAE vs prototype MSE plot |
| `rounds/round_09/researcher_proposal.md` | Round-09 integrated proposal |
| `paper/outline.md` | Draft paper section skeleton |
| `code/benchmark_round08.py` | Round-08 TANGO stress benchmark |
| `code/benchmark_round07.py` | Original TANGO proof of concept |
| `code/benchmark_round06.py` | JASPER assignment negative result |
| `code/benchmark_round05.py` | GARD stress test |
| `code/benchmark_round03.py` | JOLI Stage-3 polish background |
