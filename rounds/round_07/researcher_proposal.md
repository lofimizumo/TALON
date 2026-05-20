# Round 07 -- Researcher Proposal: TANGO Terminal-Update Aggregate Inversion

## Goal

Round 06 showed that recovering per-sample snapshots from hidden per-batch incidence is not a promising main path. Round 07 therefore pivots to a weaker and more realistic observation model: the honest-but-curious server observes only terminal client model updates over several server rounds, plus the public model state and optimizer hyperparameters. It does not observe intermediate minibatch gradients, batch order, per-batch averages, or a batch incidence matrix.

The proposed method is **TANGO: Terminal Aggregate Neural Gradient Observation**. TANGO does not try to reconstruct SHARD-style per-batch snapshots. It recovers class-level feature sums, dataset means, class counts, and prototypes from terminal updates produced by honest server-chosen initial classifier biases.

## Weaker Observation Model

SHARD's Stage-2 setting assumes access to a stream of batch-average snapshots or gradients and then attempts to solve an incidence/snapshot recovery problem. TANGO removes that stream entirely.

Attacker observes:

- initial global linear-head weights and biases for each server round;
- the terminal client model delta after local training;
- public optimizer hyperparameters and model architecture.

Attacker does not observe:

- intermediate local minibatch gradients;
- batch order or per-batch membership;
- per-batch averages;
- per-sample snapshots.

Active choice is limited to ordinary server-side model initialization: the server chooses different initial classifier bias vectors across rounds. The benchmark does not use malicious gradient manipulation, hidden protocol changes, or client-side instrumentation.

## Method

Consider a linear softmax head with zero initial weights and server-chosen initial bias \(b^{(r)}\) in round \(r\). Let \(p^{(r)}=\operatorname{softmax}(b^{(r)})\). For sample features \(x_i\), labels \(y_i\), class sums \(S_c=\sum_{i:y_i=c}x_i\), and total sum \(S=\sum_c S_c\), the first-order full-batch weight gradient satisfies

\[
N g_{r,c}=p^{(r)}_c S - S_c.
\]

With one neutral round, this system is rank deficient: many shifted class sums explain the same terminal update. With two or more distinct honest bias probes, TANGO solves the overdetermined linear system for \(S\) and \(S_c\). Bias updates similarly estimate class counts. Class prototypes are then \(\mu_c=S_c/n_c\).

The benchmark uses terminal deltas after three local full-batch steps, not an exposed first gradient, so the estimator is mildly misspecified. This is intentional: it tests whether the first-order aggregate signal survives a realistic terminal-update observation.

## Minimal Experiment

Implemented in `code/benchmark_round07.py`.

Synthetic setup:

- 3 classes, 8 samples per class, 10-dimensional hidden features.
- Samples are Gaussian perturbations around separated class prototypes.
- Local client training uses a softmax linear head for 3 full-batch steps.
- The attacker sees only terminal updates for 1, 2, 4, or 8 server rounds.
- Seeds: 3, 7, 11, 19, 23, 29, 31, 37.

Metrics:

- prototype MSE against empirical class means;
- dataset mean MSE;
- class-count MAE;
- terminal-update residual;
- permutation-invariant individual MSE when prototypes are repeated as individual reconstructions;
- within-class variance as the aggregate-observation lower bound for individual reconstruction.

SHARD comparison under the same observations is deliberately negative: with no batch-average rows and no incidence matrix, SHARD Stage 2 cannot be instantiated. The relevant comparison is therefore not "TANGO beats SHARD at per-sample recovery"; it is "TANGO recovers a weaker but identifiable privacy target under observations where SHARD has no Stage-2 input."

## Numeric Results

Mean over 8 seeds:

| Terminal rounds | Prototype MSE | Dataset mean MSE | Individual MSE from prototypes | Within-class variance |
|---:|---:|---:|---:|---:|
| 1 | 0.143141 | 0.143078 | 0.428464 | 0.253177 |
| 2 | 0.000274 | 0.000131 | 0.266743 | 0.253177 |
| 4 | 0.000188 | 0.0000719 | 0.266697 | 0.253177 |
| 8 | 0.000151 | 0.0000717 | 0.266669 | 0.253177 |

The single-round case is nearly uninformative about absolute class means because the linear system is rank deficient. Two terminal rounds are already enough to reduce prototype MSE by roughly 522x relative to the one-round baseline. Additional rounds improve conditioning slightly.

Individual recovery does not improve comparably. The best individual MSE is 0.2667, close to the within-class variance lower bound 0.2532. This is the expected result: class sums identify prototypes but not within-class sample deviations.

## Identifiability Characterization

For first-order terminal updates, TANGO's observations are invariant to any transformation that preserves each class sum and class count. Therefore individual samples are not identifiable. A server can recover class statistics, but every within-class zero-sum perturbation produces the same aggregate gradients.

The multi-step terminal setting breaks this invariance only weakly at the tested learning rate. Empirically, the update residual is tiny while individual recovery remains at the within-class variance floor. Low terminal-update residual should therefore not be interpreted as individual reconstruction success.

## Viability Verdict

**Viable as a weaker aggregate/prototype leakage contribution; not viable as an individual sample recovery replacement for SHARD.**

Round 07 provides a defensible pivot for the paper: instead of claiming that hidden per-batch incidence can be recovered from insufficient information, the attack should target identifiable aggregate privacy leakage from terminal model updates. The strongest claim is that an honest-but-curious server can actively choose ordinary initial model states across rounds and recover class prototypes or dataset statistics without observing any intermediate batch gradients.

This is not a full SHARD replacement if the paper requires individual sample reconstruction. It is a more credible alternative contribution if the paper reframes the target from per-sample inversion to aggregate/prototype leakage under a strictly weaker observation model.

## Artifacts

- `code/benchmark_round07.py`
- `artifacts/round07_metrics.json`
- `artifacts/round07_terminal_update_metrics.svg`
- `logs/experiment_round07.log`
