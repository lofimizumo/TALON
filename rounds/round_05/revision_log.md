# Round 05 Revision Log

## Supervisor weaknesses addressed

### 1. Move beyond oracle incidence

Implemented attacker-assumed incidence regimes in `code/benchmark_round05.py`:

- `oracle`: exact true incidence.
- `wrong20`: partially wrong assignment, roughly one wrong member per batch.
- `wrong40`: stronger partial assignment error, roughly two wrong members per batch.
- `unknown_random`: a proxy for unknown assignment where the attacker guesses random batch membership.

Observed batch means are generated from the true incidence, while the solver receives the attacker-assumed incidence. This directly tests whether GARD survives assignment mismatch rather than only missing rows.

Result: it mostly does not. At 40% observed rows, GARD with the oracle graph improves MSE from 0.4825 to 0.0236 under oracle assignment, but only to 0.3625 under `wrong20`, 0.8294 under `wrong40`, and 1.0060 under `unknown_random`.

### 2. Stress-test the graph prior

Added graph regimes:

- oracle chain graph;
- noisy chain graph;
- wrong permuted chain graph;
- graph inferred from observed batch co-occurrence;
- graph inferred from SHARD-style LS recovered snapshots via kNN.

Result: the oracle graph is decisive, the noisy graph is partly useful, and inferred/wrong graphs do not provide reliable gains. Under oracle assignment at 40% observed rows:

- oracle graph: 0.0236 MSE;
- noisy graph: 0.1765 MSE;
- wrong graph: 0.4825 MSE;
- co-occurrence graph: 0.5818 MSE;
- LS-kNN graph: 0.4514 MSE.

### 3. Add baselines

Added:

- SHARD-style partial-observation LS;
- validation-selected ridge LS;
- validation-selected low-rank PCA prior;
- wrong-graph MAP with both validation-selected lambda and forced \(\lambda=0.3\).

Result: ridge and low-rank baselines help somewhat under wrong assignment by damping unstable LS, but they do not solve the problem. At 40% rows and `wrong20`, ridge is 0.7803 MSE and low-rank PCA is 0.7811, while oracle-graph GARD is 0.3625. None reaches the target 0.10 MSE under assignment error.

### 4. Select regularization by validation observations

All ridge and graph regularization strengths are selected using held-out observed batch rows. Snapshot MSE is used only for final evaluation. The validation split is recorded per run in `artifacts/round05_metrics.json`.

This revealed an important limitation: validation residual under the same wrong incidence model does not reliably detect assignment error. For example, wrong-graph validation can select nonzero lambdas at very low observation budgets and worsen snapshot recovery.

### 5. Quantify threat-model reduction

Defined target snapshot MSE \(\le 0.10\).

Under oracle assignment:

- random-row observation:
  - SHARD-style LS: 48/60 rows;
  - ridge LS: 48/60 rows;
  - low-rank PCA: 48/60 rows;
  - GARD oracle graph: 15/60 rows;
  - GARD noisy graph: 36/60 rows.
- prefix-epoch observation:
  - SHARD-style LS: 4/5 epochs;
  - ridge LS: 4/5 epochs;
  - GARD oracle graph: 1/5 epoch;
  - GARD noisy graph: 3/5 epochs.

Under `wrong20`, `wrong40`, and `unknown_random`, no method reaches MSE \(\le 0.10\) at any tested random-row budget. Therefore the threat-model reduction is real only under nearly correct assignment and a good graph prior.

## Pivot / honest assessment

Round 05 does not justify claiming that GARD solves SHARD Stage 2. It supports a narrower claim:

> GARD is an effective conditional nullspace prior for missing Stage-2 observations when assignment is correct and the graph is aligned with snapshot geometry.

It also shows the main failure mode:

> Assignment error dominates graph smoothness. If batch membership is wrong or unknown, even an oracle graph cannot recover accurate snapshots in this benchmark.

The next research direction should be assignment-first: a joint soft-assignment and graph-regularized Stage-2 method, or a new validation criterion based on cross-epoch consistency rather than residuals under assumed incidence.

## Reproducibility notes

- Script: `code/benchmark_round05.py`
- Metrics: `artifacts/round05_metrics.json`
- Plots:
  - `artifacts/round05_random_oracle_assignment.png`
  - `artifacts/round05_assignment_stress.png`
- Log: `logs/experiment_round05.log`
- Runtime in saved metrics: 2.89 seconds on the local workspace virtualenv.

The first attempted run with system `python3` failed because `matplotlib` was unavailable. The final artifacts were regenerated with `.venv/bin/python`.
