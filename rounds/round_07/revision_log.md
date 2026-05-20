# Round 07 Revision Log

## Supervisor-Driven Pivot

Round 06 received `REVISE_MAJOR / PIVOT_REQUIRED` because JASPER still depended on the hidden per-batch incidence bottleneck and failed under unknown within-epoch assignment. Round 07 therefore discards per-batch incidence recovery as the main route and tests a terminal-update-only attack surface.

## Changes Made

1. Defined **TANGO: Terminal Aggregate Neural Gradient Observation**.
   - Uses only aggregate terminal model deltas from honest server rounds.
   - Does not consume intermediate local gradients, batch rows, batch order, or incidence priors.
   - Uses server-chosen initial classifier biases as benign active probes.

2. Implemented a runnable benchmark in `code/benchmark_round07.py`.
   - Synthetic 3-class feature dataset with hidden individual samples.
   - Local client performs 3 full-batch softmax-training steps.
   - Server observes terminal deltas for 1, 2, 4, or 8 rounds.
   - Estimator solves a linear aggregate moment system for class sums and prototypes.

3. Added identifiability-aware metrics.
   - Prototype MSE.
   - Dataset mean MSE.
   - Class-count MAE.
   - Terminal-update residual.
   - Individual reconstruction MSE from repeated prototypes.
   - Within-class variance lower bound for individual recovery.

4. Added a same-observation SHARD comparison.
   - SHARD Stage 2 is marked not applicable under the same reduced observations because no batch-average row stream or incidence matrix exists.
   - The benchmark therefore evaluates whether a weaker leakage target is recoverable, not whether SHARD-style individual snapshots can be recovered from absent inputs.

5. Added dependency-free SVG plotting.
   - `artifacts/round07_terminal_update_metrics.svg` visualizes prototype MSE, individual MSE, and the within-class lower bound versus number of terminal rounds.

## Run Details

Command used:

```bash
"/Users/yetao/Documents/06.My Papers/CQU/2026/01.SHARD/source/experiments/vqc_pennylane/.venv/bin/python" "/Users/yetao/Documents/06.My Papers/CQU/2026/01.SHARD/research-artifacts/novel-inversion-vs-shard/code/benchmark_round07.py"
```

Runtime: approximately 0.19 seconds.

Outputs:

- `artifacts/round07_metrics.json`
- `artifacts/round07_terminal_update_metrics.svg`
- `logs/experiment_round07.log`

## Key Results

Mean over 8 seeds:

| Terminal rounds | Prototype MSE | Dataset mean MSE | Individual MSE | Within-class variance |
|---:|---:|---:|---:|---:|
| 1 | 0.143141 | 0.143078 | 0.428464 | 0.253177 |
| 2 | 0.000274 | 0.000131 | 0.266743 | 0.253177 |
| 4 | 0.000188 | 0.0000719 | 0.266697 | 0.253177 |
| 8 | 0.000151 | 0.0000717 | 0.266669 | 0.253177 |

Best headline at 8 terminal rounds:

- prototype MSE: 0.0001506;
- dataset mean MSE: 0.0000717;
- individual MSE from prototype reconstruction: 0.266669;
- within-class variance lower bound: 0.253177.

## Honest Assessment

TANGO succeeds on a weaker target: it recovers class prototypes and dataset means from terminal updates without intermediate batch gradients. This is a real threat-model reduction relative to SHARD's Stage-2 observation stream.

TANGO does not recover individual samples. The result is not merely an optimizer failure: first-order terminal updates depend on class sums and counts, so all within-class zero-sum perturbations are observationally equivalent. The defensible contribution is aggregate/prototype leakage under terminal-only observations, plus an impossibility argument against individual recovery from these aggregates alone.
