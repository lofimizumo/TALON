# Round 02 — Full 28×28 MNIST reconstruction + dim_g sweep

## Problem & goal

Round 01 validated the E2E stack at **8×8** (`d=64`) but deferred the config-preferred **28×28** track. Round 02 runs the primary FederatedDataLoader resolution at `d=784`, fixes weak **SHARD L2** / **T1p** snapshot recovery, sweeps `dim_g ∈ {100, 160, 256}` for compressive JOLI benefit, and reports **per-seed pass rates** against acceptance targets.

## Hypothesis / claim

**H1:** Graph-anchored SHARD L2 (`max_iter=500`, `graph_term_map` init) improves oracle snapshot MSE vs Round-01 Gaussian init at 28×28.

**H2:** T1p with **8 terminal rows/epoch** (full last-epoch minibatch leak) + higher `partial_max_iter` narrows the T1p→L3 gap without T1b’s 320-row terminal flood.

**H3:** JOLI TV (`tv_lbfgs` when `dim_g < d`) beats shard L3 on weak paths in the compressive sweep; `dim_g=160` is the sweet spot between underfitting (100) and looser compression (256).

## Method

- **Data:** `FederatedDataLoader(resize=28)`, `N=32`, `B=4`, `E=10`, seeds 3/7/11.
- **Snapshot paths:** SHARD oracle (graph-init L2), LASA-QTERM T1p/T1b.
- **L3:** `level3_invert` and `joli_invert` (default compressive TV).
- **Grids:** oracle / T1p / T1b via JOLI at `dim_g=160`, seed 7.

## Experiment plan

- **Script:** `code/benchmark_round02.py`
- **Outputs:** `artifacts/round02_metrics.json`, `round02_recon_grid_dim{g}.png`, `logs/experiment_round02.log`

## Open risks

- 28×28 L3 is ~12× heavier per pixel than 8×8; dim_g triple-sweep multiplies runtime.
- T1b may still dominate means via B=1 exact disaggregation — reported separately from primary weak-path story (T1p).
