# Round 01 — Real VQC-stack MNIST reconstruction (researcher proposal)

## Problem & goal

Prior sibling run `qfl-terminal-snapshot` established **snapshot MSE** on encoded MNIST vectors under LASA-QTERM tiers, but not **end-to-end image reconstruction** after Level-3 inversion. This run (`qfl-vqc-mnist-recon`) must close the loop: real MNIST pixels → `SurrogateQFL` federated gradients → snapshot recovery (SHARD oracle, LASA-QTERM T1p/T1b) → `ShardAttacker.level3_invert` / `joli_invert` → **input MSE / PSNR** against acceptance targets in `config.json`.

## Hypothesis / claim

**H1:** With compressive `dim_g=100` on **8×8 MNIST** (`d=64`), the SHARD oracle snapshot path plus canonical L3 achieves input MSE ≤ 0.05 and PSNR ≥ 18 dB within 3 seeds.

**H2:** LASA-QTERM **T1b** (B=1 terminal rows) can approach oracle image quality when snapshot MSE stays below ~0.15; **T1p** will lag T1b at equal terminal-row budgets because partial rows omit most within-epoch structure.

*Speculation (not yet accepted):* JOLI TV polish helps weak snapshot paths more than the oracle path when `dim_g < d`.

## Related work

- **SHARD / SurrogateQFL:** vendored `vendor/shard_sim` — linear gradient–snapshot structure exploited by Stage 1–3 (`literature/vqc_stack_bridge.md`).
- **LASA-QTERM:** sibling `qfl-terminal-snapshot` Round 03–04 — production `QtermAttack` tiers T1p/T1b (`code/qterm_attack.py`).
- **JOLI:** parent `code/joli_invert.py` — SHARD-compatible L3 with optional TV in L-BFGS when compressive.
- **Prior MNIST L3 tuning:** parent `code/benchmark_round03.py` — Pareto on 28×28; informs later rounds, not Round-01 budget.

## Method

### Assumptions

1. **SurrogateQFL** is an adequate classical stand-in for LASA-VQC gradients (cosine RFF encode, random `A^(e)` per epoch).
2. **Honest attacker** knows `A^(e)`, batch structure, and tier-appropriate gradient rows; no label leakage.
3. **Matching:** Hungarian alignment for snapshot and image MSE/PSNR (per-seed, reported as mean over seeds).
4. **Round-01 resolution:** **8×8 MNIST** via `FederatedDataLoader(resize=8)` — documented departure from 28×28 preference in `config.json` for L3 tractability (6 path×inverter combos × 32 samples × 3 seeds).

### Pipeline (per seed)

1. Load `N=32`, `B=4`, `E=10` MNIST train subset; flatten to `[0,1]`.
2. `SurrogateQFL(input_dim=d, dim_g=100, n_params=100, noise=0.01)`.
3. Per epoch: `compute_batch_gradients` → `coeff_matrices`, `batch_gradients`; parallel **T1b** track with B=1 shuffled clients.
4. `level1_mean_recovery` → snapshot recovery:
   - **shard_oracle:** `level2_disaggregate` (full intermediate rows)
   - **lasa_qterm_T1p:** `QtermAttack(T1P, partial_rows=7)`
   - **lasa_qterm_T1b:** `QtermAttack(T1B)` on B=1 coefficients
5. L3 on recovered snapshots: **shard_l3** (`level3_invert`), **joli_l3** (`joli_invert`, default TV if compressive).
6. Metrics: snapshot MSE, input MSE, PSNR; artifact `artifacts/round01_recon_grid.png` (seed 7, four methods).

### Baselines & failure modes

| Failure mode | Signal |
|------------|--------|
| Snapshot collapse (T1p/T1b) | High snapshot MSE → blurred L3 regardless of inverter |
| Compressive null space | Input MSE plateau with good snapshot MSE on oracle |
| L3 local minima | Low snapshot residual but wrong image; low matching acc |
| 28×28 premature | OOM / multi-hour L3; deferred to Round 2 |

## Experiment plan

- **Script:** `code/benchmark_round01.py`
- **Seeds:** 3, 7, 11
- **Targets:** input MSE ≤ 0.05, PSNR ≥ 18 dB (`config.json`)
- **Outputs:** `artifacts/round01_metrics.json`, `round01_recon_grid.png`, `logs/experiment_round01.log`

## Changes this round

Initial round — establish runnable E2E stack and honest gap table vs targets.

## Open risks

- 8×8 may **overstate** feasibility vs 28×28 (lower `d`, easier L3).
- T1b uses **320 terminal rows** vs **80 intermediate** for SHARD — budget asymmetry; report both fairly.
- Single-machine CPU/MPS runtime may force reduced L3 budget in Round 2 if targets missed (not hidden).
