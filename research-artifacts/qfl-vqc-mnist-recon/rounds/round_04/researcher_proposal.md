# Round 04 — R3 JOLI L3 + improved snapshot paths

## Problem & goal

Round 02 showed **no seed** met `input_mse ≤ 0.05` and `psnr ≥ 18` on 28×28, though T1p + JOLI beat oracle on mean MSE. Round 04 combines the **parent Round-03 JOLI operating point** (`tv_lbfgs`) with stronger Stage-2 paths from the privacy-breakthrough GARD stack and terminal graph priors.

## Hypothesis / claim

**H1:** **GARD-SPARSE** with **oracle assignment** and T1p-aligned partial rows (8/epoch) lowers snapshot MSE vs partial SHARD alone, improving JOLI-R3 input fidelity.

**H2:** **T1p + graph prior polish** (`graph_term_map` blend before L3) narrows the snapshot→image gap without T1b’s row flood.

**H3:** If 28×28 still misses acceptance, **14×14** compressive fallback (`dim_g=100`) may hit ≥2/3 seed pass on a weak path.

## Method

| Component | Implementation |
|-----------|----------------|
| L3 | `joli_r3`: `joli_invert(..., tv_lbfgs=0.005)` from parent `artifacts/round03_metrics.json` |
| GARD-SPARSE | Oracle incidence + last 8 rows/epoch + chain Laplacian MAP (`benchmark_round04.py`) |
| T1p+graph | `QtermAttack` T1p → `graph_prior_polish` before L3 |
| Oracle upper bound | `ShardAttackerR02` graph-init L2 (Round 02) |
| Data | `FederatedDataLoader`, `N=32`, `B=4`, `E=10`, seeds 3/7/11 |
| Acceptance | ANY config: MSE ≤ 0.05, PSNR ≥ 18 on **≥2/3** seeds |

## Experiment plan

- **Script:** `code/benchmark_round04.py`
- **Outputs:** `artifacts/round04_metrics.json`, `logs/experiment_round04.log`, optional `round04_recon_grid_*.png`
- **No** `supervisor_review.md` this round (per mission).

## Open risks

- With `K=8` batches/epoch, 8 partial rows = full epoch leak for GARD/T1p (same as Round 02 T1p budget).
- GARD gains assume oracle assignment (conditional threat, not wrong40).
- 14×14 is a tractability fallback, not a substitute for 28×28 deployment claims.
