# Round 03 — L3 budget grid on 28×28 T1p (experiment-only)

## Goal

After Round 02 established full-resolution tractability but missed acceptance targets, Round 03 sweeps **Level-3 compute budget** on the primary weak path (**LASA-QTERM T1p**) at `dim_g=100`, comparing **JOLI** and **LAPIN**, while fixing the **SHARD oracle** upper bound via **B=1** disaggregation.

## Hypotheses

**H1:** Increasing `adam_steps` ∈ {2000, 4000, 6000} with `lbfgs=1000` and `n_batch=max` improves T1p→image pass rate without changing snapshot recovery.

**H2:** JOLI (`tv_lbfgs=0.005`, parent Pareto) beats LAPIN on T1p when snapshot MSE is in the weak-but-usable band.

**H3:** B=1 oracle (`level2_disaggregate_b1`) yields lower oracle snapshot MSE than B=4 graph L2 (Round 02 non-convergence warnings).

## Protocol

| Knob | Value |
|------|-------|
| Resolution | 28×28 (`d=784`) |
| `dim_g` | 100 |
| Seeds | 3, 7, 11 |
| Snapshot path (L3) | `lasa_qterm_T1p` only |
| L3 inverters | `joli_l3`, `lapin_l3` |
| `adam_steps` | 2000, 4000, 6000 |
| `n_batch` | `shard_n_batch` max (~500) |
| `lbfgs` | 1000 |
| Warm-start | Encoding lstsq seeds + random fill |
| SHARD oracle | `level2_disaggregate_b1` or B=1 fallback |

## Deliverables

- **Script:** `code/benchmark_round03.py`
- **Metrics:** `artifacts/round03_metrics.json`
- **Grids:** `artifacts/round03_recon_grid_*.png`
- **Log:** `logs/experiment_round03.log`
- **No** `supervisor_review.md` (experiment-only round)

## Run

```bash
cd /workspace/research-artifacts/qfl-vqc-mnist-recon
python3 code/benchmark_round03.py
```
