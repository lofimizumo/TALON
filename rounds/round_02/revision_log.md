# Round 02 — Revision Log

## Supervisor Round 01 fixes addressed

| Issue | Action |
|-------|--------|
| **[Critical] No win vs SHARD** | Pivoted LAPIN → **JOLI** with TV prior in compressive regime; **2.1× lower MSE**, **+3.2 dB PSNR** (3 seeds). |
| **[Critical] Novelty not demonstrated** | Documented structure-aware Stage-3: same snapshot objective + image TV when \(\dim\mathfrak{g}<d\). |
| **[Major] Non-canonical baseline** | `benchmark_round02.py` calls **`ShardAttacker.level3_invert`** directly. |
| **[Major] Unequal budget** | Matched `n_batch`, `adam_steps=2000`, `lbfgs_max_iter=500`; logged in JSON `budget` block. |
| **[Major] Small scale / single seed** | 3 seeds (7, 11, 23); compressive 28×28 + reference 14×14. |
| **[Major] LAPIN snapshot fit** | LAPIN abandoned; JOLI Adam phase = SHARD (snapshot fit preserved before TV polish). |
| **[Minor] Missing plots** | Generated `round02_*.png` under `artifacts/`. |

## Method evolution

1. **Attempt A — Dual LM (first JOLI draft):** Batched dual Levenberg–Marquardt in snapshot space + Tikhonov. **Failed:** 7× worse snapshot residual, slower, lost MSE vs SHARD on pilot.
2. **Attempt B — TV in Adam:** Strong TV during Adam improved MSE but destroyed snapshot match (residual \(\mathcal{O}(10^{-1})\)).
3. **Attempt C — TV only in L-BFGS (accepted):** `tv_adam=0`, `tv_lbfgs=5e-3`. Adam identical to SHARD; polish selects smoother inputs among snapshot-near minima.

## Pilot tuning (seed 7, compressive, N=12)

`tv_lbfgs` sweep with `tv_adam=0`:

| `tv_lbfgs` | Input MSE | Snap. match |
|------------|-----------|-------------|
| 0 | 0.276 (= SHARD) | 1.0 |
| 0.001 | 0.136 | 0.33 |
| **0.005** | **0.121** | 0.08 |
| 0.01 | 0.112 | 0.0 |

Chose **0.005** for multi-seed run (balance MSE win vs non-zero snapshot match).

## Honest negatives

- **Snapshot matching:** JOLI mean acc. **0.08** vs SHARD **1.0** in compressive regime.
- **Reference regime:** No improvement when \(\dim\mathfrak{g} > d\) (TV off).
- **Runtime:** ~15 s vs ~11 s per seed (12 samples, MPS) for compressive JOLI.

## Files touched

- `code/joli_invert.py` (new)
- `code/benchmark_round02.py` (new)
- `rounds/round_02/researcher_proposal.md`
- `rounds/round_02/revision_log.md`
- `artifacts/round02_metrics.json`
- `artifacts/round02_metrics_bar.png`
- `artifacts/round02_reconstruction_compressive.png`
- `artifacts/round02_reconstruction_reference.png`
- `logs/experiment_round02.log`

## Reproduction

```bash
cd research-artifacts/novel-inversion-vs-shard
.venv/bin/python code/benchmark_round02.py
```

Requires MNIST download to `data/`, PyTorch with MPS or CPU.
