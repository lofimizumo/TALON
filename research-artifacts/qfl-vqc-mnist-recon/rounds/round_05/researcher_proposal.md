# Round 05 — L3 quality budget, stall monitoring, honest quick→full gate

## Response to Round 04 supervisor (`REVISE_MAJOR`)

Round 04 proved snapshot recovery (T1p+graph, SHARD) but **never ran 28×28** under default `QFL_QUICK=1`, omitted **GARD-SPARSE**, and under-budgeted L3 (adam=120 → PSNR ~9 dB at 14×14). Round 05 addresses:

1. **Three snapshot paths** on every run: `lasa_qterm_T1p`, `gard_sparse_oracle`, `shard_oracle`.
2. **L3 adam sweep** on quick 14×14 (500, 1000) to test H1 before multi-hour 28×28.
3. **`StallWatchdog`** + per-image JOLI logs (`run_monitor.py`, `l3_budget.py`).
4. **Honest JSON labeling** (`run_env`, `mnist_tracks`, `full_28x28_runtime_estimate`; no false “28 failed” flags).
5. **Full-track profile** `mnist28_quality` (adam=400, lbfgs=200, n_batch≤128, **no LAPIN** on d=784) — documented separately from R3 “tv_lbfgs only” label.

---

## Hypotheses

| ID | Statement | Test |
|----|-----------|------|
| **H1** | Raising adam 500→1000 at 14×14 improves input PSNR monotonically at fixed `tv_lbfgs=0.005` | Quick adam grid per path/seed |
| **H2** | GARD-SPARSE oracle MAP yields lower snap MSE than T1p and narrows snap→image gap vs R4 | Compare snap vs input MSE per path |
| **H3** | `dim_g=160` beats `dim_g=100` on joint metrics at comparable L3 cost | Quick dim_g sweep {100, 160} |
| **H4** | `mnist28_quality` at 28×28 can beat R3 adam=250 (mean PSNR ~7.2 dB) without R3-style stall | **Deferred** — full run only if quick trend + estimate logged |

**Not claimed:** 14×14 quick results satisfy `config.json` `require_mnist_end_to_end` (28×28 preferred).

---

## Method

| Component | Quick (`QFL_QUICK=1`) | Full (`QFL_QUICK=0 QFL_FULL=1`) |
|-----------|----------------------|--------------------------------|
| Resolution | 14×14 (d=196) | 28×28 (d=784) |
| N / epochs | 12 / 5 | 32 / 10 |
| Seeds | 3, 7 | 3, 7, 11 |
| dim_g | 100, 160 | 160 (sweep 100,256 optional via `QFL_FULL_DIM_SWEEP=1`) |
| Snapshots | T1p, GARD-SPARSE, SHARD oracle | same |
| L3 | `quick_r5_l3_sweep`: adam∈{500,1000}, lbfgs=120, n_batch≤64 | `mnist28_quality`: adam=400, n_batch≤128 |
| Monitor | StallWatchdog, heartbeat 600s, wall≤3× estimate | same |
| TV | `tv_lbfgs=0.005` from R3 Pareto (`load_r3_tv_lbfgs`) | same |

**Stack:** SurrogateQFL → LASA-QTERM T1p / GARD-SPARSE (oracle incidence) / SHARD L2 → `joli_invert_single` (JOLI quality).

**Baselines:** R3 `round03_metrics.json` (28×28, T1p, adam=250); R4 quick (14×14, adam=120, 2 paths).

---

## Experiment plan

1. **Quick smoke (required this round):**
   ```bash
   cd /workspace/research-artifacts/qfl-vqc-mnist-recon
   QFL_QUICK=1 PYTHONPATH=/workspace/vendor:/workspace/code python3 code/benchmark_round05.py
   ```
   Outputs: `artifacts/round05_metrics.json`, `logs/experiment_round05.log`, `artifacts/round05_recon_grid.png`.

2. **Full 28×28 (conditional):** Run only if quick shows **meaningful PSNR lift** toward 18 dB *and* logged estimate is acceptable. Command in `full_28x28_runtime_estimate` inside metrics JSON (~11 h CPU for dim_g=160, 3 seeds, 3 paths, N=32).

3. **Metrics:** Per-seed joint pass (MSE≤0.05 **and** PSNR≥18); aggregate uses **best adam** per seed/path. Report snap MSE separately for H2.

4. **Failure modes:** L3 stall (see `paper/root_cause_l3_stall.md`); abort optional `QFL_ABORT_ON_STALL=1`; kill and reduce budget if wall >3× estimate with no per-image logs.

---

## Assumptions & limitations

- GARD-SPARSE uses **oracle assignment** (conditional upper bound; not deployable threat model).
- `image_matching_acc=0` at N=12 is expected (Hungarian identity rarely aligns at low N).
- Quick PASS_MIN_SEEDS=2 of **2** seeds run; project acceptance requires **≥2 of 3** at 28×28.
- Adam scaling on 14×14 may not transfer linearly to d=784 (different n_batch cap and conditioning).

---

## Round 05 results (quick track — scientist run)

**Environment:** `QFL_QUICK=1`, 14×14, seeds 3 & 7, dim_g∈{100,160}, adam∈{500,1000}, wall ~98 s.

| Path (best adam/seed) | Best input MSE | Best PSNR | Joint pass (2 seeds) |
|-----------------------|---------------:|----------:|:--------------------:|
| GARD @ dim_g=160 seed 3 adam=1000 | **0.039** | **14.09 dB** | 0/2 (MSE ok, PSNR fail) |
| T1p @ dim_g=160 mean | 0.066 | 11.88 dB | 0/2 |
| SHARD @ dim_g=160 mean | 0.112 | 9.62 dB | 0/2 |

**H1 partial:** adam 1000 vs 500 helps GARD seed 3 (+0.17 dB) but not uniformly (e.g. GARD seed 7 regresses). No seed hits PSNR≥18.

**H2 supported on snap MSE** (GARD ~0.006 vs T1p ~0.014 at dim_g=160); **image gap remains** (~4 dB below PSNR gate on best draw).

**Full 28×28 recommendation:** **Run for acceptance evidence, not because quick predicts pass.** Quick shows ~+5 dB vs R4 quick on best GARD draw but still ~4 dB short of 18 dB; R3 at 28×28 was ~7 dB — full `mnist28_quality` may reach ~10–12 dB (speculative). Estimate ~11 h; stall monitor mandatory. **Do not start full run in this round** without supervisor sign-off on wall clock.

---

## Expected supervisor outcome

**REVISE_MAJOR** or **REVISE** unless full 28×28 achieves ≥2/3 seed joint pass. Round 05 closes the infrastructure/labeling loop; acceptance likely needs further snapshot or inversion changes beyond adam budget alone.
