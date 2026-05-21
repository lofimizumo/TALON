# Round 05 revision log

## Supervisor inputs

- Full demands: `rounds/round_04/supervisor_review.md` (`REVISE_MAJOR`).
- Root cause: `paper/root_cause_l3_stall.md`.
- R3 baseline: `artifacts/round03_metrics.json` (28×28, 0/3 joint pass, T1p PSNR ~7.2 dB mean).

---

## Changes from Round 04 feedback

| R04 issue | Round 05 action |
|-----------|-----------------|
| No 28×28 in artifact | Quick track labeled `mnist_tracks.primary_28x28.skipped_reason=QFL_QUICK`; `full_28x28_runtime_estimate` logged (~11 h) — **full run not started** this round |
| GARD-SPARSE not evaluated | Included in `SNAPSHOT_PATHS` for quick and full configs |
| Proposal ↔ artifact mismatch | Quick env explicit in `run_env` + proposal tables; full protocol unchanged in `benchmark_round05.py` defaults |
| Misleading fallback JSON | Round 05 uses `mnist_tracks` (no `ran_because_28_failed` when 28×28 skipped) |
| L3 budget naming | Distinguish `tv_lbfgs=0.005` (R3 Pareto) vs `mnist28_quality` (adam=400, n_batch≤128) vs quick `quick_r5_l3_sweep` |
| Per-seed pass rates | `per_seed_pass_best_adam` in metrics JSON + tables in `researcher_proposal.md` |
| Stall logging | `StallWatchdog` + per-image lines in `logs/experiment_round05.log` |

---

## Code touched

- `code/benchmark_round05.py` — env labeling, `per_seed_pass_best_adam`, `_full_mode_estimate()`, accurate resize in log header.
- `code/quick_config.py` — round≥5: 3 paths, dim_g {100,160}, adam_grid {500,1000}.
- `code/run_monitor.py`, `code/l3_budget.py` — unchanged API; used by R05 benchmark.

---

## Experiments executed (Round 05 scientist)

**Command:**
```bash
cd /workspace/research-artifacts/qfl-vqc-mnist-recon
QFL_QUICK=1 PYTHONPATH=/workspace/vendor:/workspace/code python3 code/benchmark_round05.py
```

**Log:** `logs/experiment_round05.log`  
**Metrics:** `artifacts/round05_metrics.json`  
**Grid:** `artifacts/round05_recon_grid.png` (seed 3, dim_g=100, quick — first dim in sweep)

| Metric | Value |
|--------|------:|
| Wall time | ~96 s |
| Est. JOLI | ~234 s |
| Stall warnings | **0** |
| Joint pass (any path, 2 seeds) | **0/2** |
| Best single draw | seed 3, dim_g=160, GARD, adam=1000 — MSE **0.039**, PSNR **14.09 dB** |
| Best mean PSNR (aggregate) | GARD @ dim_g=160 — **12.43 dB** mean |

**Adam sweep (H1):** Increasing 500→1000 gives **mixed** gains; best improvement on GARD seed 3 (+~0.17 dB PSNR, MSE 0.041→0.039); T1p seed 7 adam=1000 **worse** than 500 on PSNR. Not monotonic — do not assume higher adam fixes 28×28.

---

## Full 28×28 decision

| Criterion | Assessment |
|-----------|------------|
| Quick PSNR trend toward 18 dB? | **No** — ceiling ~14 dB on best draw (~4 dB short) |
| MSE≤0.05 on any seed? | **Yes** — one draw (GARD seed 3); not ≥2/2 seeds |
| Logged full estimate? | **Yes** — ~40.1k s (~11.1 h, dim_g=160) in `full_28x28_runtime_estimate` |
| Stall monitor ready? | **Yes** |

**Recommendation:** Schedule `QFL_FULL=1` as **mandatory acceptance evidence** for supervisor round 5 closeout, but **expect REVISE** on metrics unless inversion/snapshot changes. Do **not** block on quick success. Parent draft metrics replaced by this run (2026-05-21).

---

## Honesty notes

- No injected metrics; log lines match JSON `per_seed` rows.
- `image_matching_acc=0`: small N=12, metric not used for acceptance.
- Replaced mistaken parent-written round 05 drafts with scientist-owned proposal + this log.
