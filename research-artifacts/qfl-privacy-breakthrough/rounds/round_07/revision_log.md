# Round 07 — Revision Log

## Pivot from Round 05

Snapshot-DP killed for primary path (supervisor Round 05). Round 07 returns to **assignment / terminal-channel** stack on JASPER-Q with structural upgrades, not defense noise.

## Changes

- Added `code/benchmark_round07.py`:
  - `recover_t1p_multi_epoch` — cumulative T1p recovery with decay-weighted epoch snapshots
  - `blended_graph_laplacian` — co-occurrence + spectral kNN on `s_hat`
  - `run_jasper_q_v7` — conditional T1p blend; fraction grid 0.25–0.75
  - `run_leak_cert_t1p_v2` — trace/Ky-Fan certificate vs trace-inflated naive
- Pre-registered relaxed gates in JSON `preregistered_gates`
- Scientist-only round (no supervisor review artifact)

## Dependencies

- Round 02 Stage-2 core (`benchmark_round02.py`)
- Round 04 JASPER Sinkhorn / overlap (`benchmark_round04.py`)

## Results (`artifacts/round07_metrics.json`, ~48s)

### JASPER-Q v7 (level1, mean over seeds)

| Assignment | Best fraction | JASPER-Q v7 MSE | GARD @ same frac |
|------------|---------------|----------------:|-----------------:|
| wrong40 | 0.75 | **0.888** | 1.11 |
| wrong40 | 0.25 | 0.949 | 2.65 |
| oracle | 0.75 | **0.669** | ~0 |
| oracle | 0.25 | 0.851 | 0.402 |

Multi-epoch + spectral graph modestly improves wrong40 at high fractions; oracle path benefits from **disabled** T1p warm-start (0.85 → 0.67–0.85).

### Pre-registered gates

| Gate | Pass? |
|------|-------|
| Relaxed A (MSE ≤0.15, ≥50% rows) | **No** |
| wrong40 @0.10, ≥40% rows | **No** |
| Standard A @0.10, ≥25% rows | **No** |
| Criterion C (trace naive / cert ≥2×, coverage) | **Yes** (5/5 seeds) |

### LEAK-CERT-T1p v2

Mean empirical T1p MSE **0.424**; cert **1.08**; trace-naive **16.0**; ratio **14.8×**; all seeds covered.
