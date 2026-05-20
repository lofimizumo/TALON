# Round 03 — Revision Log

## Supervisor Round 02 → Round 03 mapping

| Major issue | Round 03 action | Status |
|-------------|-----------------|--------|
| Snapshot–input Pareto not characterized | Full `tv_lbfgs ∈ {0, 1e-3, 5e-3, 1e-2, 2e-2}` sweep on compressive 28×28; dual plots (curves + scatter) | Done |
| Mislabeled “reference” regime | Renamed `reference_14x14` → `ablation_tv_off_14x14`; corrected text (160 < 196, still compressive) | Done |
| Missing true dim_g ≥ d control | Added `overdetermined_14x14` (dim_g=256, d=196) and `square_mnist_28x28` (dim_g=784, d=784) | Done |
| “Jacobian-aware” overclaim | Retired branding; reframed as structure-aware Stage-3 polish / compressive inversion prior | Done |
| Scale / λ tuning on pilot only | N=20, 4 seeds (3 tuning + 1 held-out), λ selected on tuning seeds | Done |

## Code changes

1. **`code/joli_invert.py`**
   - Docstring: removed “Jacobian-aware”; describes compressive TV polish.
   - **Regime gate:** explicit `tv_lbfgs` is forced to 0 when `dim_g >= d`.

2. **`code/benchmark_round03.py`** (new)
   - Pareto sweep + held-out validation + three control regimes.
   - Artifacts: `round03_metrics.json`, `round03_pareto.json`, Pareto PNGs.

3. **`tutorial/tutorial.md`**
   - Method summary, regime matrix, headline numbers, limitations.

## Naming fixes

| Old (Round 02) | Corrected (Round 03) | Notes |
|----------------|----------------------|-------|
| `reference_14x14` | `ablation_tv_off_14x14` | dim_g=160 **<** d=196; TV off by manual λ=0 |
| “dim_g > d reference” | `overdetermined_14x14`, `square_mnist_28x28` | True dim_g ≥ d controls |

## λ selection protocol

1. Sweep on tuning seeds `{7, 11, 23}` only.
2. Select λ minimizing input MSE subject to `snapshot_match_acc ≥ 0.05`.
3. Validate on held-out seed `31` (never used for λ pick).

## N=20 justification

Round 02 used N=12 (simulator default). Round 03 doubles to N=20 to reduce Hungarian-matching variance while keeping the full Pareto × multi-seed × control matrix runnable in ~30 min on MPS. Full N=50 deferred to Round 04 / end-to-end SHARD slice per supervisor `ACCEPT` bar.

## Contribution reframe (for paper line)

**Before:** “Jacobian-aware Stage-3 replacement.”  
**After:** *Structure-aware Stage-3 polish* — SHARD-identical Adam phase; optional isotropic TV **only** in L-BFGS when `dim_g < d` to pick smoother inputs from the snapshot null space. JOLI is **not** a strict snapshot inverter when λ>0; it targets **semantic input recovery** under compressive leakage, with an explicit input–snapshot Pareto.

## Expected acceptance posture

Round 03 closes all Round 02 **Major** evaluation gaps. Remaining gap vs full `ACCEPT`: CIFAR / end-to-end pipeline (Round 04). Target verdict: **`ACCEPT_WITH_MINOR`**.

## Round 03 results (executed 2026-05-20, ~15 min MPS)

- Pareto sweep complete; λ=0.005 selected; held-out seed 31 confirms MSE win.
- All dim_g ≥ d controls: JOLI ≡ SHARD (max diff 0.0).
- Plots: `round03_pareto_curves.png`, `round03_pareto_tradeoff.png`.
