# Round 01 — Researcher Proposal: LAPIN for SHARD Stage 3

## Problem

After SHARD Stages 1–2, the attacker holds per-sample snapshot vectors \(s_i \approx \cos(W_{\text{enc}} x_i + b_{\text{enc}})\). Stage 3 must recover inputs \(x_i \in [0,1]^d\) by solving, for each \(i\),

\[
\min_{x \in [0,1]^d} \; \|\cos(W x + b) - s_i\|_2^2 .
\]

The current implementation (`ShardAttacker.level3_invert`) uses **batched parallel Adam** over \(\mathcal{O}(10^2)\) random initializations, encoding-aware `lstsq` seeds from multiple \(\mathrm{acos}\) branches, then **L-BFGS** on the best candidate. This is effective but:

1. Expensive (GPU memory scales with number of parallel starts).
2. Structurally identical to generic non-convex first-order inversion (akin to DLG-style descent, but on snapshot residual).
3. Does not exploit the **analytic Jacobian** of the cos encoding.

## Named method: **LAPIN**

**L**atent **P**rojected **Ga**uss–**N**ewton **IN**version.

### Algorithm (per snapshot)

1. **Seeds (5):** Box-clamped least squares on \(W x \approx \theta - b\) for four \(\mathrm{acos}(s)\) branches plus gray \(x=0.5\).
2. **Multi-start projected GN (53 starts):** Five seeds + 48 uniform random \(x_0 \in [0,1]^d\). For each, run up to 25 GN steps:
   - \(z = Wx+b\), residual \(r = \cos(z)-s\).
   - \(J = \mathrm{diag}(-\sin z)\, W\).
   - Solve \((J^\top J + \lambda I)\,\delta = J^\top(-r)\), backtracking line search, clip \(x\) to \([0,1]\).
3. **Polish:** Single L-BFGS on the best GN point (same objective as SHARD polish).

**Complexity:** \(\mathcal{O}(K \cdot T_{\text{GN}} \cdot (d \cdot \dim\mathfrak{g}^2))\) with small \(K \approx 53\), vs SHARD \(K \approx 120\) parallel Adam × 800 steps.

### Why novel vs prior art

| Approach | Distinction |
|----------|-------------|
| DLG / iDLG / Geiping | Match **gradients** w.r.t. network weights; no fixed \(\cos(Wx+b)\) encoder |
| Analytic VQC inversion (Heredge et al.) | Requires separable / polynomial structure; not applicable to dense RFF |
| SHARD L3 | **Many parallel Adam** trajectories + L-BFGS |
| **LAPIN** | **Second-order GN** using structured Jacobian; **no Adam**; comparable multi-start count but different optimization geometry |

Not claimed: fewer restarts alone. Claimed: **exploitation of cos-encoding curvature** via GN in input space.

## Experiment plan (Round 01 executed)

- **Data:** MNIST train, first 12 digits, resized \(14\times14\) (\(d=196\)).
- **Surrogate:** `SurrogateQFL(dim_g=160, seed=7)`, noiseless snapshots.
- **Baselines:** Faithful SHARD L3 objective with reduced parallel budget (`n_batch=120`, Adam 800 steps, L-BFGS 300) for laptop reproducibility.
- **Metrics:** Hungarian-aligned input MSE/PSNR, snapshot matching accuracy, mean snapshot residual, wall-clock.

### Round 01 results (actual run)

| Method | Input MSE ↓ | PSNR ↑ | Snap. match acc. | Runtime (s) |
|--------|-------------|--------|------------------|-------------|
| SHARD L3 | **0.0127** | **19.0 dB** | **0.58** | 4.9 |
| LAPIN | 0.0877 | 10.6 dB | 0.00 | 4.1 |

Artifacts: `artifacts/round01_metrics.json`, `round01_reconstruction_grid.png`, `round01_metrics_bar.png`.

**Interpretation:** On this micro-benchmark, SHARD L3 still dominates perceptual error; LAPIN achieves similar wall-clock but lands in worse local minima despite GN + L-BFGS. Round 2 should test (i) LAPIN-only init + SHARD-scale L-BFGS, (ii) joint multi-snapshot GN, (iii) underdetermined \(d \ll \dim\mathfrak{g}\) regime where GN may shine.

## Assumptions & limitations

- Stages 1–2 assumed correct (oracle snapshots).
- Surrogate matches paper RFF encoding.
- Benchmark SHARD uses reduced Adam batch (documented in `logs/experiment_round01.log`); full 500-start SHARD may widen quality gap.
- No CIFAR / larger \(d\) yet.

## Code map

- `code/lapin_invert.py` — LAPIN implementation
- `code/benchmark_round01.py` — comparison driver
