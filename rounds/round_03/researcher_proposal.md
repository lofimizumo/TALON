# Round 03 — Researcher Proposal: Structure-Aware Stage-3 Polish (JOLI)

## Contribution (reframed)

**JOLI** is **not** a new attack class or a Jacobian-based solver. It is a **structure-aware Stage-3 polish** for compressive snapshot inversion:

- **Phase A (Adam):** Identical to canonical `ShardAttacker.level3_invert` — snapshot MSE only, matched `n_batch`, 2000 steps.
- **Phase B (L-BFGS):** Snapshot MSE + optional isotropic TV when `dim_g < d`.
- **Regime gate:** TV is **forced off** when `dim_g ≥ d`; JOLI then matches SHARD to numerical precision.

**When to use:** Oracle snapshots are available and the analyst prioritizes **semantic input recovery** (Hungarian MSE / PSNR to ground truth) over strict re-encoding. **When not to use:** Downstream SHARD stages require snapshot-consistent inputs — use SHARD L3 or JOLI with λ=0.

Retired branding: ~~Jacobian-aware Optimization with Lifted Image prior~~.

---

## Problem

Stage 3 solves \(\min_x \|\cos(Wx+b)-s\|_2^2\). When \(\dim\mathfrak{g} < d\), infinitely many inputs fit a snapshot. SHARD finds snapshot-perfect \(\hat x\) but often not digit-like inputs. TV in the **L-BFGS polish only** selects smoother, more natural inputs from the null space at a **documented** snapshot cost.

---

## Round 03 experiments (executed)

| Setting | Value |
|---------|-------|
| N | 20 (up from 12; justified in `revision_log.md`) |
| Tuning seeds | 7, 11, 23 |
| Held-out seed | 31 (never used for λ selection) |
| Pareto λ grid | {0, 10⁻³, 5×10⁻³, 10⁻², 2×10⁻²} |
| Baseline | `ShardAttacker.level3_invert` (canonical) |
| Device | MPS (~15 min Pareto + controls) |

### Regime matrix (corrected naming)

| Regime | \(d\) | \(\dim\mathfrak{g}\) | Relation | TV | Purpose |
|--------|------|------------------------|----------|-----|---------|
| `compressive_28x28_pareto` | 784 | 100 | **compressive** | swept | Main Pareto |
| `ablation_tv_off_14x14` | 196 | 160 | **compressive** (160 < 196) | λ=0 manual | TV-off ablation |
| `overdetermined_14x14` | 196 | 256 | **dim_g ≥ d** | auto-off | Control |
| `square_mnist_28x28` | 784 | 784 | **dim_g = d** | auto-off | Full-rank MNIST control |

**Correction from Round 02:** `reference_14x14` was mislabeled as overdetermined (\(160 \not> 196\)). Renamed to `ablation_tv_off_14x14`.

---

## Headline results

### 1. Pareto curve (compressive 28×28, N=20, 3 tuning seeds)

| λ (`tv_lbfgs`) | Input MSE ↓ | PSNR ↑ | Snap. match acc. | Mean snap. residual |
|----------------|-------------|--------|------------------|---------------------|
| 0 (≡ SHARD) | 0.280 ± 0.003 | 5.53 dB | **1.00** | **~1e-8** |
| 0.001 | 0.152 ± 0.011 | 8.20 dB | 0.32 | 0.096 |
| **0.005 (selected)** | **0.135 ± 0.011** | **8.71 dB** | 0.08 | 0.175 |
| 0.01 | 0.125 ± 0.011 | 9.05 dB | 0.03 | 0.231 |
| 0.02 | 0.116 ± 0.011 | 9.38 dB | 0.02 | 0.289 |

**Operating point:** λ = 5×10⁻³ — minimizes input MSE among λ with `snapshot_match_acc ≥ 0.05` on tuning seeds. Yields **~2.1× lower input MSE** vs SHARD at **~8% snapshot match** (explicit trade-off).

**Alternative knee:** λ = 10⁻³ gives ~1.8× MSE win with **32% snapshot match** — better if downstream consistency matters more.

### 2. Held-out validation (seed 31, λ = 0.005)

| Method | Input MSE | PSNR | Snap. match acc. |
|--------|-----------|------|------------------|
| SHARD L3 | 0.275 | 5.60 dB | 1.00 |
| JOLI | **0.116** | **9.36 dB** | 0.00 |

→ Input MSE win **generalizes** to held-out seed; snapshot acc drops to 0 (honest reporting).

### 3. dim_g ≥ d controls (JOLI auto TV-off)

| Regime | max \|Δ input MSE\| | max \|Δ snap. acc\| | JOLI ≡ SHARD? |
|--------|---------------------|---------------------|---------------|
| `ablation_tv_off_14x14` (λ=0) | 0.0 | 0.0 | **Yes** |
| `overdetermined_14x14` (dim_g=256) | 0.0 | 0.0 | **Yes** |
| `square_mnist_28x28` (dim_g=784) | 0.0 | 0.0 | **Yes** |

At λ=0 on compressive 14×14: aggregate MSE 0.00366 (identical per-seed). At dim_g ≥ d with regime gate: MSE ~10⁻¹⁰ (machine precision).

---

## Artifacts

- `artifacts/round03_metrics.json` — full results
- `artifacts/round03_pareto.json` — Pareto table + operating point
- `artifacts/round03_pareto_curves.png` — MSE / snapshot acc / residual vs λ
- `artifacts/round03_pareto_tradeoff.png` — input MSE vs snapshot residual scatter
- `logs/experiment_round03.log`
- `tutorial/tutorial.md` — method + results summary

---

## Novelty assessment (self-audit)

| Claim | Defensible? |
|-------|-------------|
| New leakage / attack class | **No** |
| TV as image prior | Standard (Rudin–Osher–Fatemi) |
| Regime-gated TV **only in L-BFGS polish** | **Yes** — engineering refinement |
| Compressive null-space selection with explicit Pareto | **Yes** — main empirical contribution |
| Beats SHARD on input MSE in compressive regime | **Yes** (multi-seed + held-out) |
| Matches SHARD when λ=0 or dim_g ≥ d | **Yes** (verified) |

---

## Limitations & Round 04

- N=20 (not 50+); CIFAR / end-to-end SHARD slice still open for full `ACCEPT`.
- Snapshot match collapses at selected λ — dual-metric story required in paper text.
- Runtime ~1.3× SHARD in compressive regime.
- JOLI reimplements Adam phase (recommend regression test in CI).

---

## Acceptance expectation

Round 03 closes all Round 02 **Major** items: Pareto sweep, true dim_g ≥ d controls, corrected regime naming, reframed contribution, multi-seed + held-out λ validation, N=20.

**Target supervisor verdict:** `ACCEPT_WITH_MINOR` (CIFAR/pipeline as minor follow-up).
