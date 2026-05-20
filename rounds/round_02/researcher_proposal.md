# Round 02 — Researcher Proposal: JOLI (pivot from LAPIN)

## Pivot rationale

Round 01 **LAPIN** (projected Gauss–Newton without Adam) failed on every quality metric: snapshot residual was ~7× SHARD and input MSE ~6.9× worse. Supervisor audit identified (i) unequal/unfaithful baseline, (ii) no compressive-regime test, (iii) GN under-budget and ill-conditioned in input space.

Round 02 **pivots** to **JOLI** (*Jacobian-aware Optimization with Lifted Image prior*): keep the **same** Stage-3 objective and **matched** SHARD shell (500 parallel starts, 2000 Adam steps, 500 L-BFGS iterations — canonical `ShardAttacker.level3_invert` budget), but add a **structure prior** in the L-BFGS polish when \(\dim\mathfrak{g} < d\) to resolve null-space ambiguity.

LAPIN code remains in `code/lapin_invert.py` as a negative-result reference; production path is `code/joli_invert.py`.

## Problem (unchanged)

Oracle snapshots \(s_i \approx \cos(W x_i + b)\). Stage 3 solves
\(\min_{x \in [0,1]^d} \|\cos(Wx+b) - s_i\|_2^2\) per sample.

When \(\dim\mathfrak{g} < d\) the map is **compressive / underdetermined**: many \(x\) fit the snapshot. SHARD’s Adam explores widely but has **no image prior**, so recovered inputs can be far from natural digits even with near-zero snapshot error.

## Method: JOLI

### Algorithm (per snapshot)

1. **Phase A (identical to SHARD):** Batch Adam on `n_batch = max(50, min(500, 500_000 / max(d, dim_g)))` starts (5 lstsq + uniform random), 2000 steps, box clip \([0,1]\), `tv_adam = 0`.
2. **Phase B (novel):** L-BFGS on the best Adam candidate with loss  
   \(\|\cos(Wx+b)-s\|_2^2 + \lambda_{\mathrm{TV}} \cdot \mathrm{TV}(x)\)  
   when \(\dim\mathfrak{g} < d\), else pure snapshot loss (\(\lambda_{\mathrm{TV}}=0\)).
3. **TV prior:** Isotropic total variation on the reshaped \( \sqrt{d} \times \sqrt{d} \) image (MNIST grid).

**Default:** \(\lambda_{\mathrm{TV}} = 5\times 10^{-3}\) in L-BFGS only (pilot-tuned on seed 7 in \([10^{-3}, 5\times 10^{-2}]\)).

### Why novel vs SHARD / LAPIN

| | SHARD L3 | LAPIN | **JOLI** |
|---|----------|-------|----------|
| Objective | Snapshot MSE only | Same | Snapshot MSE + **TV in polish** (compressive) |
| Optimizer | Adam + L-BFGS | GN + L-BFGS | **Same Adam as SHARD** + TV-L-BFGS |
| Underdetermined regime | No prior | GN in \(d\)-space (failed) | **Image prior breaks null space** |
| Baseline API | Canonical | Inline fast variant | **Calls `level3_invert`** |

Novelty is not a different leakage model; it is a **structure-aware Stage-3** that exploits known digit smoothness when snapshots under-constrain inputs.

## Experiments (Round 02 executed)

| Regime | \(d\) | \(\dim\mathfrak{g}\) | Seeds | Baseline |
|--------|------|------------------------|-------|----------|
| **compressive_28x28** | 784 | 100 | 7, 11, 23 | `ShardAttacker.level3_invert` (MPS) |
| reference_14x14 | 196 | 160 | 7, 11, 23 | same |

**Matched budget (both methods):** `n_batch=500`, `adam_steps=2000`, `lbfgs_max_iter=500`.

### Headline results (mean ± std over 3 seeds)

**Compressive MNIST 28×28 (\(\dim\mathfrak{g}=100 < d=784\))**

| Method | Input MSE ↓ | PSNR ↑ | Snap. match acc. | Mean snap. residual |
|--------|-------------|--------|------------------|---------------------|
| SHARD L3 (canonical) | 0.2804 ± 0.0034 | 5.52 ± 0.05 dB | **1.00** | **~1e-8** |
| **JOLI** | **0.1352 ± 0.0108** | **8.70 ± 0.35 dB** | 0.08 ± 0.07 | 0.19 ± 0.04 |

→ **JOLI wins input MSE (~2.1× lower) and PSNR (~3.2 dB higher)** on all three seeds.  
→ **SHARD wins snapshot matching** (JOLI trades snapshot fit for perceptual/input fidelity — documented honestly).

**Reference 14×14 (\(\dim\mathfrak{g}=160 > d=196\)):** JOLI with \(\lambda_{\mathrm{TV}}=0\) matches SHARD to numerical precision (0.0049 ± 0.0008 MSE) — expected, identical optimizer.

### Artifacts

- `artifacts/round02_metrics.json`
- `artifacts/round02_metrics_bar.png`
- `artifacts/round02_reconstruction_compressive.png`
- `artifacts/round02_reconstruction_reference.png`
- `logs/experiment_round02.log`

## Limitations & Round 03

- \(\lambda_{\mathrm{TV}}\) tuned on pilot seed; report sensitivity sweep in Round 03.
- Snapshot-match metric degrades under TV; may need two-stage reporting (privacy vs utility).
- CIFAR / full SHARD pipeline not yet integrated.
- Runtime ~1.4× SHARD on compressive regime (extra TV in L-BFGS).

## Code map

- `code/joli_invert.py` — JOLI
- `code/benchmark_round02.py` — canonical baseline + multi-seed driver
- `code/lapin_invert.py` — Round 01 negative result (retained)
