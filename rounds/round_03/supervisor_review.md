# Supervisor Review — Round 03

**Run folder:** `research-artifacts/novel-inversion-vs-shard/`  
**Reviewed:** `researcher_proposal.md`, `revision_log.md`, `tutorial/tutorial.md`, `code/joli_invert.py`, `code/benchmark_round03.py`, `artifacts/round03_metrics.json`, `artifacts/round03_pareto.json`, `logs/experiment_round03.log`  
**Baseline reference:** `supplementary_materials/code/shard_sim/attacker.py` (`level3_invert`, L693–851)  
**Prior audit:** `rounds/round_02/supervisor_review.md` (`REVISE_MINOR`)

---

## 1. Verdict

**`ACCEPT_WITH_MINOR`**

Round 03 **closes all Round 02 Major evaluation gaps**: multi-seed Pareto over `tv_lbfgs`, true \(\dim\mathfrak{g} \ge d\) controls, corrected regime naming, reframed contribution, held-out \(\lambda\) validation, and increased \(N=20\). Experiments are **honest, reproducible, and not hardcoded**.

The contribution is a **defensible compressive-regime Stage-3 polish** (regime-gated TV in L-BFGS only) with a **documented input–snapshot trade-off** and **~2× input MSE win** vs canonical SHARD L3 on the intended surrogate setting. Novelty remains **incremental** (standard TV prior; engineering refinement of SHARD L3), which is **sufficient for `ACCEPT_WITH_MINOR`** but **not** for unconditional `ACCEPT` without CIFAR / end-to-end pipeline evidence.

**Multi-round loop:** Per `config.json` (`stop_mode: "until_acceptance"`, `acceptance.supervisor_verdicts: ["ACCEPT", "ACCEPT_WITH_MINOR"]`), **this run may stop** after this verdict. Optional follow-ups below are **minor**, not blocking another research round unless the user wants paper-scale scaling.

---

## 2. Executive summary

**JOLI** keeps SHARD’s Adam phase (`tv_adam=0`, matched `n_batch`, 2000 steps) and adds isotropic TV **only** in the L-BFGS polish when \(\dim\mathfrak{g} < d\). A **regime gate** forces \(\lambda_{\mathrm{TV}}=0\) when \(\dim\mathfrak{g} \ge d\).

**Compressive Pareto (28×28, \(\dim\mathfrak{g}=100\), \(d=784\), \(N=20\), tuning seeds 7/11/23)**

| \(\lambda\) (`tv_lbfgs`) | Input MSE ↓ | PSNR ↑ | Snap. match acc. | Mean snap. residual |
|--------------------------|-------------|--------|------------------|---------------------|
| 0 (≡ SHARD objective) | 0.280 ± 0.003 | 5.53 dB | **1.00** | **~1e-8** |
| 0.001 | 0.152 ± 0.011 | 8.20 dB | 0.32 | 0.096 |
| **0.005 (selected)** | **0.135 ± 0.011** | **8.71 dB** | 0.08 | 0.175 |
| 0.01 | 0.125 ± 0.011 | 9.05 dB | 0.03 | 0.231 |
| 0.02 | 0.116 ± 0.011 | 9.38 dB | 0.02 | 0.289 |

**Operating point:** \(\lambda=5\times10^{-3}\) — min input MSE among \(\lambda>0\) with `snapshot_match_acc ≥ 0.05` on tuning seeds → **~2.07×** lower MSE vs SHARD at **~8%** snapshot match (explicit trade-off).

**Held-out seed 31 (\(\lambda=0.005\)):** JOLI MSE **0.116** vs SHARD **0.275** (MSE win holds); snapshot match acc. **0.00** (must be reported in any paper claim).

**Controls (\(\lambda=0\) or auto TV-off, \(\dim\mathfrak{g} \ge d\)):** JOLI ≡ SHARD on all reported metrics (max abs diff **0.0**).

---

## 3. Honesty / reproducibility audit

| Check | Finding |
|-------|---------|
| Hardcoded metrics | **Pass.** `benchmark_round03.py` computes metrics from arrays; JSON via `json.dumps`. |
| JSON internal consistency | **Pass.** Supervisor spot-check: Pareto aggregates match `round03_pareto.json`; \(\lambda=0\) per-seed JOLI = SHARD; control regimes identical per-seed. |
| Full benchmark re-run | **Deferred** (~15 min MPS). Prior run logged `logs/experiment_round03.log` (2026-05-20, ~15 min); metrics audited against log and code. |
| Canonical baseline | **Pass.** `ShardAttacker.level3_invert` with paper budgets (`n_batch` formula, Adam 2000, L-BFGS 500). |
| \(\lambda=0\) / \(\dim\mathfrak{g}\ge d\) equivalence | **Pass** (code + live CPU test; see §4). |
| Pareto / regime plots | **Gap (Minor).** `benchmark_round03.py` writes `round03_pareto_curves.png` and `round03_pareto_tradeoff.png`, but `artifacts/` currently has only JSON; regenerate and commit. |
| Contribution framing | **Pass.** “Jacobian-aware” retired; tutorial and proposal state JOLI is **not** a strict snapshot inverter when \(\lambda>0\). |

---

## 4. Verification: \(\lambda=0\) and \(\dim\mathfrak{g} \ge d\) match SHARD

### 4.1 Code audit

| Element | SHARD `level3_invert` | JOLI `joli_invert_single` |
|---------|----------------------|---------------------------|
| `n_batch` | `max(50, min(500, 500_000 // max(d, dim_g)))` | Same via `shard_n_batch` |
| Adam seeds | Gray + 4 lstsq + uniform; `seed=42+i` | Same structure |
| Adam objective | \(\sum_i \|\cos(XW^T+b)-s\|^2\) | Same (`tv_adam=0`) |
| L-BFGS | Snapshot MSE only, CPU, strong Wolfe, 500 iter | Same when `tv_lbfgs=0` |
| TV gate | N/A | `joli_invert` forces `tv_lbfgs=0` if `dim_g >= d` (L156–157) |

JOLI duplicates SHARD’s loop rather than calling `level3_invert` for Phase A; structures are aligned.

### 4.2 Artifact evidence

- **Pareto \(\lambda=0\):** `aggregate_joli` equals `aggregate_shard` exactly on all four headline metrics (MSE, PSNR, snap. acc., residual).
- **Controls:** `ablation_tv_off_14x14` (\(\lambda=0\), compressive), `overdetermined_14x14` (\(\dim\mathfrak{g}=256\ge196\)), `square_mnist_28x28` (\(\dim\mathfrak{g}=784\)): per-seed and aggregate diffs **0.0**.

### 4.3 Live equivalence (supervisor, CPU, single snapshot)

| Regime | `max \|x_{\mathrm{SHARD}} - x_{\mathrm{JOLI}}\|` |
|--------|--------------------------------------------------|
| \(\dim\mathfrak{g}=d=784\), \(\lambda=0\) | **0.0** |
| Compressive \(\dim\mathfrak{g}=100\), \(\lambda=0\) | **0.0** |
| \(\dim\mathfrak{g}=784\), caller passes `tv_lbfgs=0.01` (gated off) | **0.0** |

**Conclusion:** When TV is off (explicit \(\lambda=0\) or \(\dim\mathfrak{g}\ge d\)), JOLI matches canonical SHARD L3 to numerical identity on tested paths.

---

## 5. Novelty vs acceptance bar

**User goal (`config.json`):** Novel gradient/snapshot inversion replacing SHARD Stage-3 with **measurable gains** on the surrogate QFL simulator.

| Question | Assessment |
|----------|------------|
| New attack / leakage channel? | **No** — same oracle snapshots. |
| TV as prior? | **Standard** (Rudin–Osher–Fatemi). |
| Defensible contribution? | **Yes** — **regime-gated** structure prior **only in L-BFGS polish**, preserving SHARD Adam exploration; **multi-seed Pareto** characterizes null-space selection vs snapshot consistency. |
| “Replace SHARD L3”? | **Conditionally yes** — as a **compressive-regime polish** when semantic input recovery matters more than strict re-encoding; **not** as a drop-in strict inverter at \(\lambda>0\). |
| Enough for `ACCEPT_WITH_MINOR`? | **Yes** — meets Round 02 bar: framing + Pareto + true \(\dim\mathfrak{g}\ge d\) controls + multi-seed win. TV-only polish alone would **not** merit `ACCEPT`; the **package** (equivalence proofs + Pareto + held-out MSE) does. |
| Enough for `ACCEPT`? | **No** — still MNIST surrogate only, \(N=20\), no CIFAR / full SHARD pipeline slice, held-out snapshot acc. collapses at selected \(\lambda\). |

**Novelty rubric: 3.5/5** (up from 3/5 in Round 02) — empirical Pareto and regime gate elevate an incremental prior to a usable Stage-3 variant.

---

## 6. Rubric scores (1–5)

| Criterion | Score | Notes |
|-----------|-------|-------|
| **Novelty** | **3.5** | Regime-gated TV polish + Pareto; not a new attack class. |
| **Soundness** | **4.5** | SHARD equivalence verified; trade-off explicit. |
| **Feasibility** | **4** | Runnable benchmark; ~15 min full sweep on MPS. |
| **Impact** | **3.5** | Real compressive input MSE/PSNR win; snapshot fit regresses by design. |
| **Evaluability** | **4** | Multi-seed Pareto, held-out seed, controls; plots missing from artifacts. |

---

## 7. Issues

### Critical

*None.* (`max_critical_issues: 0` satisfied.)

### Major

*None.* All Round 02 Major items addressed.

### Minor

1. **[Minor] Regenerate Pareto PNGs** — commit `round03_pareto_curves.png`, `round03_pareto_tradeoff.png` under `artifacts/`.
2. **[Minor] Scale** — \(N=20\) (not paper-scale 50+); acceptable for artifact acceptance; scale before NeurIPS-style claims.
3. **[Minor] Held-out snapshot acc.** — 0.00 at \(\lambda=0.005\) vs 0.08 tuning mean; report dual-metric sensitivity or relax/tighten \(\lambda\) selection for deployment scenarios.
4. **[Minor] CI regression** — unit test `tv_lbfgs=0` ≡ `level3_invert` on fixed snapshot (guard duplicated Adam drift).
5. **[Minor] CIFAR / end-to-end SHARD slice** — optional paper follow-up; not required to stop the research loop.

---

## 8. Round 02 fix checklist

| Round 02 Major issue | Round 03 status |
|----------------------|-----------------|
| Snapshot–input Pareto | **Fixed** — 5 \(\lambda\) × 3 seeds + trade-off metrics |
| Mislabeled reference regime | **Fixed** — `ablation_tv_off_14x14` |
| Missing \(\dim\mathfrak{g}\ge d\) control | **Fixed** — `overdetermined_14x14`, `square_mnist_28x28` |
| Jacobian-aware overclaim | **Fixed** — structure-aware polish framing |
| Pilot-only \(\lambda\) / scale | **Fixed** — held-out seed 31; \(N=20\) |

---

## 9. Conditions met vs `config.json`

| Requirement | Status |
|-------------|--------|
| `supervisor_verdicts` includes `ACCEPT_WITH_MINOR` | **Met** |
| `max_critical_issues: 0` | **Met** |
| `require_runnable_experiments` | **Met** |
| `require_baseline_comparison_vs_shard_level3` | **Met** |
| `require_tutorial_pdf` | **N/A** (tutorial.md present) |

---

## 10. Recommendation

| Question | Answer |
|----------|--------|
| Accept now? | **`ACCEPT_WITH_MINOR`** |
| Stop multi-round loop? | **Yes** (per `config.json` stop rules) |
| Is compressive win real? | **Yes** (multi-seed + held-out MSE) |
| Is TV polish enough for minor acceptance? | **Yes**, with Pareto + SHARD equivalence + honest trade-off framing |
| Full `ACCEPT` without further work? | **No** — CIFAR/pipeline + larger \(N\) + stronger held-out snapshot story |

---

*Supervisor audit: JSON spot-check and live SHARD≡JOLI equivalence verified 2026-05-20; full `benchmark_round03.py` re-run not repeated (prior log + metric audit).*
