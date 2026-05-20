# Supervisor Review — Round 01

**Run folder:** `research-artifacts/novel-inversion-vs-shard/`  
**Reviewed:** `researcher_proposal.md`, `revision_log.md`, `code/lapin_invert.py`, `code/benchmark_round01.py`, `artifacts/round01_metrics.json`, `logs/experiment_round01.log`  
**Baseline reference:** `supplementary_materials/code/shard_sim/attacker.py` (`level3_invert`, L693–832)

---

## 1. Verdict

**`REVISE_MAJOR`**

Round 01 is **honest and reproducible**, but it **does not meet** `config.json` acceptance: LAPIN is not experimentally validated against SHARD L3 (quality metrics are substantially worse), and novelty relative to SHARD is **incremental** (same objective, different first-order vs second-order multi-start solver) without a compensating theoretical or empirical win.

**Not eligible:** `ACCEPT`, `ACCEPT_WITH_MINOR` (failed validation + blocking Critical issues below).

---

## 2. Executive summary

The scientist framed Stage 3 correctly as per-snapshot nonlinear least squares on \(\cos(Wx+b)\), proposed **LAPIN** (multi-start projected Gauss–Newton + L-BFGS, no Adam), and ran a runnable MNIST micro-benchmark (\(N{=}12\), \(14{\times}14\), `dim_g=160`). I re-ran `code/benchmark_round01.py`; metrics match `artifacts/round01_metrics.json` exactly (no hardcoded results).

**SHARD L3 (benchmark variant)** dominates: input MSE **0.0127** vs **0.0877** (~6.9×), PSNR **19.0 dB** vs **10.6 dB**, snapshot matching **58%** vs **0%**, mean snapshot residual **0.038** vs **0.267**. Runtime is similar (~4.9 s vs ~4.1 s), so LAPIN is **not** a Pareto improvement.

The proposal’s honesty about failure is commendable; the research direction may still be viable, but Round 2 must **fix the benchmark protocol**, **diagnose LAPIN underperformance**, and **demonstrate a measurable win** (quality or clearly stated trade-off) against a faithful SHARD L3 baseline before any acceptance verdict.

---

## 3. Honesty / reproducibility audit

| Check | Finding |
|-------|---------|
| Hardcoded metrics | **Pass.** `benchmark_round01.py` computes MSE/PSNR/runtime from arrays; JSON is written via `json.dumps(results)` (L279–280). No injected constants. |
| Re-run | **Pass.** Supervisor re-run (2026-05-20): `input_mse`, `input_psnr_db`, `snapshot_match_acc`, `mean_snapshot_residual` match stored JSON to full precision; runtime within ~1%. |
| Claims vs code | **Mostly pass.** LAPIN matches proposal (5 lstsq seeds + 48 random, 25 GN steps, L-BFGS polish) in `lapin_invert.py` L114–128. |
| SHARD baseline fidelity | **Partial.** Benchmark defines inline `level3_fast` (`benchmark_round01.py` L108–183) instead of calling `ShardAttacker.level3_invert`. Structure matches official L3 (same seeds, batch Adam, best-start L-BFGS) but **differs from paper code**: `n_batch=120` vs up to **500**, Adam **800** vs **2000** steps, L-BFGS **300** vs **500** iter (`attacker.py` L738–739, L788, L814). This **weakens** SHARD relative to the repo default; LAPIN still loses badly—so the headline result is not an artifact of an unfairly strong SHARD. |
| Plot artifacts | **Gap.** Proposal cites `round01_reconstruction_grid.png` and `round01_metrics_bar.png`; `artifacts/` currently contains only `round01_metrics.json`. Plots may have been omitted from the run folder—regenerate and commit in Round 2. |
| Seeds | Single global `seed=7`; per-snapshot restart seeds `42+i` aligned between SHARD and LAPIN—good for paired comparison, insufficient for variance reporting. |

**Jacobian / objective sanity (LAPIN):** Residual \(r=\cos(Wx+b)-s\), \(J=-\mathrm{diag}(\sin z)\,W\) (`lapin_invert.py` L75–85) is consistent with the SHARD objective. Polish uses the same \(\|\cos(Wx+b)-s\|_2^2\) as `level3_invert` L-BFGS phase (`attacker.py` L820–825). No sign error found on audit.

---

## 4. Feasibility & resources

- **Runnable:** Yes—documented venv + `benchmark_round01.py`; ~11 s end-to-end on supervisor machine.
- **Scope:** Micro-benchmark only (\(N{=}12\), one `dim_g`, one resize). Feasible to scale; not yet done.
- **Compute fairness:** SHARD explores **120** trajectories in **parallel** Adam; LAPIN runs **53** **sequential** GN trajectories. Wall-clock parity (~4 s) therefore **hides** unequal search width and different step budgets (800 Adam steps × 120 vs 25 GN × 53). Round 2 must report **matched restart counts** and/or **equal wall-clock** sweeps.
- **Official L3 at full budget:** `n_batch` up to 500 and 2000 Adam steps will increase SHARD quality and runtime; any claim of beating SHARD must specify which baseline configuration.

---

## 5. Theory & assumptions

**Strengths**

- Problem formulation (oracle snapshots after Stages 1–2) matches SHARD Stage 3.
- GN exploits analytic curvature of \(\cos(\cdot)\); plausible when underdetermined or well-conditioned in latent space.

**Weaknesses**

- **No convergence or identifiability analysis** for projected GN on a non-convex, box-constrained NLS with RFF encoding. GN can stall when \(J^\top J\) is ill-conditioned; early exit on failed backtracking (`lapin_invert.py` L93–94) is observed in code path and may explain weak minima.
- **Novelty vs prior art** is undersold in the proposal and **not supported by results**: swapping Adam for GN on the **same** SHARD snapshot objective is an engineering variant, not a new attack model (still not gradient matching à la DLG). Distinction from Heredge et al. is asserted but not tested on a VQC setting.
- **Hypothesis for Round 2** (“GN shines when \(d \ll \dim\mathfrak{g}\)”) is reasonable but **untested**; current setting is \(d=196 > \dim\mathfrak{g}=160\) (overdetermined in input dimension).

**Assumptions (acceptable for Round 1, must relax later)**

- Perfect Stages 1–2 (oracle snapshots).
- Noiseless encoding; no end-to-end SHARD pipeline.

---

## 6. Rubric scores (1–5)

| Criterion | Score | Notes |
|-----------|-------|-------|
| **Novelty** | **2** | Structured GN + same seeds as SHARD is a modest optimizer change; no new leakage surface or information model. |
| **Soundness** | **3** | Implementation is coherent; empirical result is negative, not fraudulent. |
| **Feasibility** | **4** | Code and benchmark are lightweight and extensible. |
| **Impact** | **1** | No improvement on MSE/PSNR/snapshot match; no deployment story. |
| **Evaluability** | **3** | Metrics are standard and computed; baseline is approximate; sample size and variance missing. |

---

## 7. Issues

### Critical

1. **[Critical] Acceptance criterion failed — no experimental win vs SHARD L3.** LAPIN is worse on every primary quality metric (MSE, PSNR, snapshot matching, snapshot residual). `config.json` requires validated gains for `ACCEPT` / `ACCEPT_WITH_MINOR`.
2. **[Critical] Novelty not demonstrated in practice.** A Stage-3 replacement must show **why** GN-in-input-space should replace batch Adam (quality, speed at equal quality, or theory). Round 01 shows the opposite on quality at similar runtime.

### Major

3. **[Major] Baseline is not the canonical API.** Benchmark should call `ShardAttacker.level3_invert` (with documented `device`, and a “paper” config row: default `n_batch`/steps) alongside the reduced laptop variant.
4. **[Major] Unequal optimization budgets.** 120×800 Adam parallel vs 53×25 sequential GN; L-BFGS iterations differ from official (300 vs 500). Comparisons are not budget-matched.
5. **[Major] Evaluation scale too small.** \(N{=}12\), single seed, one geometry—no error bars, no CIFAR/larger \(d\), no `dim_g < d` regime promised in Round 2 plan.
6. **[Major] LAPIN fails at snapshot fitting before input semantics.** `mean_snapshot_residual` 0.267 vs 0.038 and 0% snapshot match imply the method is not even solving the stated Stage-3 objective well—debug before claiming curvature exploitation.

### Minor

7. **[Minor] Missing plot artifacts** despite proposal references.
8. **[Minor] Both methods have `pixel_match_acc_0.1 = 0`**—metric may be too strict or uninformative; clarify or adjust threshold.
9. **[Minor] Proposal rounds metrics** (19.0 dB vs 18.978 in JSON)—use consistent precision in tables.

---

## 8. Actionable suggestions (prioritized)

1. **Establish a baseline ladder** in `benchmark_round02.py`: (a) `level3_invert` defaults, (b) reduced laptop config (current), (c) matched-budget LAPIN (same restart count as SHARD, report wall-clock). Require LAPIN to beat (a) or (b) on **input MSE** or document a **strict** Pareto point (e.g. ≥2× speed at ≤10% MSE degradation).
2. **Diagnose LAPIN failure mode:** Log per-sample snapshot residual after GN vs after L-BFGS; sweep `damping`, `max_gn`, and line-search exit policy; try GN-only polish from SHARD’s best Adam start (ablation: is GN harmful or just under-budget?).
3. **Test the stated Round 2 hypothesis:** `dim_g < d` compressive settings and larger \(N\) with Hungarian metrics + 3–5 seeds (mean ± std).
4. **Hybrid pipeline (scientist’s plan):** LAPIN best-start → long L-BFGS only is fine **only if** it beats SHARD at equal L-BFGS budget; otherwise it is not a Stage-3 replacement.
5. **Regenerate and store** reconstruction grids and bar charts under `artifacts/` for supervisor audit.
6. **Reframe contribution** unless quality wins: e.g. “Adam-free Stage 3” is only interesting if cheaper **at equal MSE**—otherwise report as negative result and pivot (joint multi-snapshot inversion, trust-region GN, Levenberg–Marquardt on GPU batch).

---

## 9. Conditions for upgraded verdict (Round 2+)

| Target verdict | Requirements |
|----------------|--------------|
| `REVISE_MINOR` | LAPIN ≥ SHARD on **input MSE or PSNR** on **official** `level3_invert` at laptop settings, on ≥2 configurations, with matched or clearly justified budgets; multi-seed; zero Critical issues. |
| `ACCEPT_WITH_MINOR` | Above + novelty paragraph defensible vs DLG/SHARD; remaining issues only Minor (plots, prose). |
| `ACCEPT` | Sustained wins vs full-default SHARD L3 on paper-relevant settings (MNIST + at least one larger-scale path), with runnable scripts and no Critical/Major evaluation gaps. |

---

## 10. Top 3 required fixes for scientist (Round 2)

1. **Beat or fairly trade off against canonical SHARD L3** — use `ShardAttacker.level3_invert` as primary baseline; match restart/wall-clock budgets; report MSE/PSNR/snapshot residual with multiple seeds.
2. **Fix LAPIN snapshot fit** — explain and remedy why `mean_snapshot_residual` is ~7× SHARD and snapshot match is 0%; tune GN (damping, iterations, backtracking) before claiming method readiness.
3. **Validate the compressive regime** — run `dim_g < d` (and larger \(N\)) where GN is hypothesized to help; if LAPIN still loses, document and pivot (joint inversion or abandon GN-first narrative).

---

*Supervisor audit: metrics re-run verified 2026-05-20; no code changes made by supervisor.*
