# Round 11 — Supervisor Review (Phase 2, Minibatch Mandate)

## Verdict

**REVISE_MAJOR**

---

## Executive summary

Round 11 correctly diagnoses why vanilla TANGO fails under minibatch SGD in this simulator: the inversion uses `local_steps` in the gradient denominator while training applies `N/B` minibatch updates per local step with `1/B` normalization (`code/benchmark_round09.py:136-146`, `209-222`). **TANGO-MB** applies a deterministic rescaling `Δ ← Δ · (T / T_eff)` with `T_eff = T·(N/B)` before the unchanged Round-09 solve (`code/benchmark_round11.py:79-109`). Metrics are **computed, not injected**; log lines match `artifacts/round11_metrics.json` aggregates (verified for `minibatch_sgd`, 8 seeds). On the Phase-2 primary scenario, **TANGO-MB prototype MSE `0.0110` beats passive `0.7532` on every seed** (log `experiment_round11.log:4-58`). That satisfies the **numeric** `require_minibatch_primary_win` gate, but it is **not** a fundamental, theory-complete fix for realistic FL: the correction is exact only under **W₀=0 linearization**, within-step weight drift leaves a large approximate floor (`0.011` vs full-batch `0.00015`), count recovery **regresses** vs passive, and **STORM** is numerically identical to TANGO-MB. Recommend major revision: formalize Lemma MB-A (fix proof sketch), bound drift error, iterate for W₀≠0, broaden scenarios, and tie passive+MB baselines into claims before supervisor **ACCEPT**.

---

## Honesty / reproducibility audit

| Check | Finding | Reference |
|--------|---------|-----------|
| Hardcoded metrics | None found; JSON written from `run()` loop | `benchmark_round11.py:278-494` |
| Log ↔ JSON consistency | `minibatch_sgd` means match log-derived averages | log `:231-232` vs JSON `phase2_primary` |
| Primary scenario order | `minibatch_sgd` listed first in `SCENARIOS` | `benchmark_round11.py:53-54` |
| Per-seed primary win | TANGO-MB `<` passive on prototype MSE for seeds 3,7,11,19,23,29,31,37 | log `:4-58` |
| Vanilla failure control | Mean proto MSE `6.5877` (seed 37 outlier `36.97` inflates mean; MB still wins all seeds) | log `:52-54` |
| STORM vs TANGO-MB | Identical to machine precision on all logged rows | log e.g. `:5-6`, `:18-19` |
| Full-batch regression | `balanced_clean`: MB equals vanilla (`0.000151`) | log `:117-118`, JSON headlines |
| Passive+MB ablation | Scaling alone improves passive (`0.1505`) but not active MB (`0.0110`) | JSON `phase2_primary` |
| Intermediate gradients | Not observed (`observed_intermediate_batch_gradients: 0`) | JSON observation_model |
| Batch-order oracle | Shuffles use client seed only inside simulator; estimator uses public `N,B,T` only | `benchmark_round09.py:134-146`, `benchmark_round11.py:434-443` |

**TANGO-MB is not a metrics cheat**, but it **is** largely **bookkeeping on the existing first-order pipeline**: one scale factor then `tango_estimate_sums`. STORM at `ridge=0` re-implements the same denominator without adding identifiability.

---

## Feasibility & resources

- Benchmark reruns in ~0.7s; 8 seeds × 4 scenarios × 7 methods — adequate for regression, thin for publication.
- Simulator remains **synthetic hidden-feature logistic head** (`benchmark_round09.py`); no CNN / FedAvg / SHARD `level3_invert` cross in Round 11.
- `config.json` still requires `require_baseline_comparison_vs_shard` at project level — **not addressed** this round.

---

## Theory & assumptions

**What is sound (under stated assumptions).**

- At **W=0**, softmax logits depend only on bias; summing `M=N/B` disjoint minibatch gradients with `(π−y)/B` normalization yields a **deterministic** factor `N/B` per local step before weights move — consistent with `T_eff` in code (`benchmark_round11.py:79-89`, Remark MB-A in `paper/proofs.md:132-144`).
- Public `(η, T, B, N)` is standard FL metadata; no batch-incidence leakage in the estimator.

**What is not yet fundamental.**

1. **W₀≈0 only** — `minibatch_sgd` uses `init_weight_scale=0.0` (`benchmark_round11.py:54`). Nonzero-init stress (`0.0168` MB vs `7.45` passive) is encouraging but still one linearized correction, not iterative Jacobians.
2. **Within-step drift** — `local_terminal_update` updates `w` after every minibatch (`benchmark_round09.py:135-146`). Remark MB-A assumes frozen weights **within** the minibatch sum; residual proto MSE `~0.011` (~73× above full-batch exact tier) is the expected violation.
3. **“Stochastic gradients”** — Under W₀=0 the minibatch sum is **deterministic** (shuffle irrelevant to the sum at fixed W). The fix is **not** a general SGD-noise theory; it is **gradient-norm accounting** for PyTorch-style `1/B` scaling.
4. **Proof sketch gap** — In `paper/proofs.md:142`, the bias aggregation identity `(M/B)(N p - n_c) = (N/B)(N p - n_c)` is **algebraically inconsistent** unless stated symbols differ from \(M=N/B\). Needs a corrected derivation and explicit drift bound.
5. **Count metric** — On `minibatch_sgd`, passive count MAE `0.277` beats TANGO-MB `0.362` (JSON aggregates). Phase-2 headline uses prototype MSE only; dual-metric honesty is required in claims.

---

## Rubric scores (1–5)

| Criterion | Score | Note |
|-----------|------:|------|
| Novelty | 2 | Effective-step scaling is the right bugfix, not a new attack surface |
| Soundness | 3 | Empirically consistent; theory sketch incomplete, scope narrow |
| Feasibility | 4 | Runnable, fast, reproducible |
| Impact | 3 | Restores minibatch prototype tier in simulator; not yet deployable FL |
| Evaluability | 4 | Clear ablations (vanilla, passive, passive+MB); needs more scenarios |

---

## Issues

### Critical

| Tag | Issue |
|-----|--------|
| Critical | **Scope:** Fix is **Lemma A + divisor correction at W₀=0**, not general minibatch FL with drifting representations. Mandate asks for fundamental progress under realistic local training — partially met in simulator, not met for general FL. |
| Critical | **Theory:** Remark MB-A is a sketch with a **suspect bias summation identity**; no bound tying within-step drift to the `0.011` floor. |
| Critical | **Dual metric:** Count MAE **worse than passive** on primary `minibatch_sgd` while claiming Phase-2 readiness — risks over-claiming “attack fixed.” |

### Major

| Tag | Issue |
|-----|--------|
| Major | **STORM = TANGO-MB** numerically; second name adds no new tier. |
| Major | **passive_mb (`0.1505`)** shows much of the gain is **scaling**; active probing value must be quantified separately in prose. |
| Major | **Approximate tier gap:** minibatch MB `0.011` vs full-batch `0.00015` — correction does not restore “exact” tier. |
| Major | **Evidence breadth:** Only bundled synthetic scenarios; no SHARD baseline, noise, secure agg, or label skew this round. |

### Minor

| Tag | Issue |
|-----|--------|
| Minor | Vanilla mean dominated by seed 37 (`36.97`); report medians/IQR alongside means. |
| Minor | `config.json` `status: PHASE2_MINIBATCH_FIX_CANDIDATE` presupposes acceptance before supervisor sign-off. |

---

## Actionable suggestions (prioritized)

1. **Prove Lemma MB-A rigorously** — correct the bias sum, state when shuffle is irrelevant, and add **‖W_end − W₀‖** bound vs `T_eff` error (targets `0.011` floor).
2. **Report uncertainty** — per-seed table, median, and std for `minibatch_sgd`; flag outlier sensitivity on vanilla only.
3. **Iterative / nonzero W₀ tier** — even one Newton step or trajectory linearization; align `minibatch_nonzero_init` with theory scope.
4. **Disentangle scaling vs probing** — headline both `passive_mb` and `tango_mb`; do not imply active TANGO alone fixes minibatch.
5. **Merge or demote STORM** unless ridge / stacked moments show a measurable gain over scaled TANGO.
6. **Restore count recovery** or narrow Phase-2 claim to **prototype MSE only** with explicit limitation.
7. **Phase-2 exit experiments** — SHARD comparison, terminal noise, imbalanced minibatch, optional real FL head — before global `ACCEPT`.

---

## Phase 2 acceptance criteria (`config.json`)

| Criterion | Met? | Evidence |
|-----------|------|----------|
| `require_minibatch_primary_win` | **Yes (numeric)** | `tango_mb` `0.0110` < passive `0.7532` on `minibatch_sgd`; `phase2_primary_win: 1.0` |
| `forbid_primary_eval_on_fullbatch_only` | **Yes** | Primary scenario is `minibatch_sgd`; full-batch rows are regression only |
| `no_limits_section_as_substitute_for_fix` | **Yes** | Implemented `TANGO-MB` / `STORM`; limits document residual drift, not substitute |
| Supervisor `ACCEPT` + `max_critical_issues: 0` | **No** | This review: **3 Critical**, **4 Major** |
| `require_baseline_comparison_vs_shard` (project) | **No** | Not in Round 11 benchmark |
| Fundamental / rigorous fix (user mandate) | **Partial** | Correct divisor under W₀≈0; not sufficient for **ACCEPT** |

**Bottom line:** The **headline minibatch prototype win is real and reproducible**, but Phase 2 should remain **suspended** until theory, drift bounds, broader evaluation, and metric scope are upgraded. **Do not grant global ACCEPT on Round 11.**

---

## Return summary (for parent agent)

- **Verdict:** `REVISE_MAJOR`
- **Phase 2 config gates (`require_minibatch_*`):** primary numeric win **met**; full Phase 2 restoration / honest **ACCEPT** **not met** (critical theory/scope issues remain).
- **TANGO-MB:** Legitimate **effective-step correction**, not a fake metric — but **thin** relative to “fundamental fix” bar (≈ rescaling + unchanged Round-09 inversion under W₀≈0).
