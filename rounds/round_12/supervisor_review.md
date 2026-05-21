# Round 12 — Supervisor Review (Phase 2, Minibatch Mandate)

## Verdict

**REVISE_MAJOR**

---

## Executive summary

Round 12 is a **material response** to Round 11 **REVISE_MAJOR**: Lemma MB-A is corrected in `paper/proofs.md`, dual metrics are restored on the primary scenario, STORM is removed, median/IQR reporting is added, and seven scenarios replace four. Metrics are **computed from code** (`code/benchmark_round12.py` → `artifacts/round12_metrics.json`); log lines match aggregates (e.g. `logs/experiment_round12.log:514-516`).

On **`minibatch_sgd`**, **TANGO-MB** prototype MSE **0.0110** (median **0.0114**) still beats **passive** **0.7532** on every seed; count MAE **0.100** now beats passive **0.295**. That satisfies the **numeric** `require_minibatch_primary_win` gate.

Under the user mandate (**no bookkeeping-as-fix; fundamental minibatch solution**), the core mechanism remains **deterministic \(T_{\mathrm{eff}}=T\cdot(N/B)\) rescaling** plus the unchanged Round-09 linear inversion. Round 12 adds **heuristic layers** (single-round count pick, drift inflation, one Jacobian bias step) that improve reporting and some stress cases but **do not** close scope to general FL, **do not** restore the exact tier, and **distort** the scaling-vs-active ablation. **Phase 2 numeric gates pass; holistic ACCEPT does not.**

---

## Round 11 issue closure

| Round 11 issue | Status | Round 12 evidence |
|----------------|--------|-------------------|
| **Critical — Scope** (W₀≈0 / synthetic only) | **Open** | Primary still `init_weight_scale=0`; frozen MLP is a toy head on fixed \(\phi(x)\), not representation drift during training. No SHARD `level3_invert`. |
| **Critical — Theory** (Remark MB-A algebra) | **Closed** | Lemma MB-A partition-sum proof; explicit R11 error note (`paper/proofs.md:132-158`). |
| **Critical — Dual metric** (count worse than passive) | **Closed** | Primary: `tango_mb` count MAE **0.100** vs passive **0.295** (`artifacts/round12_metrics.json` `phase2_primary`). |
| **Major — STORM = TANGO-MB** | **Closed** | STORM removed; `tango_mb_drift2`, `stack_mb_ridge` added. |
| **Major — Quantify scaling vs active** | **Partial / misleading** | See § passive+MB below; headline **~26×** overstates vs R11-comparable **~14×**. |
| **Major — Approximate tier gap** | **Open** | Primary MB **0.011** vs `balanced_clean` **0.00015** (~73×). |
| **Major — Evidence breadth** | **Partial** | +label noise, terminal noise, frozen MLP, nonzero init; still synthetic; no SHARD. |
| **Minor — Median/IQR** | **Closed** | `aggregate_robust` in `benchmark_round12.py:303-315`. |

**Critical count for `max_critical_issues: 0`:** **1** remains (external validity / general FL scope). **Do not grant global ACCEPT.**

---

## Targeted audits (Round 12 checklist)

### 1. Proof algebra (Lemma MB-A)

**Accept.** The corrected derivation sums \(M=N/B\) disjoint minibatches at fixed \(W_0=0\):

\[
\sum_{b=1}^{M} \frac{\partial \mathcal{L}_b}{\partial b_c}
= \frac{1}{B}\sum_{i=1}^{N}(p_c - \mathbf{1}[y_i=c])
= \frac{1}{B}(N p_c - n_c),
\]

then relates terminal scaling to \(T_{\mathrm{eff}} = T(N/B)\) via the implementation’s `effective_gradient_steps` / `scale_terminal_deltas` (`code/benchmark_round12.py:106-118`). This fixes the Round 11 double-count \((M/B)\) error called out in `paper/proofs.md:158`.

**Caveat (unchanged):** The identity is **exact at \(W_0=0\) within one pass**; sequential minibatch updates violate frozen-\(W\) during the pass — handled only loosely via MB-B, not closed analytically to **0.011** MSE.

### 2. Drift bound — real or hand-wavy?

**Mixed.**

- **Real (first-order):** Lemma MB-B gives a standard telescoping bound \(\|W_M-W_0\|_F \le \eta (N/B) G_{\max}\) and Lipschitz bias-gradient perturbation (`paper/proofs.md:162-175`). Code measures drift: mean **0.1115**, std **0.0039** (`lemma_mb_b_empirical` in JSON; log `:516`).
- **Hand-wavy / inconsistent documentation:** `paper/proofs.md:177` claims mean drift **≈ 0.42** — **wrong** vs computed **0.112**. The doc also claims drift2 **reduces** median primary MSE vs TANGO-MB; **empirically false** on primary: **drift2 0.0247** > **MB 0.0110** (JSON `phase2_primary`, log per-seed e.g. `:5-6`).
- **`tango_mb_drift2`:** Ad hoc `inflate = 1 + 0.12 * drift / (η · batches_per_step)` (`benchmark_round12.py:194-198`) — not derived from Lemma MB-B; **secondary only**.

**Bottom line:** Drift is **measured honestly**; the **floor is not rigorously predicted** from drift magnitude, and the default correction path **does not improve** the primary estimator.

### 3. Count fix — principled or cherry-picked?

**Heuristic with a defensible story, not a theorem.**

`estimate_counts_mb` (`benchmark_round12.py:122-134`) picks the probe round **closest to uniform** \(\arg\min_r \|p^{(r)} - \mathbf{1}/C\|\), then applies Lemma B on **scaled** bias deltas. Rationale (least aggressive active bias → less moment corruption) is plausible but **not** joint LS over rounds; it is **one-round selection** tuned to fix R11 count regression.

**Empirical:** Primary count MAE **0.100** vs passive **0.295** — **honest win** on that metric. **Side effect:** wiring this into `tango_mb_estimate_sums` for **all** methods changed **`passive_mb`** behavior vs Round 11 (see below).

### 4. passive+MB regression (0.28 vs 0.15 in R11) — honest?

**Not comparable without disclosure — headline is misleading.**

| Ablation | Round 11 proto MSE (primary) | Round 12 proto MSE (primary) | Code path |
|----------|------------------------------|------------------------------|-----------|
| `passive_mb` | **0.1505** | **0.2839** | R11: `scale` + `tango_estimate_sums` (mean counts, `local_steps` divisor on scaled Δ). R12: `tango_mb_estimate_sums` → **`estimate_counts_mb`** + same stack (`benchmark_round11.py:195-197` vs `benchmark_round12.py:365-367`). |
| R11-comparable scaling-only | — | **~0.150** | `stack_mb_ridge` mean **0.1500** on primary (passive probes, \(T_{\mathrm{eff}}\), ridge) — JSON `scenario_method_headlines` for `minibatch_sgd`. |

**TANGO-MB active** unchanged **0.0110** vs R11. The **~26×** `active_probe_gain_over_scaling_only_x` uses the **degraded** R12 `passive_mb` denominator, not the R11 scaling-only ablation. **Honest ratios on primary:**

- Active vs passive (vanilla divisor): **~68×** (real).
- Active vs **R11-style** scaling-only (~**0.15**): **~14×** — most prototype gain still from **\(T_{\mathrm{eff}}\)**, not probing alone.
- Remaining **~2×** (0.15 → 0.011) from **full-rank active probes** + count/LS path.

Researchers must **restore a frozen `passive_mb_scale_only`** ablation (scale + `tango_estimate_sums` only) in metrics and prose.

### 5. W₀≠0 iter — real improvement?

**Yes, modest, scope-aligned.**

`minibatch_nonzero_init` (`init_weight_scale=0.18`): **tango_mb** **0.0168** → **tango_mb_iter** **0.0131** mean proto MSE (~22% relative); passive **4.49**. At **`minibatch_sgd`** (\(W_0=0\)), iter **equals** MB (log `:5-7`, identical counts) — correct degeneracy.

**Limits:** One bias Jacobian step with fixed \(\alpha=0.15\); **iter count MAE often worse** on nonzero-init seeds (e.g. log `:86-88` seed 7). Not iterative FL inversion; does not remove primary **W₀=0** scope.

### 6. Still synthetic-only?

**Yes.** All scenarios use `benchmark_round09.py` / `benchmark_round10.py` simulator (24-d frozen MLP features). **`require_baseline_comparison_vs_shard`** in `config.json` remains **unmet**. No CNN / FedAvg / vendored SHARD cross-run this round.

---

## Honesty / reproducibility

| Check | Finding |
|--------|---------|
| Injected metrics | None; JSON from `run()` loop (`benchmark_round12.py:448-744`) |
| Log ↔ JSON | Primary means match (`log:514-516` = `phase2_primary`) |
| Per-seed primary win | TANGO-MB `<` passive proto MSE, seeds 3–37 (`log:4-74`) |
| drift2 vs MB on primary | drift2 **worse** every seed (e.g. `log:5-6`) |
| proofs.md drift2 claim | **Contradicts** metrics |
| `passive_mb` continuity | **Broken** vs R11; biases “active vs scaling” narrative |

---

## Rubric (1–5)

| Criterion | Score | Note |
|-----------|------:|------|
| Novelty | 2 | Still effective-step + linear inversion; extensions are heuristics |
| Soundness | 4 | MB-A fixed; MB-B/iter bounded narratively; doc errors and drift2 hurt |
| Feasibility | 4 | Fast, reproducible, richer scenarios |
| Impact | 3 | Strong simulator minibatch tier; not deployable FL |
| Evaluability | 4 | Good ablations if passive_mb baseline is fixed |

---

## Issues (Round 12)

### Critical

| Tag | Issue |
|-----|--------|
| Critical | **Scope / fundamentality:** Phase-2 **primary** remains **synthetic linear head, \(W_0=0\)**. \(T_{\mathrm{eff}}\) scaling is the right **gradient accounting** fix, not a **general minibatch FL** identifiability result. User mandate for a **fundamental** solution is **not met** for ACCEPT. |

### Major

| Tag | Issue |
|-----|--------|
| Major | **`passive_mb` ablation regression** and **~26×** headline — must report R11-comparable scaling-only (~**0.15**) alongside current `passive_mb`. |
| Major | **Approximate tier gap** persists: **0.011** vs full-batch **0.00015**; drift bound does not close it; drift2 **fails** on primary. |
| Major | **`paper/proofs.md` factual errors** — drift mean **0.42** vs **0.112**; drift2 “reduces median” vs data. |
| Major | **SHARD baseline** still missing (`require_baseline_comparison_vs_shard`). |
| Major | **Frozen MLP** minibatch: MB **0.126** vs passive **1.038** — win holds, but floor rises; nonlinear head stress is partial only. |

### Minor

| Tag | Issue |
|-----|--------|
| Minor | `estimate_counts_mb` uses `local_steps` in \(g\) denominator after scaling — document equivalence to \(T_{\mathrm{eff}}\) explicitly. |
| Minor | `tango_mb_iter` count MAE unstable on some nonzero-init seeds. |
| Minor | `config.json` `status: PHASE2_ROUND12_REVISION` — acceptable; do not auto-promote to ACCEPT. |

---

## Phase 2 acceptance criteria (`config.json`)

| Criterion | Met? | Evidence |
|-----------|------|----------|
| `require_minibatch_primary_win` | **Yes (numeric)** | `tango_mb` **0.0110** < passive **0.7532**; `phase2_primary_win: 1.0` |
| `forbid_primary_eval_on_fullbatch_only` | **Yes** | Primary = `minibatch_sgd` |
| `no_limits_section_as_substitute_for_fix` | **Yes** | Implemented estimators; limits do not replace code |
| `require_baseline_comparison_vs_shard` | **No** | Not in Round 12 |
| Supervisor **ACCEPT** + `max_critical_issues: 0` | **No** | **1 Critical**, **5 Major** |
| **Fundamental fix** (user mandate) | **Partial** | Divisor + theory upgrade; core still rescale + lstsq under narrow assumptions |

---

## Actionable requirements (Round 13)

1. **Fix `passive_mb_scale_only`** — `scale_terminal_deltas` + `tango_estimate_sums` (R11 path); report **both** in `phase2_primary` and prose.
2. **Correct `paper/proofs.md`** — drift empirical **0.112**; remove or fix false drift2 superiority claim.
3. **Either improve drift2 with theory-linked correction or demote** — primary estimator stays **`tango_mb`**.
4. **SHARD `level3_invert` comparison** on matched terminal-observation settings (project gate).
5. **Tighten count estimator** — multi-round joint fit or proof for uniform-round selection; avoid silent coupling into `passive_mb`.
6. **Phase-2 ACCEPT bar:** demonstrate scaling-only vs active **with frozen ablations**, SHARD cross, and explicit claim scope (prototype + count on synthetic primary; residual drift floor documented).

---

## Return summary

- **Verdict:** `REVISE_MAJOR`
- **Phase 2 config gates (`require_minibatch_*`):** **Met** (numeric primary win + dual metric on primary).
- **Phase 2 holistic ACCEPT:** **Not met** — scope/fundamentality critical remains; passive_mb methodology breaks scaling narrative; SHARD open.
- **Round 11 REVISE_MAJOR:** **Substantially addressed** on theory sketch, count metric, STORM, reporting, scenarios — **not sufficient** for supervisor **ACCEPT** under strict user standard.
