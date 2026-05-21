# Round 13 — Supervisor Review (Phase 2, New Directions + Honest Ablations)

## Verdict

**REVISE_MINOR**

---

## Executive summary

Round 13 is a **disciplined closure** of Round 12 **REVISE_MAJOR** methodology debt: it implements **four genuinely new estimators** (JOINT, COUPLED, DOPT, trajectory midpoint), restores the **R11 honest scaling ablation** (`passive_mb_scale_only`), corrects **Lemma MB-B** documentation (drift **0.112**, not **0.42**), and records **negative results** without promoting failed paths. Primary **`minibatch_sgd`** metrics are **bit-identical** to Round 12 for **TANGO-MB** and passive baselines; the Phase-2 numeric gate remains **met**.

Under the user mandate (**no limits-as-fix; credit honest negatives when claims are scoped**), Round 13 **does** honestly test harder inversion directions and **does** establish **TANGO-MB** as the defensible **scoped** fix: correct \(1/B\) terminal accounting at \(W_0=0\) (Lemma MB-A) plus active probing, with **explicit empirical failure** of joint/coupled/trajectory estimators (~**25×** worse prototype MSE). That is sufficient to **downgrade** from Round 12’s **REVISE_MAJOR** to **REVISE_MINOR**.

**Phase 2 holistic ACCEPT remains blocked** by project config: **`require_baseline_comparison_vs_shard`** is still unmet, and claims are still **synthetic / \(W_0\approx 0\)** — not deployed CNN FL. Those are **external-validation** gaps, not evidence that Round 13 failed to execute its mandate.

---

## Decision matrix (supervisor questions)

| # | Question | Answer |
|---|----------|--------|
| 1 | Did Round 13 honestly test **new fundamental directions** (not limits-as-fix)? | **Yes.** Four implemented estimators with full metrics; negative outcomes reported. `passive_mb_scale_only` is an **ablation repair**, not masquerading as a new method. |
| 2 | Is the JOINT/COUPLED/DOPT negative result enough for **REVISE_MINOR** vs **REVISE_MAJOR**? | **Yes → REVISE_MINOR.** R12’s majors on passive ablation, proof errors, and “try joint/coupled” are **closed**; remaining work is SHARD + scope, not another estimator pivot. |
| 3 | Is **TANGO-MB** the defensible **scoped** fundamental fix with explicit non-identifiability of harder estimators? | **Yes**, with disciplined wording: **fundamental for Lemma MB-A regime** (\(W_0=0\), first-order terminal, synthetic linear/frozen head); **not** a general minibatch-FL identifiability theorem. JOINT/COUPLED/trajectory **document** that harder stacked inversion **does not** close the within-step gap. |
| 4 | **Phase 2 ACCEPT** now, or **REVISE_MAJOR** until SHARD + deployed FL? | **Neither ACCEPT nor REVISE_MAJOR.** **REVISE_MINOR** until SHARD `level3_invert` cross (config gate) and explicit deployed-FL / representation-drift scope (user bar). Numeric `require_minibatch_primary_win` is **met** but insufficient alone. |

---

## Round 12 requirements → Round 13 closure

| Round 12 requirement | Round 13 action | Outcome |
|----------------------|-----------------|---------|
| Fix `passive_mb_scale_only` | `passive_mb_scale_only_estimate_sums` in `code/benchmark_round13.py:153-161` | Primary mean **0.151**, median **0.140** — matches R11 ~**0.15** (`artifacts/round13_metrics.json` `phase2_primary`) |
| Correct `paper/proofs.md` drift | **0.112**; drift2 demoted | Aligns with `lemma_mb_b_empirical` in JSON/log |
| New fundamental directions (≥2) | JOINT, COUPLED, DOPT, trajectory midpoint | **All fail** vs TANGO-MB; documented in `paper/proofs.md` Lemma MB-JOINT |
| Keep `minibatch_sgd` primary + median/IQR | Unchanged | Gate **met**; robust aggregates retained |
| SHARD `level3_invert` cross | **Not done** | Still **Major** blocker |
| Demote drift2 / fix false superiority | Not in R13 benchmark | **Closed** |

---

## Primary evidence (`minibatch_sgd`, 8 seeds)

Source: `artifacts/round13_metrics.json` `phase2_primary`, `logs/experiment_round13.log:626-628`.

| Method | Proto MSE mean | Proto MSE median | Count MAE mean |
|--------|---------------:|-----------------:|---------------:|
| passive_multi_round | 0.753 | 0.751 | 0.295 |
| **passive_mb_scale_only** | **0.151** | **0.140** | (via vanilla path) |
| passive_mb_r12_coupled | 0.151 | 0.140 | — |
| **tango_mb** | **0.011** | **0.011** | **0.100** |
| tango_joint | 0.283 | 0.280 | 0.195 |
| tango_coupled | 0.272 | 0.276 | — |
| tango_dopt | 0.283 | — | — |
| tango_trajectory_midpoint | 0.272 | — | — |

- **`phase2_primary_win`:** 1.0 (TANGO-MB < passive on every seed in log).
- **`any_new_direction_beats_tango_mb`:** 0.0; least bad = **coupled** (**0.272**).
- **Honest active vs scaling:** **~13.7×** (`active_over_r11_scaling_only_x`), not R12’s **~26×** using degraded coupled passive_mb **0.284**.
- **R12 → R13:** `tango_mb` mean/median **unchanged** (0.0110 / 0.0114).

**Interpretation:** Scaling-only fixes **~5×** vs vanilla passive (0.753 → 0.151); active probing + MB count path adds **~14×** over scaling-only (0.151 → 0.011). The **~68×** vs vanilla passive headline is real but **conflates** both effects; prose must lead with the **two-step** decomposition (now in `paper/draft.md` §1.1).

---

## Audit: “New directions” vs limits-as-fix

### What counts as honest new-direction testing

| Estimator | Mechanism (beyond \(T_{\mathrm{eff}}\) scale) | Fundamental? |
|-----------|-----------------------------------------------|----------------|
| **TANGO-JOINT** | Weighted LS on stacked bias moments for \(n_c\); joint ridge on \((S_c, n_c)\) weight block | **Yes** — tests multi-round joint inversion |
| **TANGO-COUPLED** | 3-iter fixed point: joint → \(\hat W\) → bias Jacobian correction → joint | **Yes** — tests drift–moment coupling |
| **TANGO-DOPT** | Same as JOINT with \(w_r \propto 1/\|p^{(r)}-\mathbf{1}/C\|\) | **Intended** — but see § implementation gap |
| **trajectory_midpoint** | \(W_{\mathrm{mid}}=(W_0+W_M)/2\) bias correction + joint invert | **Yes** — tests intra-step trajectory |
| **passive_mb_scale_only** | R11 path only | **Ablation**, not a new attack |

All new directions share **Lemma MB-A scaling** then add structure; they are **not** “limits section only.” Failures are **large** (~0.27–0.28 vs 0.011), not marginal — a **credible negative result**.

### Lemma MB-JOINT (negative result value)

Documented in `paper/proofs.md:202-210`: aggressive rounds break a **shared-\(n_c\)** linear bias model; stacking without round exclusion **harms** counts (JOINT count MAE **0.195** vs MB **0.100**). This **supports** the R12 heuristic (single least-aggressive round for counts) and **rules out** promoting JOINT/COUPLED as primary — exactly the disciplined claim the user bar rewards.

**Why REVISE_MINOR, not REVISE_MAJOR:** Round 12 asked to *try* these directions and fix ablations; Round 13 did. **REVISE_MAJOR** would imply the round failed its charter or reintroduced dishonest metrics — it did not.

---

## TANGO-MB as scoped fundamental fix

**Defensible claim (use in prose):**

> Under **terminal-only** observation, **linear head**, **\(W_0=0\)**, and **PyTorch-style \(1/B\)** minibatch normalization, **TANGO-MB** applies the **correct effective gradient step count** \(T_{\mathrm{eff}}=T\cdot(N/B)\) (Lemma MB-A) and **active multi-round probing** for full-rank weight inversion, with **uniform-round-adjacent count selection** (`estimate_counts_mb`). Residual prototype error (~**0.01**) correlates with **within-step weight drift** (Lemma MB-B; mean \(\|W_M-W_0\|_F \approx 0.112\)) but is **not** closed by joint/coupled/trajectory corrections in this simulator.

**Non-identifiability of harder estimators (explicit):**

- JOINT/COUPLED/trajectory: **~25×** higher prototype MSE on primary; COUPLED often **inflates count MAE** (e.g. log seed 7: coupled count MAE **1.60** vs MB **0.15**).
- **Theorem 2** individual recovery remains impossible; approximate tier still **~73×** above exact (`balanced_clean` TANGO-MB **1.5×10⁻⁴**).

**Not defensible (do not claim):**

- “General minibatch FL solved.”
- “Joint inversion closes stochastic within-step gap.”
- “D-opt weighting helps” without fixing the JOINT/DOPT duplicate (below).

---

## Honesty / reproducibility

| Check | Finding |
|--------|---------|
| Metrics from code | `code/benchmark_round13.py` → `artifacts/round13_metrics.json`; runtime **~1.3s** |
| Log ↔ JSON | `log:626-628` matches `phase2_primary` |
| R12 primary unchanged | Confirms no cherry-picked regression on winner |
| `passive_mb_r12_coupled` vs `scale_only` | **Identical** 0.151 on primary (passive probes; coupled count path does not change proto MSE here) |
| JOINT vs DOPT | **Identical** implementations (`benchmark_round13.py:258-278` both call `round_weights_dopt`) — **H4 not independently tested** |
| drift2 false claim | **Removed** from primary path |
| Injected metrics | None observed |

---

## Issues

### Critical (unchanged from Round 12 — blocks global ACCEPT)

| Tag | Issue |
|-----|--------|
| Critical | **External validity:** Primary remains **synthetic**, **\(W_0=0\)**, frozen-feature / toy MLP. No **deployed CNN FL**, no **representation drift during training**. `require_baseline_comparison_vs_shard` **unmet**. |

**Critical count for `max_critical_issues: 0`:** **1** — same scope class as R12, not worsened by R13.

### Major (reduced vs Round 12)

| Tag | Issue | Status |
|-----|--------|--------|
| Major | **SHARD `level3_invert` baseline** | **Open** — config `baseline_code_path`; Round 14+ |
| Major | **Approximate tier gap** (0.011 vs **1.5×10⁻⁴**) | **Open** — documented; joint/coupled do not close |
| Major | **passive_mb / scaling narrative** | **Closed** — `passive_mb_scale_only` + honest **~14×** |
| Major | **proofs.md factual errors (drift, drift2)** | **Closed** |
| Major | **Frozen MLP stress** | **Partial** — MB **0.126** still beats passive **1.038** on `frozen_mlp_minibatch`; joint **~0.58** fails |

### Minor

| Tag | Issue |
|-----|--------|
| Minor | **JOINT ≡ DOPT in code** — relabel or add uniform-weight JOINT for a fair H4 test |
| Minor | `tango_joint_estimate_sums` docstring says “joint counts” but always applies D-opt weights |
| Minor | COUPLED count MAE unstable on some seeds — report as secondary metric only |
| Minor | `paper/draft.md` §3 still says “Round 12 headlines” — update pointer to Round 13 JSON |

---

## Rubric (1–5)

| Criterion | Score | Note |
|-----------|------:|------|
| Novelty | 3 | Negative results on joint/coupled are **informative**; winner unchanged |
| Soundness | 4 | MB-A/B/JOINT lemmas aligned with code; JOINT/DOPT duplicate hurts |
| Feasibility | 5 | Fast, full matrix of methods × scenarios |
| Impact | 3 | Strong scoped simulator story; SHARD/deployed FL still missing |
| Evaluability | 5 | Excellent ablation hygiene this round |

---

## Phase 2 acceptance criteria (`config.json`)

| Criterion | Met? | Evidence |
|-----------|------|----------|
| `require_minibatch_primary_win` | **Yes** | `tango_mb` **0.011** < passive **0.753** |
| `forbid_primary_eval_on_fullbatch_only` | **Yes** | Primary = `minibatch_sgd` |
| `no_limits_section_as_substitute_for_fix` | **Yes** | Code + metrics for all estimators |
| `require_baseline_comparison_vs_shard` | **No** | Not in Round 13 |
| `require_stage2_or_full_method_improvement` | **Partial** | Terminal-tier improvement vs SHARD not demonstrated |
| `max_critical_issues: 0` | **No** | Scope critical remains |
| Supervisor **ACCEPT** | **No** | SHARD + deployed-FL scope |
| User **fundamental fix** bar | **Partial (scoped)** | MB-A physics + honest negatives; not general FL |

---

## Actionable requirements (Round 14+)

1. **SHARD cross** — `vendor/shard_sim/attacker.py` `level3_invert` on matched terminal-only settings; table in `paper/draft.md`.
2. **Deployed / CNN FL slice** — or explicit permanent scope fence in abstract (if infeasible this phase).
3. **Fix JOINT vs DOPT experiment** — uniform-round JOINT baseline vs D-opt-weighted joint.
4. **Promote scoped claims** — Abstract/conclusion: TANGO-MB + Lemma MB-A/B/JOINT; no “joint inversion wins.”
5. **Phase-2 ACCEPT bar** — Supervisor **ACCEPT** only after SHARD gate + critical scope addressed or accepted as explicit non-goals by stakeholders.

---

## Return summary

- **Verdict:** `REVISE_MINOR`
- **Round 13 mandate (new directions + honest ablations):** **Met**
- **Negative JOINT/COUPLED/DOPT:** **Credited** — supports scoped TANGO-MB, demotes harder estimators
- **TANGO-MB:** **Defensible scoped fundamental fix** (Lemma MB-A regime); harder methods **empirically non-viable**
- **Phase 2 numeric gates:** **Met** (unchanged from R12)
- **Phase 2 holistic ACCEPT:** **Not met** — SHARD cross + deployed-FL / scope critical remain
- **vs REVISE_MAJOR:** Round 12 methodological **REVISE_MAJOR** items are **substantially closed**; do **not** require another major estimator round before SHARD
