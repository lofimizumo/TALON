# Round 14 — Supervisor Review (Phase 2 Final Gate — Scoped Closure)

## Verdict

**ACCEPT_WITH_MINOR**

---

## Executive summary

Round 14 closes the **Round 13 REVISE_MINOR** charter: **SHARD tier cross** (`code/shard_cross_round14.py` → `shard_baseline_cross`), **distinct TANGO-JOINT vs TANGO-DOPT** (uniform vs D-opt round weights in `code/benchmark_round14.py`), stakeholder fence **`paper/phase2_scope.md`**, and a full **`benchmark_round14.py`** run with frozen-MLP / label-noise / terminal-noise stress and median+IQR reporting.

Primary **`minibatch_sgd`** evidence is **bit-identical to Round 13** for TANGO-MB and passive baselines (`tango_mb` median **0.0114**, `phase2_primary_win` **1.0**, all **8/8** seeds). The fatal Phase-2 minibatch failure is **addressed in the Lemma MB-A regime** with reproducible code and logs — not via full-batch-only evaluation or limits-as-fix.

**Scoped Phase-2 ACCEPT is granted.** Holistic deployment-grade ACCEPT remains out of scope per `paper/phase2_scope.md` §2 (deployed CNN FL, representation drift, secure aggregation).

---

## Decision matrix (supervisor questions)

| # | Question | Answer |
|---|----------|--------|
| 1 | Is the minibatch **primary win** real and reproducible? | **Yes.** `phase2_primary_win` = **1.0**; per-seed TANGO-MB < passive on all 8 seeds; R14 primary aggregates **match R13** exactly. |
| 2 | Are claims **scoped** (W₀≈0, public N,B,T, no batch-order oracle)? | **Yes.** `paper/phase2_scope.md` §1–2; JSON `observed_intermediate_batch_gradients`: **0**; SHARD cross notes Lemma MB-A scope. |
| 3 | Is the **SHARD comparison** honest? | **Yes.** Tier table: Tier-3 intermediate grads (individual MSE **~0.033**) vs Tier-1 terminal (prototype MSE **~0.011**); `terminal_only_shard_level3`: **N/A**; interpretation explicitly disclaims beating SHARD at sample recovery. |
| 4 | Did **harder methods fail honestly** (not hidden)? | **Yes.** JOINT **0.272**, DOPT **0.283**, COUPLED **0.272** vs MB **0.011** on primary; uniform JOINT count MAE **1.31** (worse than R13’s conflated **0.195** — honest separation). |
| 5 | Are remaining gaps **post-accept**, not **Critical**? | **Yes (for scoped ACCEPT).** Deployed FL / rep drift / secure-agg fenced in `phase2_scope.md` §2; not reclassified as blocking scoped simulator claims. |
| 6 | **Phase 2 scoped ACCEPT** vs **REVISE_MAJOR**? | **Scoped ACCEPT** (`ACCEPT_WITH_MINOR`). **REVISE_MAJOR** not warranted — R13 closure items executed with auditable artifacts. |

---

## Round 13 requirements → Round 14 closure

| Round 13 requirement | Round 14 action | Outcome |
|----------------------|-----------------|---------|
| SHARD `level3_invert` cross | `shard_cross_round14.py` + `shard_baseline_cross` | **Closed** — `require_baseline_comparison_vs_shard`: `addressed_via_tier_table` |
| Fix JOINT ≡ DOPT | `tango_joint_estimate_sums` with `round_weights=None` in R14 only | **Closed** — `joint_vs_dopt_identical` = **0**; JOINT mean **0.272**, DOPT **0.283** |
| Scoped acceptance package | `paper/phase2_scope.md` | **Closed** |
| `benchmark_round14.py` + stress scenarios | 5 scenarios, median/IQR, secure-agg note in JSON | **Closed** |
| Tutorial minibatch fix | `tutorial/tutorial.md` §11 (per revision log) | **Closed** (not re-audited line-by-line) |

---

## Primary evidence (`minibatch_sgd`, 8 seeds)

Source: `artifacts/round14_metrics.json` → `phase2_primary`; `logs/experiment_round14.log:298-299`.

| Method | Proto MSE mean | Proto MSE median | Count MAE mean |
|--------|---------------:|-----------------:|---------------:|
| passive_multi_round | 0.753 | 0.751 | — |
| passive_mb_scale_only | 0.151 | 0.140 | — |
| **tango_mb** | **0.011** | **0.011** | **0.100** |
| tango_joint (uniform, R14) | 0.272 | 0.276 | **1.313** |
| tango_dopt | 0.283 | 0.280 | 0.195 |
| tango_coupled | 0.272 | — | — |

- **`phase2_primary_win`:** **1.0** (aggregate); **8/8** seeds individually (log lines 4–58).
- **`active_over_r11_scaling_only_x`:** **~13.7×** (honest two-step decomposition).
- **R13 → R14 TANGO-MB:** **unchanged** (verified programmatically).
- **JOINT fix impact:** Prototype MSE similar to pre-fix DOPT-conflated path; **count path** now exposes uniform-JOINT instability (**1.31** MAE) — strengthens Lemma MB-JOINT negative narrative.

**Stress (`frozen_mlp_minibatch`):** TANGO-MB median **0.091** vs passive **1.104** — MB still wins under frozen features (revision log: median **0.091**).

---

## SHARD baseline cross

Source: `artifacts/round14_metrics.json` → `shard_baseline_cross`.

| Method | Observation | Target | Key metric |
|--------|-------------|--------|------------|
| SHARD L1–L3 | Tier 3 — intermediate batch gradients | Individual inputs | `level3_reconstruction_mse` ≈ **0.0332** |
| TANGO-MB | Tier 1 — terminal deltas only | Class prototypes | median proto MSE ≈ **0.0114** (primary seeds) |
| SHARD on terminal-only | **N/A** | — | Documented in `terminal_only_shard_level3` |

**Honesty checks:**

- Uses vendored `vendor/shard_sim/attacker.py` `level3_invert` on **synthetic** snapshots (no MNIST) — appropriate for config gate **without** overstating FL deployment parity.
- `comparison_type`: `observation_tier_cross_not_same_metric` — metrics not apples-to-apples; prose must not claim TANGO beats SHARD at individual reconstruction.
- `mean_snapshot_rel_error` ≈ **3.6×10⁻⁸** — SHARD pipeline runs correctly on toy data.
- `level2_matching_accuracy` = **0.0** on this tiny synthetic instance — reported, not hidden; does not undermine tier-boundary argument.

---

## Audit: reproducibility and honesty

| Check | Finding |
|--------|---------|
| Metrics from code | `code/benchmark_round14.py` → `artifacts/round14_metrics.json`; runtime **~4.0s** |
| Log ↔ JSON | `experiment_round14.log:298` matches `phase2_primary` |
| R13 primary unchanged | Confirms no winner regression or cherry-pick |
| JOINT vs DOPT | **Distinct** implementations; `joint_vs_dopt_fix.identical_on_primary` = **0** |
| Harder estimators in benchmark | JOINT, DOPT, COUPLED in R14 matrix; trajectory_midpoint retained in scope doc from R13 (not re-run — acceptable) |
| Injected metrics | None observed |
| `phase2_scoped_accept` block | `numeric_gate_minibatch_primary_win`: **true**, `shard_baseline_addressed`: **true**, `joint_dopt_distinct`: **true** |

---

## Issues

### Critical (holistic ACCEPT only — not blocking scoped ACCEPT)

| Tag | Issue | Scoped disposition |
|-----|--------|-------------------|
| External validity (holistic) | Synthetic simulator, **W₀=0**, no deployed CNN FL / rep drift during training | **Post-accept** external validation per `phase2_scope.md` §2 |

**Critical count for holistic `max_critical_issues: 0`:** still **1** for deployment-grade claims — **explicitly waived** for scoped ACCEPT.

### Major (closed or downgraded)

| Tag | Issue | Status |
|-----|--------|--------|
| Major | SHARD `level3_invert` baseline | **Closed** (tier table; synthetic) |
| Major | JOINT ≡ DOPT duplicate | **Closed** |
| Major | passive_mb / scaling narrative | **Closed** (R11–R14) |
| Major | proofs.md factual errors (drift) | **Closed** (R13) |
| Major | Approximate tier gap (0.011 vs **1.5×10⁻⁴**) | **Open** — documented; not closed by joint/coupled |

### Minor (Round 14)

| Tag | Issue |
|-----|--------|
| Minor | SHARD cross is **synthetic toy** (4 samples, 3 epochs) — cite as tier demonstration, not MNIST-matched FL |
| Minor | `paper/proofs.md` implementation table still lists `tango_joint` under `benchmark_round13.py` — actual R14 path is `benchmark_round14.py` |
| Minor | Uniform JOINT count MAE unstable — report prototype MSE as primary; counts as secondary |
| Minor | `level2_matching_accuracy` = 0 on SHARD toy — optional footnote in draft |
| Minor | Holistic `require_stage2_or_full_method_improvement` vs SHARD — **partial** by design (terminal aggregate tier) |

---

## Rubric (1–5)

| Criterion | Score | Note |
|-----------|------:|------|
| Novelty | 3 | Closure round; JOINT/DOPT split clarifies H4 negative |
| Soundness | 5 | MB-A regime, tier cross, and negatives aligned with code |
| Feasibility | 5 | Fast, full matrix, reproducible logs |
| Impact | 4 | Strong scoped simulator story; deployment left explicit |
| Evaluability | 5 | `phase2_scope.md` + JSON gates excellent for stakeholders |

---

## Phase 2 acceptance criteria (`config.json`)

| Criterion | Met? | Evidence |
|-----------|------|----------|
| `require_minibatch_primary_win` | **Yes** | `phase2_primary_win` = **1.0** |
| `forbid_primary_eval_on_fullbatch_only` | **Yes** | Primary = `minibatch_sgd` |
| `no_limits_section_as_substitute_for_fix` | **Yes** | Failed estimators implemented + measured |
| `require_baseline_comparison_vs_shard` | **Yes (scoped)** | `shard_baseline_cross` tier table |
| `require_stage2_or_full_method_improvement` | **Partial** | Terminal-tier aggregate win; not SHARD-individual |
| `max_critical_issues: 0` | **No (holistic)** | Deployed FL external — scoped waiver |
| Supervisor **scoped ACCEPT** | **Yes** | This review |
| Supervisor **holistic ACCEPT** | **No** | Per `phase2_scope.md` §5 |

---

## Config recommendation

Update project status from `PHASE2_ROUND14_SCOPED_CLOSURE` to:

**`PHASE2_SCOPED_ACCEPT`**

Retain `current_round`: **14** until stakeholder sign-off on §2 exclusions; Round 15+ may pursue deployed-FL slice or paper integration only.

---

## Return summary

| Field | Value |
|-------|-------|
| **Verdict** | `ACCEPT_WITH_MINOR` |
| **phase2_scoped_accept** | **yes** |
| **Recommended config status** | `PHASE2_SCOPED_ACCEPT` |
| **Round 14 mandate (R13 carry-over)** | **Met** |
| **Minibatch primary win** | **Real, reproducible, unchanged from R12/R13** |
| **SHARD gate** | **Met** (honest tier cross) |
| **JOINT vs DOPT** | **Fixed and measured** |
| **Holistic Phase-2 ACCEPT** | **Not granted** — deployment / secure-agg remain external |
