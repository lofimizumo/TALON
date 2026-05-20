# Round 09 Supervisor Review

## 1. Verdict

**ACCEPT_WITH_MINOR**

Round 09 closes the Round-08 evaluation checklist honestly: extended stress tests, separated baselines, count-vs-prototype reporting, theorem-scope labeling, and active-probing threat-model wording. Metrics are computed from simulation code (not injected), reproducible from logs, and the minibatch-SGD failure is reported prominently rather than buried. Remaining gaps are manuscript-scope (nonlinear FL, semantic decoder evidence, formal proofs), not round-blocking integrity flaws.

---

## 2. Executive Summary

The proposal correctly reframes TALON/TANGO as a **tiered aggregate-leakage** contribution, not a SHARD-equivalent individual attack. Round-09 experiments extend `benchmark_round08.py` with minibatch SGD, nonzero head initialization, and 10-class/30-dim scale-up, plus passive multi-round, public prior, and oracle-aggregate baselines.

Saved artifacts agree with `logs/experiment_round09.log` on all headline numbers (balanced TANGO prototype MSE `0.000151`, passive `0.143141`, minibatch TANGO prototype MSE `6.5877` vs passive `0.7532`). The code records `observed_intermediate_batch_gradients = 0.0` and does not hardcode results.

**Minibatch SGD is an honest negative result:** TANGO active prototype MSE `6.59` is worse than passive multi-round `0.75` (`tango_vs_passive_proto_gain_x` ≈ `0.11` on that scenario). The proposal, tutorial, metrics JSON `theorem_scope`, and log lines all surface this.

Round-08 required evaluation edits are **substantively closed**. Paper drafting can proceed under `paper/outline.md`, with non-blocking fixes listed below. The project still lacks real-backbone FL, decoder evidence, and formal proofs in-repo.

---

## 3. Honesty / Reproducibility Audit

| Check | Finding | Reference |
|---|---|---|
| Metrics computed, not injected | All values derived from `run()` simulation loops; no hardcoded headline constants | `code/benchmark_round09.py:449–629` |
| JSON ↔ log consistency | Headline fields match last log line and per-seed aggregates (verified by recomputing means from `per_seed`) | `artifacts/round09_metrics.json:538–551`, `logs/experiment_round09.log:234` |
| Intermediate gradients absent | `observed_intermediate_batch_gradients: 0.0` on every row | `code/benchmark_round09.py:505`, log throughout |
| Oracle separated from attack | `oracle_aggregate_prototypes` uses ground-truth sums/counts; labeled upper bound in proposal and outline | `code/benchmark_round09.py:268–272`, `paper/outline.md:64` |
| Minibatch failure visible | High TANGO errors logged per seed; headline includes `minibatch_tango_prototype_mse` | `logs/experiment_round09.log:69–100`, `artifacts/round09_metrics.json:3986–3991` |
| Public-prior count metric | Uniform count guess → `count_mae=0` on balanced; misleading on imbalanced (`2.6667`) | `code/benchmark_round09.py:484`, `logs/experiment_round09.log:38` |
| Deterministic seeds | Fixed `seeds` tuple, 8 seeds per scenario | `code/benchmark_round09.py:45`, `96–111` |
| Runnable command | `python3 code/benchmark_round09.py` writes JSON, SVGs, log in &lt;2s | `revision_log.md:28–32`, log timestamp |

**Round-08 edit closure (verified):**

| Round-08 requirement | Round-09 status |
|---|---|
| Minibatch SGD stress | ✅ `minibatch_sgd` scenario; failure documented |
| Nonzero initial head | ✅ `nonzero_init_head` |
| Larger classes/dimensions | ✅ `large_10class_30dim` |
| Passive multi-round baseline | ✅ `passive_multi_round` |
| Public/statistical prior | ✅ `public_prior` |
| Oracle aggregate (not TANGO) | ✅ `oracle_aggregate` |
| Count vs prototype reporting | ✅ separate metrics and SVG |
| Exact vs approximate theorem scope | ✅ `theorem_scope` in JSON + proposal § |
| Active probing threat model | ✅ proposal, tutorial §3, JSON `threat_model` |
| Tutorial duplicate JOLI block | ✅ single integrated tutorial; JOLI background only |

**Not rerun by supervisor;** audit is artifact-consistent only.

---

## 4. Feasibility & Resources

- **Compute:** Benchmark completes in ~0.6s (log: `07:06:11.171` → `07:06:11.762`). Feasible for CI and tutorial reproduction.
- **Scope:** Still a **synthetic linear-head** simulator on fixed hidden features. No CNN/transformer, no federated deployment, no secure aggregation. Manuscript claims must stay in the exact/approximate tiers.
- **SHARD baseline:** Round 09 does not re-run vendored `level3_invert`; comparison is **tier-theoretic** (SHARD Stage 2 needs intermediate rows). Acceptable given `config.json` `require_baseline_comparison_vs_shard` is satisfied by the observation ladder + prior-round SHARD context, but the paper should keep one explicit “SHARD N/A under terminal-only” paragraph.

---

## 5. Theory & Assumptions

**Sound under stated narrow model:** linear softmax head, fixed hidden features, known \(\eta, T, N\), full-batch first-order terminal deltas, zero initial weights, full-rank probe design. The coordinate-wise linear system in the proposal matches the implementation (`tango_estimate_sums`, `estimate_counts`).

**Correctly demoted to approximate tier:** minibatch SGD, nonzero `w0`, more local steps. Round 09 adds empirical evidence that minibatch breaks the first-order mapping (prototype MSE `6.59`, count MAE `2.57`).

**Non-identifiability:** Individual recovery stays near within-class floor (`high_within_class_variance` individual MSE `0.983` vs floor `0.837`). Consistent with Theorem 2 narrative.

**Gaps:** No formal proof artifacts (LaTeX/proof script) in the reviewed inputs; theorems remain sketches. Nonlinear representation drift during local training is unaddressed.

---

## 6. Rubric Scores (1–5)

| Criterion | Score | Note |
|---|---:|---|
| Novelty | 4 | Tiered leakage framework + active terminal aggregate target |
| Soundness | 4 | Toy exact regime + honest stress failures |
| Feasibility | 4 | Runnable, fast, reproducible; not yet real FL |
| Impact | 3.5 | Meaningful if hidden features are privacy-relevant; decoder gap limits stakes |
| Evaluability | 4 | Baselines, separated metrics, negative minibatch row |

**Paper readiness:** **4/5** (up from Round-08 **3.5/5**). Evaluation closure is sufficient to draft; not sufficient for submission without nonlinear/semantic extensions.

---

## 7. Issues

### Critical

*None.* Minibatch failure is disclosed; metrics are not fabricated; threat model is not mislabeled as passive.

### Major

1. **Synthetic-only evaluation** — All scenarios use fixed hidden features and a linear head. Claims must not generalize to nonlinear FL without new experiments (`researcher_proposal.md:169`, `paper/outline.md:68`).
2. **No semantic/decoder bridge** — Hidden prototypes may not imply raw-input leakage. Round-08 Major item remains open (`round_08/supervisor_review.md:130`).
3. **Count recovery is not uniformly won by active probing** — On `balanced_clean`, passive count MAE `0.0152` beats TANGO `0.0419` while TANGO wins on prototypes (`researcher_proposal.md:95–97`). Paper must not imply “active dominates passive” on all metrics.
4. **Minibatch SGD breaks the main estimator** — TANGO prototype MSE `6.59` vs passive `0.75`; undermines default FL training narrative unless relegated to limits/future work (`artifacts/round09_metrics.json:3986–3991`).
5. **Formal proofs absent from artifact chain** — Theory sections need written proofs before submission.

### Minor

1. **`large_10class_30dim`:** passive count MAE `0.0038` beats TANGO `0.0294` despite strong TANGO prototypes (`scenario_method_headlines`).
2. **Public-prior baseline** — Crude single-round mean; count metrics misleading on imbalanced data (`logs/experiment_round09.log:38`).
3. **`more_local_steps` listed in `approximate_empirical`** but still full-batch — scope table should clarify it is multi-step approximation, not minibatch (`artifacts/round09_metrics.json:4043–4050`).
4. **No secure aggregation, label noise, or client drift** — standard FL deployment gaps.
5. **Individual MSE** — Nearest-neighbor on repeated prototypes is a diagnostic, not SHARD-level reconstruction metric.

---

## 8. Actionable Suggestions (prioritized)

1. **Manuscript limits section (blocking for submission, not this round):** Lead with minibatch failure table; state first-order TANGO is exact-regime only; cite passive `0.75` vs TANGO `6.59` on `minibatch_sgd`.
2. **Separate “prototype win” from “count win”** in every summary table and abstract; acknowledge passive can estimate counts better in neutral-bias regimes.
3. **Add one decoder or semantic probe experiment** (even synthetic) before claiming raw-input privacy impact — or keep endpoint explicitly at hidden-representation aggregates.
4. **Promote theorem sketches to proofs** in an appendix with assumption list matching `theorem_scope.exact`.
5. **Optional Round-10 experiment:** one nonlinear backbone with frozen features to test whether Tier-1 story survives representation drift.
6. **Keep oracle_aggregate out of attack comparisons** in figures except as dashed ceiling (already correct in `round09_baselines.svg` design).

---

## Round-08 → Round-09 Acceptance Criteria Check

| Criterion (`config.json`) | Met? |
|---|---|
| `require_runnable_experiments` | ✅ |
| `require_stage2_or_full_method_improvement` | ✅ (threat-model reduction + tier framing, not Stage-3 polish) |
| `require_threat_model_mitigation` | ✅ (no intermediate gradients) |
| `stage3_only_is_insufficient` | ✅ (TANGO Tier 1, JOLI background only) |
| Zero Critical issues (paper-ready target) | ✅ |

---

## Bottom Line

Round 09 delivers an honest, reproducible evaluation package that closes Round-08 supervisor requests. The supported thesis remains:

> Active terminal probes identify class-level hidden prototypes and aggregate moments under exact-regime assumptions, but not individual samples; minibatch SGD breaks the first-order estimator; SHARD-equivalent individual recovery requires stronger observations.

Proceed to manuscript drafting with listed Major items as explicit limits, not as hidden failures.
