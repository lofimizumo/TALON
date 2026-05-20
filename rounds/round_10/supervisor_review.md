# Round 10 Supervisor Review

## 1. Verdict

**ACCEPT**

Round 10 closes the Round-09 **Major** manuscript gaps with reproducible artifacts: formal proofs in `paper/proofs.md`, a synthetic decoder semantic probe, frozen-MLP backbone experiments, and limits-first manuscript text (`paper/draft.md`, `paper/method.tex`). Metrics are computed from simulation code (not injected), agree with `logs/experiment_round10.log`, and decoder/minibatch numbers match Round-09 baselines where expected. No **Critical** issues remain under `config.json` acceptance rules.

---

## 2. Executive Summary

The proposal targets full **ACCEPT** by addressing Round-09 supervisor items #2 (decoder bridge), #3 (count vs prototype separation), #4 (minibatch limits), and #5 (formal proofs), plus optional frozen-backbone support.

`code/benchmark_round10.py` imports the Round-09 core, adds decoder and frozen-MLP scenarios, and writes `artifacts/round10_metrics.json` in ~0.11s. Headline numbers match the log and recomputed per-seed means: decoder balanced hidden/pixel MSE `0.000151`, decoder minibatch `6.5877`, frozen MLP balanced TANGO `0.00302` vs passive `0.367` (~218× gain).

`paper/proofs.md` supplies complete proofs for Lemmas A–B and Theorems 1–2 with an assumption list aligned to `theorem_scope.exact` in metrics JSON. Proofs are substantive (not sketch-only), with explicit scope boundaries for minibatch, drift, and passive probing.

The decoder experiment is honestly scoped: fixed orthonormal $D$, pixel MSE tracks hidden MSE under Tier-1 success; minibatch failure propagates to pixel MSE; high correlation under failure is flagged as misleading. Frozen MLP fixes $\phi(x)$ during local training—supporting fixed-features + linear head, not representation drift.

Manuscript artifacts lead with minibatch failure and separate count vs prototype wins. Residual gaps (no deployed CNN FL, known decoder at eval time) are documented limits, not hidden failures. **Paper readiness: 5/5** for the stated narrow exact-regime contribution.

---

## 3. Honesty / Reproducibility Audit

| Check | Finding | Reference |
|---|---|---|
| Metrics computed, not injected | All values from `run()` loops; no hardcoded headline constants | `code/benchmark_round10.py:170–287`, `376–494` |
| JSON ↔ log consistency | Per-seed lines match log; means match `decoder_headlines` / `frozen_mlp_headlines` / `headline` | `artifacts/round10_metrics.json`, `logs/experiment_round10.log:2–35` |
| Decoder balanced = Round 09 exact tier | Hidden MSE `0.000151` matches `balanced_clean` TANGO prototype MSE | `artifacts/round10_metrics.json:14–15`, Round-09 `0.000151` |
| Decoder minibatch = Round 09 failure | Hidden/pixel MSE `6.5877` matches Round-09 `minibatch_sgd` | `artifacts/round10_metrics.json:35–44` |
| Pixel MSE equals hidden MSE (balanced) | Expected under fixed linear $y=xD$ on prototypes | `code/benchmark_round10.py:190`, `74–75` |
| Oracle decoder labeled | `fit_linear_decoder_lstsq` on all $(x,y)$; metrics `decoder_lstsq_*` are upper-bound diagnostics | `code/benchmark_round10.py:192–213`, `decoder_setup.semantic_claim` |
| Frozen features actually frozen | `frozen_mlp_features` draws weights once; no update during local steps | `code/benchmark_round10.py:103–113`, `149–167` |
| Intermediate gradients absent | Round-10 reuses Round-09 observation path; no intermediate batch rows | `benchmark_round09.py:505` (inherited) |
| Deterministic seeds | Same 8-seed tuple via imported `SimConfig` | `benchmark_round09.py:45`, `96–111` |
| Runnable command | `python3 code/benchmark_round10.py` → JSON, SVGs, log | `revision_log.md:40–45`, `runtime_seconds: 0.109` |

**Round-09 Major closure (verified):**

| Round-09 Major | Round-10 status |
|---|---|
| #2 No semantic/decoder bridge | ✅ `decoder_probe` scenarios + limits in `paper/draft.md` §1, `method.tex` §decoder |
| #3 Count vs prototype asymmetry | ✅ `paper/draft.md` §1.2, `method.tex` §baselines |
| #4 Minibatch breaks estimator | ✅ Limits lead; `decoder_minibatch` pixel MSE `6.5877` |
| #5 Formal proofs absent | ✅ `paper/proofs.md` (Lemmas A–B, Theorems 1–2, scope table) |
| #1 Synthetic-only scope | ⚠️ Still synthetic; frozen MLP + decoder extend but do not deploy CNN FL—documented as limit (not dishonesty) |

**Not rerun by supervisor;** audit is artifact-consistent and per-seed mean recomputation verified.

---

## 4. Feasibility & Resources

- **Compute:** Benchmark ~0.11s (`artifacts/round10_metrics.json:488`). Trivial for CI/tutorial.
- **Scope:** Synthetic hidden-feature (or frozen-MLP) simulator with linear head. Manuscript and `theorem_scope` keep claims tiered.
- **SHARD baseline:** Tier-theoretic comparison unchanged from Round 09; acceptable under `require_baseline_comparison_vs_shard` via observation ladder + vendored SHARD context.

---

## 5. Theory & Assumptions

**Proofs are complete for the stated exact tier**, not sketch-only:

- **Lemma A** derives bias/weight gradient identities at $W_0=0$ with full-batch softmax CE; maps to `estimate_counts` / `tango_estimate_sums`.
- **Lemma B** gives count identifiability from bias deltas.
- **Theorem 1** stacks coordinate-wise linear systems with consistency constraint and full-rank probe condition; matches per-coordinate `lstsq` in code.
- **Theorem 2** gives constructive within-class zero-sum perturbation family; corollary ties to `individual_mse_from_prototypes`.

**Assumption alignment:** `theorem_scope.exact` in Round-10 JSON matches proof preamble and `paper/proofs.md` scope table. `frozen_mlp_*` correctly notes fixed $\phi(x)$ during local training (linear head on fixed nonlinear features).

**Residual theory nuance (Minor, not Critical):** Default `local_steps=3` (`benchmark_round09.py:38`). Lemma A proof text emphasizes initialization at $W_0=0$ then scales by $\eta T$ under a **first-order** terminal model—the same approximation the implementation uses via `avg_grad = -delta / (lr * local_steps)`. Exactness for $T>1$ with moving weights is empirical (`more_local_steps` in approximate tier), not re-proved for arbitrary $T$. This is disclosed via `theorem_scope` and does not invalidate the identifiability structure.

**Decoder theory:** No new theorem—correctly framed as a **linear semantic bridge** when Tier-1 succeeds, not raw-image inversion.

**Frozen MLP:** Extends fixed-feature assumption to nonlinear $\phi$; does **not** prove robustness to representation drift (explicit in proposal, tutorial §7, `method.tex` §limits).

---

## 6. Rubric Scores (1–5)

| Criterion | Score | Note |
|---|---:|---|
| Novelty | 4 | Tiered framework + active terminal aggregates; decoder/frozen extensions are supporting evidence |
| Soundness | 4.5 | Formal proofs + honest minibatch failure + scoped decoder claims |
| Feasibility | 4 | Fast, reproducible; still not deployed FL |
| Impact | 4 | Decoder bridge raises stakes for hidden-feature privacy when Tier-1 holds |
| Evaluability | 4.5 | Separated metrics, baselines, negative rows, proof–code map |

**Paper readiness:** **5/5** (up from Round-09 **4/5**). Meets the strict acceptance target for the narrow exact-regime paper package.

---

## 7. Issues

### Critical

*None.*

### Major

*None blocking acceptance.* Round-09 Majors #2–#5 are closed. Round-09 Major #1 (synthetic-only) is reframed as an explicit, documented evaluation boundary with Round-10 extensions (decoder, frozen MLP), not an undisclosed over-claim.

### Minor

1. **Deployed FL still absent** — No CNN/transformer federated run; claims must stay simulator-tiered (`researcher_proposal.md:120–125`, `paper/draft.md` §1.4).
2. **Decoder evaluation uses known $D$** — Demonstrates structure correlation when prototypes are recovered; not an attack that learns semantics from gradients alone (`code/benchmark_round10.py:64–71`, `180`).
3. **Minibatch correlation footnote** — High corr (`≈0.999`) with huge MSE must stay in every table caption; tutorial §7 does; ensure figures do not over-read corr.
4. **Frozen MLP ≠ drift** — Does not model features changing during local training (`tutorial/tutorial.md:196`).
5. **No frozen_mlp + minibatch cross** — Optional stress; would strengthen “nonlinear fixed features still fail under minibatch” narrative.
6. **Lemma A / multi-step** — First-order terminal model with default `local_steps=3`; full nonlinear multi-step proof not required for acceptance but worth one sentence in `proofs.md` Remark.
7. **Carry-over from Round 09** — Public-prior count metric quirks; passive can beat TANGO on counts in neutral regimes (still correctly reported).

---

## 8. Actionable Suggestions (prioritized)

1. **Submission polish:** Promote `paper/draft.md` limits §1 into the LaTeX intro/abstract first paragraph; keep minibatch table above contributions.
2. **Figure discipline:** On `round10_decoder_probe.svg`, annotate that minibatch correlation is not semantic recovery when MSE $\gg 1$.
3. **Proof remark:** Add one sentence in Lemma A that $T>1$ steps are treated via first-order accumulated-gradient equivalence (matching code division by `local_steps`).
4. **Optional (post-accept):** `frozen_mlp` + `decoder_minibatch` combined scenario; secure-aggregation / label-noise rows.
5. **Keep oracle decoder metrics** out of attack comparison figures (already separated in JSON).

---

## Round-09 → Round-10 Acceptance Criteria Check

| Criterion (`config.json`) | Met? |
|---|---|
| `supervisor_verdicts` includes only **ACCEPT** | ✅ This review |
| `max_critical_issues: 0` | ✅ |
| `require_runnable_experiments` | ✅ |
| `require_baseline_comparison_vs_shard` | ✅ (tier ladder; Round 09+) |
| `require_stage2_or_full_method_improvement` | ✅ |
| `require_threat_model_mitigation` | ✅ |
| `stage3_only_is_insufficient` | ✅ |
| Paper-ready: formal proofs | ✅ `paper/proofs.md` |
| Paper-ready: decoder probe | ✅ `decoder_probe` scenarios |
| Paper-ready: limits documented | ✅ `paper/draft.md`, `method.tex` limits-first |

---

## Bottom Line

Round 10 delivers an honest, reproducible manuscript-closing package. Round-09 **Major** items on proofs, decoder bridge, count/prototype separation, and minibatch limits are substantively closed. Metrics match logs; experiments do not inject results; claims remain tiered and limits-led.

Supported thesis:

> Active terminal probes identify class-level hidden prototypes and aggregate moments under the exact linear full-batch model (with formal proofs); a linear decoder probe links recovered prototypes to synthetic pixel class means when Tier-1 succeeds; frozen nonlinear features preserve Tier-1 with small error; minibatch SGD breaks the estimator; individuals remain non-identifiable; SHARD-equivalent recovery requires stronger observations.

**Verdict: ACCEPT.** Acceptance criteria in `config.json` are met.
