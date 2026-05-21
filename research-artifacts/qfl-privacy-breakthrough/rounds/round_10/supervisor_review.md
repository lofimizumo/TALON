# Round 10 — Supervisor Review (Final)

## Verdict

**ACCEPT**

Round 10 closes the ten-round `qfl-privacy-breakthrough` run with a **held-out replication package** and **integrated QFL-PRIVACY-MAP synthesis** that satisfies the pre-registered early-accept rule: **two independent breakthrough signals** on seeds never used in Rounds 01–09 development (`{41, 43, 47, 53, 59}`). Supervisor re-ran `python3 code/benchmark_round10.py` (~6.7s); aggregates match `artifacts/round10_metrics.json` within float tolerance.

**Acceptance scope (mandatory framing):** This ACCEPT is for a **defender/auditor privacy map and tiered leakage science** — assignment floors (ABT-1 + T1), conditional oracle threat quantification (GARD-SPARSE 75%), honest kill-list documentation — **not** for a deployable QFL defense, wrong40 row-efficiency, honest criterion **C**, or criterion **B**.

---

## Executive summary

| Policy check | Result |
|--------------|--------|
| Signal 1: ABT-1 + T1 impossibility cite | **Pass** |
| Signal 2: Oracle GARD-SPARSE 75% @ MSE ≤ 0.10 (held-out) | **Pass** |
| Kill list respected | **Pass** |
| Forbidden: trace-C gaming | **Not used** — honest **C** fails; gaming flagged |
| Forbidden: wrong40 attack success | **Not claimed** — barrier holds (MSE ≫ 0.15) |
| Forbidden: limits-only / negative-only | **Not sole basis** — measurable oracle quant + barrier replication |
| `supervisor_verdicts` in `config.json` | **ACCEPT** (only allowed verdict) |
| `max_critical_issues` | **0** |

Deliverables audited: `paper/final_privacy_breakthrough.md`, `code/benchmark_round10.py`, `artifacts/round10_metrics.json`, `logs/experiment_round10.log`, `rounds/round_10/researcher_proposal.md`, `rounds/round_10/revision_log.md`, `config.json`, `tutorial/tutorial.md`.

---

## Signal 1 — ABT-1 + T1 impossibility (privacy floor)

### ABT-1 (held-out replication)

`audits.S2-wrong40.pass: true` on seeds `{41, 43, 47, 53, 59}` @ fractions **0.25** and **0.50**:

| Fraction | GARD mean MSE | JASPER-Q mean MSE | Floor 0.15 |
|----------|--------------:|------------------:|:----------:|
| 0.25 | **0.949** | **0.947** | Holds |
| 0.50 | **2.948** | **0.908** | Holds |

All per-seed snapshot MSE values at 0.25% are **≥ 0.89**; JASPER at 50% remains **≫ 0.15** (primary wrong40 attacker per Rounds 07–09). Theorem document `paper/assignment_barrier_theorem.md` is cited and falsifiable on simulator.

**Supervisor ruling:** ABT-1 is a **limits / barrier object**, not a defense — correctly scoped in the final paper. It **does** qualify as **Signal 1** under the Round 08/09 **Pair 1** accept bar when paired with oracle sparse quant.

### T1 impossibility (cross-cite)

Round 10 does **not** re-run T1 (charter: cite parent). Evidence chain:

1. **Proof:** `research-artifacts/qfl-terminal-snapshot/paper/impossibility_t1.md`
2. **Development replication:** `artifacts/round08_metrics.json` — `T1_impossibility_confirmed: true`, `t1_mse_mean ≈ 0.9999`
3. **Round 10 JSON:** `breakthrough_signals.signal_1.t1_impossibility_ref` points to parent proof; note states cross-cite

**Supervisor ruling:** T1 cross-cite is **valid** for Signal 1. Combined with held-out ABT-1, **Signal 1 is independent of oracle row-efficiency** (orthogonal observation classes: terminal-only vs mis-assigned Stage-2).

---

## Signal 2 — Oracle GARD-SPARSE 75% (conditional threat)

Held-out `audits.S2-oracle` (supervisor re-run):

| Metric | Value |
|--------|------:|
| Observed rows | **15 / 60** |
| Row reduction | **75%** |
| Mean GARD-SPARSE MSE | **0.0491** |
| Target MSE | **0.10** |
| Per-seed max MSE | **0.0684** (seed 47) |
| `parent_mean_reaches_target` | **true** |
| `pass` | **true** |

Development ensemble (R08, seeds 3/7/11/19/23) mean **≈ 0.050** — held-out **≈ 0.049**; no material regression.

**Supervisor ruling:** This is **conditional threat quantification** (oracle incidence + chain graph + mean anchor), **not** wrong40 deployment success. The final paper and `honesty_notes` state this explicitly. It satisfies **config criterion A** as **`A_oracle_conditional`** and **Signal 2** under Pair 1 — **acceptable** with mandatory qualifiers in all external copy.

**Forbidden misread rejected:** ACCEPT is **not** granted on wrong40 row reduction; `S2-wrong40` shows **no** path to MSE ≤ 0.10 at tested fractions.

---

## Signal 3 (secondary) — LEAK-CERT; criterion C

Held-out T1p audit: **coverage 5/5**, `criterion_c_2x_honest_naive: false`, `naive_broadcast/cert ≈ 0.93×`, `round07_trace_inflated_gaming_detected: true`.

**Supervisor ruling:** Correctly **excluded** from breakthrough count. Trace-inflated naive (**≈ 14.8×**) is diagnostic only — **not** an accept path (Rounds 07–08 supervisor kills).

---

## Kill list — respected

| Lane | Status in Round 10 |
|------|-------------------|
| Snapshot-DP (R05) | Not relitigated; criterion **B** false |
| ASSIGN-LOCK (R02/R04) | Survey/killed; not revived |
| Trace-inflated C naive (R07→R08) | Flagged; honest **C** fails |
| PROBE-RAND (R09) | Not relitigated; listed in `kill_list_confirmed` |

No killed lane is promoted as breakthrough in `final_privacy_breakthrough.md` or metrics JSON.

---

## Config gates (final)

| Gate | Status | Notes |
|------|:------:|-------|
| **A** (≥25% rows @ MSE 0.10) | **Pass** | Oracle only |
| **B** (defense +50% attack MSE) | **Fail** | Snapshot-DP killed R05 |
| **C** (cert ≥2× honest naive) | **Fail** | Documented; not gamed |
| **ABT-1** wrong40 floor | **Pass** | Dev + held-out |
| **T1** impossibility | **Pass** | Parent cite + R08 emp |
| **Two independent signals** | **Pass** | Signal 1 + Signal 2 |
| `forbid_accept_on_limits_essay_only` | **Satisfied** | Runnable benchmarks + tier metrics |
| `forbid_accept_on_negative_result_only` | **Satisfied** | Positive conditional quant + barrier |
| `forbid_accept_before_round` 8 | **Satisfied** | Round 10 |
| `require_runnable_experiments` | **Satisfied** | `benchmark_round10.py` |
| `max_critical_issues` | **0** | This review |

---

## Honesty / reproducibility audit

1. **Held-out discipline** — `held_out_seeds` ∩ `development_seeds` = ∅ per `config.json`.
2. **Re-run fidelity** — Supervisor replication: `Acceptance: ACCEPT | two_signals=True | A=True ABT1=True C_2x=False`.
3. **No metric injection** — Audits computed in `qprivacy_map_core.py` from raw runs.
4. **Oracle qualifiers** — Present in paper §Signal 2, metrics `honesty_notes`, proposal.
5. **wrong40 residual gaming** — `used_vs_true_residual` exported; JASPER `ratio_true_over_used ≈ 1.0–1.14` at 25% — does not undermine barrier (snapshot MSE still ≫ floor).

---

## Deviations from Rounds 07–09 carry demands

| Demand | Round 10 | Ruling |
|--------|----------|--------|
| Held-out seeds `{5,13,17,29,31}` (R07–R09) | Used `{41,43,47,53,59}` per `config.json` | **Acceptable** — disjoint from dev ensemble; pre-registered in config |
| `acceptance_decision.md` (R09) | Absent; `final_privacy_breakthrough.md` substitutes | **Minor** — synthesis doc is adequate |
| `supervisor_review.md` R10 (scientist charter) | **This document** | Closes policy gap |
| README supervisor table R6–R10 | Partially stale | **Minor** — update recommended |

These are **not** critical blockers for ACCEPT.

---

## Critical issues

**None** (0/0 required for accept).

## Major issues

**None** at Round 10 — prior rounds’ Path-2 protocol-fairness and gate-A labeling concerns are **resolved** by explicit oracle/wrong40 tier separation in QFL-PRIVACY-MAP.

## Minor issues

1. **T1 not re-run on held-out seeds** — reliance on R08 dev replication + parent proof; acceptable for cross-cite signal.
2. **GARD wrong40 @ 50%** unstable (**≈ 2.1–4.1** MSE) — report JASPER as primary (consistent with R09).
3. **README** lists Round 6 as “skipped” despite `rounds/round_06/supervisor_review.md` existing.
4. **`A_gard_oracle_*` JSON naming** still overloads global “A”; external copy must keep oracle qualifier.

---

## Rubric scores (1–5)

| Dimension | Score | Note |
|-----------|------:|------|
| Novelty | 4 | Integrated map + dual-signal synthesis across 10 rounds |
| Soundness | 5 | Honest scoping; held-out replication; gaming retired |
| Feasibility | 5 | ~7s held-out benchmark; full artifact chain |
| Impact | 4 | Actionable tiered audit for QFL defenders; not a shipped defense |
| Evaluability | 5 | JSON audits + reproduction commands |

---

## 10-round arc (supervisor summary)

| Round | Contribution |
|-------|----------------|
| 01 | Oracle GARD-SPARSE conditional win; B/C fail |
| 02–03 | Assignment stress; barrier emerges |
| 04–05 | JASPER diagnostic; Snapshot-DP killed |
| 06 | ABT-1 theorem; Path 2 fail |
| 07 | JASPER-Q v7; trace-C gaming |
| 08 | QFL-PRIVACY-MAP integration; honest C fix |
| 09 | PROBE-RAND honest failure |
| 10 | Held-out replication + **ACCEPT** package |

---

## Actionable post-accept notes (non-blocking)

1. External papers should cite **ACCEPT scope** (audit framework + two signals), not “QFL privacy solved.”
2. Optional: rename `A_gard_oracle_75pct_reduction_at_mse_0.10` → `A_oracle_conditional_pass` in future JSON revisions.
3. Fix README Round 6 row for archival consistency.

---

## Supervisor sign-off

| Field | Value |
|-------|-------|
| Round | 10 (final) |
| Verdict | **ACCEPT** |
| Framework | **QFL-PRIVACY-MAP** (defender/auditor) |
| Signal 1 | **ABT-1 + T1 impossibility** — pass |
| Signal 2 | **Oracle GARD-SPARSE 75%** — pass (held-out) |
| Kill list | **Respected** |
| Forbidden paths | **Not used** (trace-C, wrong40 win, limits-only) |
| Critical issues | **0** |
| `config.json` `supervisor_status` | **ACCEPT** |
