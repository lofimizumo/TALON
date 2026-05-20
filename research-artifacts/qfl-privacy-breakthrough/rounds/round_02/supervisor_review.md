# Round 02 — Supervisor Review

## Verdict

**REVISE_MAJOR**

Round 02 is a **substantive compliance round**: the scientist implemented nearly all Round 01 mandatory fixes, withdrew the inflated **85%** headline, and produced reproducible assignment/graph stress artifacts. There is **no breakthrough under stressed assignment** (`wrong20`, `wrong40`, `unknown_random`). Criterion **A** survives only as **parent-aligned conditional replication** under oracle assignment + oracle chain graph + oracle mean anchor—the same favorable stack parent TALON Round 05 already reported. Policy forbids **ACCEPT** before round 8 regardless.

---

## Executive summary

The deliverables (`code/benchmark_round02.py`, `artifacts/round02_metrics.json`, `logs/experiment_round02.log`, `rounds/round_02/researcher_proposal.md`, `rounds/round_02/revision_log.md`) are **honest and runnable**. Supervisor re-ran `python3 code/benchmark_round02.py` (~34s); metrics match the submitted JSON.

**Three verification targets (Round 02 sign-off):**

| Check | Result | Supervisor ruling |
|-------|--------|-------------------|
| **85% withdrawal** | `claim_withdrawal.round01_headline_85pct_withdrawn: true`; inflated 9/60 (85%) vs parent gate 15/60 (75%) @ MSE 0.10 | **Confirmed** — headline correctly withdrawn |
| **Assignment stress** | `wrong20`, `wrong40`, `unknown_random` ported; observations from true H, solver from assumed H | **Confirmed** — regimes exercised in code and JSON |
| **Criterion A under wrong assignment** | All three non-oracle regimes: `reaches_target: false` @ MSE 0.10 parent gate | **Confirmed fail** — no ≥25% reduction at equal MSE |

**Breakthrough gates (`config.json`):**

| Criterion | Round 02 JSON / proposal | Supervisor ruling |
|-----------|--------------------------|-------------------|
| **A** — ≥25% row reduction @ equal snapshot MSE | Oracle parent gate @ 0.10: **75%** (15/60). wrong20/40/unknown: **none** | **Conditional replication only** — not deployment breakthrough |
| **B** — ≥50% attack MSE cut @ ≤10% utility | `B_defense_50pct_at_10pct_utility: false`; 0/100 SHIELD cells feasible | **Fail** (kill criteria met → deprioritize OK) |
| **C** — cert ≥2× tighter than naive | Parked Round 03+ | **Fail / N/A** |

`best_honest_row_reduction_at_0.10_parent_gate`: **75%**, assignment **oracle**, method **gard_oracle_graph** — no stressed-assignment competitor qualifies.

---

## Verification audit

### 1. 85% claim withdrawal — **CONFIRMED**

Pre-registered gate: `parent_mean_mse_at_minimum_rows` (`criterion_A_preregistered_gate` in JSON).

| Statistic | Round 01 (inflated) | Round 02 (parent-aligned) |
|-----------|--------------------:|--------------------------:|
| Gate | any single seed @ MSE ≤ 0.10 | mean MSE ≤ target across seeds at minimum rows |
| Min rows (GARD oracle graph @ 0.10) | **9/60** | **15/60** |
| Reduction | **85%** | **75%** |

From `artifacts/round02_metrics.json` → `claim_withdrawal`:

- `round01_inflated_min_rows_at_0.10`: 9, `round01_inflated_reduction`: 0.85
- `parent_aligned_min_rows_at_0.10`: 15, `parent_aligned_reduction_at_0.10`: 0.75
- Oracle `parent_mean_mse_gate` @ 0.10: `mean_mse` ≈ **0.049**, `fraction` 0.25 (15 rows)

The **85%** figure persists only as a **diagnostic** at MSE ≤ **0.15** (9 rows, mean MSE ≈ 0.130). Proposal and revision log correctly **withdraw** it as the MSE 0.10 headline. This reconciles Round 01 with parent `benchmark_round05.py` semantics (`snapshot_mse_mean <= target` over seeds at budget).

Round-01-style diagnostic retained: `round01_any_seed_min_rows_at_0.10: 9` under oracle — documents the metric regression, not a claim.

### 2. Assignment stress — **CONFIRMED**

`code/benchmark_round02.py` implements four regimes with true-H observations and assumed-H solve (`make_assumed_incidence`, lines ~168–178). Log: `assignments=['oracle', 'wrong20', 'wrong40', 'unknown_random']`.

**GARD oracle graph mean MSE @ fraction 0.25 (oracle_true anchor):**

| Regime | Mean MSE | Passes MSE ≤ 0.10? |
|--------|---------:|:------------------:|
| oracle | 0.049 | Yes (parent gate) |
| wrong20 | 0.629 | No |
| wrong40 | 0.844 | No |
| unknown_random | 1.032 | No |

Best stressed budget (wrong20 @ 36 rows, frac 0.6): mean MSE ≈ **0.31** — still **> 0.10**. No fraction for any wrong regime satisfies the parent gate at targets 0.05, 0.10, or 0.15.

**Structural finding:** Under `wrong20`/`wrong40`/`unknown_random`, `gard_oracle_graph` MSE often **equals** `shard_style_ls` (e.g. wrong20 frac 1.0: both ≈ 2.79 on all seeds). Graph prior does not rescue assignment error—consistent with parent Round 05 supervisor narrative.

### 3. Criterion A under wrong assignment — **CONFIRMED FAIL**

`config_breakthrough`:

- `A_row_reduction_25pct_oracle_parent_gate`: **true** (75% ≥ 25%)
- `A_row_reduction_25pct_wrong40_parent_gate`: **false**
- `A_row_reduction_25pct_unknown_random_parent_gate`: **false**

`wrong20` is tested in `threat_reduction_by_assignment` and proposal/revision_log but **not** exported as a separate boolean in `config_breakthrough` — supervisor confirms **wrong20: false** by inspection (same `parent_mean_mse_gate` structure as wrong40).

**Implication:** Config criterion **A** is **not met** for any deployment-relevant threat model in this run. Oracle-only 75% is **replication**, not a new breakthrough signal.

---

## Round 01 mandatory demands — compliance

| # | Demand | Status |
|---|--------|--------|
| 1 | Assignment stress (wrong20/40/unknown) | **Done** |
| 2 | Graph stress (noisy, wrong, co-occurrence, LS-kNN) | **Done** |
| 3 | Parent threat metric + per-seed pass rates; pre-register gate | **Done** (`parent_mean_mse_at_minimum_rows`) |
| 4 | Mean-anchor ablation | **Done** (oracle / noisy / none / level1) |
| 5 | Ridge + low-rank PCA baselines | **Done** |
| 6 | MSE targets {0.05, 0.10, 0.15} | **Done** |
| 7 | ASSIGN-LOCK stub | **Done** (permute slots; lock LS MSE ≈ 1.30 vs oracle GARD ≈ 0.72) |
| 8 | SHIELD utility grid ≤10% | **Done** — 0 feasible cells |
| 9 | SHIELD kill / deprioritize | **Done** (`kill_criteria_met: true`) |
| 10 | LEAK-CERT park or tier-specific T1p bound | **Parked** (documented) |
| 11 | `researcher_proposal.md` pass/fail vs A/B/C | **Done** |
| 12 | New JSON/log; Round 01 artifacts untouched | **Done** |
| 13 | Revision log states 85% withdrawn | **Done** |

Round 02 **closes the metric overclaim** and **opens** assignment-first science. It does **not** close the breakthrough gap.

---

## GARD-SPARSE / Stage-2 (supervisor audit)

### Oracle lane (upper bound only)

- **75%** @ MSE 0.10 with parent gate matches parent TALON conditional result (15/60 rows, mean MSE ≈ 0.049 @ frac 0.25).
- Graph ablation @ frac 0.25: only `gard_oracle_graph` (0.049) meets gate; `gard_noisy_graph` (0.33), wrong/co-occurrence/LS-kNN (~0.66–0.66) **fail** — graph mismatch already breaks criterion A under oracle assignment.
- Mean anchor @ frac 0.15 (GARD oracle graph): `oracle_true` 0.130, `noisy_mean` 0.145, `no_anchor` 0.194, `level1_estimate` 0.362 — oracle mean still strongest; level1 inadequate at this sparsity.

### Stressed assignment (deployment-relevant)

- **No** regime achieves mean MSE ≤ 0.10 at any tested row budget.
- GARD does not beat plain LS under wrong incidence in logged runs; investing further in oracle-chain GARD without assignment recovery is **mis-prioritized** unless labeled diagnostic.

### ASSIGN-LOCK stub

- Permutation raises attacker LS MSE to ~1.3 (near broken-assignment scale) vs oracle GARD ~0.72 at comparable fractions.
- `mean_slot_overlap: 1.0` — overlap metric does not capture permutation semantics; Round 03 must treat slot-order hiding as **necessary not sufficient**.

### QFL-SHIELD

- 0/100 grid cells with normalized utility ≤ 10%; criterion **B** remains failed.
- SHARD convergence warnings in log — acceptable for deprioritized lane.

### LEAK-CERT

- Correctly parked; criterion **C** unmet.

---

## Critical issues (Round 02 → carry to Round 03)

1. **No breakthrough under stressed assignment** — REVISE_MAJOR stands; oracle 75% alone cannot satisfy `require_measurable_breakthrough` for realistic FL.
2. **Oracle side channels remain** for the only passing criterion-A path: true incidence, chain graph aligned to synthetic index order, `oracle_true` mean anchor.
3. **ASSIGN-LOCK is a stub** — permutation without incidence hiding / secure aggregation does not close the assignment gap.
4. **Graph inference not validated under wrong assignment** — co-occurrence / LS-kNN arms exist but do not beat assignment collapse.

## Major issues

5. `config_breakthrough` should expose `A_row_reduction_25pct_wrong20_parent_gate` explicitly (wrong20 tested but only wrong40/unknown in JSON flags).
6. No confidence intervals on row-reduction tables (5 seeds).
7. `README.md` supervisor column still empty for round 2 — update after this review.

## Minor issues

8. Parent Round 05 also tests `prefix_epochs` protocol — not ported; optional for Round 03 if ASSIGN-LOCK matures.
9. SHARD vendor track still shows convergence warnings — keep synthetic Stage-2 and vendor track **separate** in prose.

---

## Config / policy compliance

| Policy | Round 02 |
|--------|----------|
| `forbid_accept_before_round: 8` | **ACCEPT forbidden** (round 2 of 10) |
| `require_measurable_breakthrough` | **Not met** for non-oracle threat model |
| `forbid_accept_on_negative_result_only` | B/C negative; A is replication not independent signal |
| `early_accept_requires_two_independent_breakthrough_signals` | **Not satisfied** |
| `strict_no_early_accept` | **Honored** — proposal does not seek acceptance |

**NO ACCEPT before round 8** — reaffirmed. Round 02 does not warrant exception.

---

## Round 03 demands (mandatory)

Round 02 answered “what breaks under honest metrics?” Round 03 must deliver **at least one** of:

### Path 1 — Novel mechanism under wrong incidence (primary)

1. **Non-oracle Stage-2 win** for `wrong20` and/or `unknown_random`: attacker must **not** use true H; mechanism may combine ASSIGN-LOCK (permutation + incidence hiding), co-occurrence / LS-kNN graph **learned from observations**, and/or prefix-epoch protocol from parent Round 05.
2. **Pre-registered success:** parent mean gate @ MSE **0.10** with ≥**25%** row reduction vs full observation — same gate as Round 02; report wrong20 explicitly in JSON.
3. **Forbidden as “novel”:** oracle-assignment GARD alone; relabeling conditional 75% replication.

### Path 2 — Defense breakthrough (criterion B)

4. If SHIELD revived: demonstrate ≥**50%** T1p (and Stage-2) attack MSE reduction vs undefended with normalized utility ≤ **10%** on at least one grid cell; document rank/σ.
5. If still 0 feasible cells after redesign: **kill** SHIELD for remainder of run (one paragraph, no grid re-runs).

### Path 3 — LEAK-CERT at T1p budget (criterion C)

6. Implement tier-specific rank/sensitivity bound at **actual** T1p partial row budget (**7 rows/epoch** per LASA-QTERM), with proved bound ≥**2×** tighter than naive mean-only baseline (clarify “tighter” = smaller leak metric in code and prose).

### Deliverables (Round 03)

7. `code/benchmark_round03.py` (or extend round02 with flags) → new JSON/log.
8. `rounds/round_03/researcher_proposal.md` — which path pursued, pass/fail vs A/B/C, explicit comparison to Round 02 stressed-assignment negatives.
9. `rounds/round_03/revision_log.md` — pivot rationale if primary lane shifts.

---

## Scientist alignment

Round 02 proposal, revision log, and `honesty_notes` are **aligned with supervisor findings**. The team correctly reframed GARD-SPARSE as an **upper bound** and deprioritized SHIELD. The remaining gap is **mechanism**, not **measurement hygiene**.

---

## Supervisor sign-off

| Field | Value |
|-------|-------|
| Round | 2 |
| Verdict | **REVISE_MAJOR** |
| 85% withdrawn | **Yes** (verified in JSON + re-run) |
| Assignment stress | **Yes** (verified) |
| Criterion A @ wrong assignment | **Fail** (verified) |
| Criterion A @ oracle (parent gate) | **75%** — conditional replication only |
| Criterion B | **Fail** |
| Criterion C | **Fail / parked** |
| Breakthrough under stressed assignment | **No** |
| Accept | **No** (policy round ≥ 8 + substance) |
| Next round focus | Novel wrong20/unknown mechanism, or defense B, or T1p cert C |
