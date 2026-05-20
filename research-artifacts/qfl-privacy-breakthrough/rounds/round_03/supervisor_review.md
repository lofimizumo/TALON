# Round 03 — Supervisor Review

## Verdict

**REVISE_MAJOR**

Round 03 is a **valid negative-mechanism round**: four assignment-barrier hypotheses (INCIDENCE-REFINE, co-occurrence under wrong **H**, DP-mean anchor, HYBRID T1p + sparse GARD) are implemented, reproducible, and honestly reported. **None** break the stressed-assignment barrier; criterion **A** remains unmet for `wrong20` / `wrong40` / `unknown_random` under realistic anchors. Round 02’s **75%** row reduction @ MSE ≤ 0.10 survives only as **oracle-assignment + oracle-mean** conditional replication—not extended. Policy forbids **ACCEPT** before round 8 regardless.

---

## Executive summary

Deliverables (`code/benchmark_round03.py`, `artifacts/round03_metrics.json`, `logs/experiment_round03.log`, `rounds/round_03/researcher_proposal.md`, `rounds/round_03/revision_log.md`) are **runnable and aligned** with proposal claims. Supervisor re-ran `python3 code/benchmark_round03.py` (~2.3s); metrics match submitted JSON.

**Round 03 mission (from Round 02 supervisor) — outcome:**

| Path | Round 03 action | Result |
|------|-----------------|--------|
| **Path 1** — Novel mechanism under wrong incidence | INCIDENCE-REFINE, co-occurrence wrong-H, DP-mean, hybrid | **Fail** — `assignment_barrier_broken: false` |
| **Path 2** — Criterion **B** (SHIELD) | Not re-run (Round 02 kill) | **Fail / N/A** (deprioritized OK) |
| **Path 3** — Criterion **C** (LEAK-CERT) | Not implemented | **Fail / N/A** |

**Not pursued in Round 03 (correctly deferred, but now blocking):** parent **JASPER** soft assignment; **ASSIGN-LOCK** hardening beyond Round 02 stub. Scientist recommendation matches supervisor priority for Round 04.

---

## Breakthrough gates (`config.json`)

| Criterion | Round 03 JSON / proposal | Supervisor ruling |
|-----------|--------------------------|-------------------|
| **A** — ≥25% row reduction @ equal snapshot MSE (parent mean gate) | `A_row_reduction_25pct_wrong40_parent_gate_refine`: **false**; `A_row_reduction_25pct_oracle_parent_gate_level1_co`: **false** | **Fail** — no stressed-assignment or realistic-anchor win |
| **B** — ≥50% attack MSE cut @ ≤10% utility | Unchanged from Round 02 kill | **Fail** |
| **C** — cert ≥2× tighter than naive | `C_cert_2x_tighter_than_naive`: **null** | **Fail / N/A** |

Oracle conditional replication from Round 02 is **not re-validated** under `level1_estimate` + co-occurrence in Round 03 (`A_row_reduction_25pct_oracle_parent_gate_level1_co: false`)—expected regression when oracle mean anchor is removed.

---

## Verification audit

### 1. Assignment barrier @ fraction 0.25, `level1_estimate` anchor — **CONFIRMED FAIL**

From `assignment_barrier.cooccurrence_vs_oracle_at_0.25` (supervisor re-run matches):

| Regime | Co-occurrence MSE | Oracle chain (wrong **H**) | INCIDENCE-REFINE | Δ (co − chain) |
|--------|------------------:|---------------------------:|-----------------:|---------------:|
| wrong20 | 0.960 | 0.672 | 0.971 | **+0.288** (co worse) |
| wrong40 | 1.031 | 0.892 | 1.037 | **+0.139** (co worse) |
| unknown_random | 1.053 | 1.098 | 1.151 | −0.045 (noise only) |

**Ruling:** Learned co-occurrence graph **does not** beat a mismatched oracle-chain prior under `wrong20`/`wrong40`. Greedy INCIDENCE-REFINE **does not** improve snapshot MSE over co-occurrence alone (overlap metric can improve while MSE stays ≈1.0—self-consistent wrong mixture, consistent with parent TALON Round 06 JASPER narrative).

### 2. DP-mean anchor — **CONFIRMED NULL**

`dp_anchor_ablation_wrong40` @ fraction 0.25:

| Anchor | Co-occurrence | Refine |
|--------|--------------:|-------:|
| oracle_true | 1.023 | 1.003 |
| level1_estimate | 1.025 | 1.037 |
| dp_mean (ε=1.0) | 1.031 | 1.037 |

DP-mean is within noise of level1; neither approaches Round 02 oracle-anchor stressed curves (~0.13 MSE @ 15 rows with oracle assignment). **Not** a deployed DP-SGD guarantee (`honesty_notes` correct).

### 3. Threat reduction (parent mean-MSE gate @ 0.10) — **CONFIRMED FAIL**

`threat_reduction.wrong40_incidence_refine_level1`:

- All targets {0.05, 0.10, 0.15}: `reaches_target: false`, `min_observed_rows: null`
- @ 15 rows (fraction 0.25): mean MSE ≈ **1.037**, `per_seed_pass_rate`: **0/5**

Compare Round 02 `wrong40` + `gard_oracle_graph` + `oracle_true` @ fraction 0.25: mean MSE ≈ **0.844** (still fails gate, but much better than Round 03 level1 stack). Removing oracle mean anchor dominates incremental graph/refine work.

### 4. HYBRID T1p + sparse GARD — **CONFIRMED NON-BREAKTHROUGH**

`hybrid_summary` (Hungarian MSE means):

| Assignment | T1p | Best hybrid | Sparse GARD (wrong **H**) |
|------------|----:|------------:|--------------------------:|
| wrong20 | 0.419 | 0.396 | 0.708 |
| wrong40 | 0.419 | 0.400 | 0.691 |

- `hybrid_beats_t1p_fraction_wrong40`: **0.2** (1/5 seeds only)
- Optimal blend **favors T1p** under wrong40 (mean α ≈ 0.85)
- T1p tier uses **true batch membership** for partial rows (`honesty_notes`)—cannot count as breaking **wrong-H** Stage-2 barrier

**Ruling:** Useful terminal track diagnostic; **not** criterion-A evidence under stressed assignment.

### 5. Upper bound sanity — **CONFIRMED**

With true **H** but level1/co anchors, `gard_oracle_h_upper_bound` @ wrong40 fraction 0.25 ≈ **0.587** MSE—still **> 0.10**. Assignment error in **H** alone does not explain Round 02 oracle win; **mean-anchor oracle** was load-bearing.

### 6. Round 02 mandatory carry-over — **PARTIAL**

| Round 02 demand | Round 03 status |
|-----------------|-----------------|
| Non-oracle mechanism attempt | **Done** (four mechanisms) |
| Pre-registered parent gate @ 0.10 | **Done** (reported; all fail) |
| No “novel” relabel of oracle 75% | **Honored** |
| `wrong20` in threat tables | **Done** (proposal + JSON) |
| `wrong20` boolean in `config_breakthrough` | **Still missing** |
| ASSIGN-LOCK / JASPER | **Not started** |
| SHIELD revival or formal kill paragraph | **Implicit kill** (no re-run; acceptable) |
| LEAK-CERT @ T1p budget | **Not started** |

---

## Critical issues (Round 03 → carry to Round 04)

1. **Assignment barrier unbroken** — all Round 03 mechanisms leave wrong40 mean MSE ≈ **1.0** @ 15 rows with level1 anchor; no path to ≥25% row reduction @ MSE 0.10.
2. **Hard relabeling is the wrong tool** — INCIDENCE-REFINE improves overlap (~0.25 mean @ wrong40/oracle_true diagnostic) without snapshot MSE gain; parent JASPER Round 06 already showed soft assignment alone fails—Round 04 must test **JASPER-style** joint soft incidence, not more greedy hard swaps.
3. **ASSIGN-LOCK stalled** — Round 02 stub proves permutation raises LS MSE; Round 03 did not combine hiding + inference. Round 04 must either harden ASSIGN-LOCK or drop it with written kill.
4. **Hybrid lane confounds tiers** — T1p oracle membership + wrong-H sparse rows is honestly labeled but must not be cited as wrong-assignment breakthrough.

## Major issues

5. `config_breakthrough` still omits `A_row_reduction_25pct_wrong20_parent_gate` (and `unknown_random`) — export all stressed regimes tested in `threat_reduction`.
6. No **used-vs-true** residual split for refine/JASPER-style methods (parent Round 06 lesson)—risk of celebrating low internal residual under wrong soft assignment.
7. Criterion **C** and **B** remain entirely unaddressed for five rounds of a 10-round plan—Round 04 must pick **one** alternate path explicitly if assignment lane fails again.

## Minor issues

8. Barrier fractions {0.25, 0.15} only—consider 0.40 if soft assignment shows marginal gains (parent Round 06 used 40% rows heavily).
9. `README.md` supervisor column empty for rounds 2–3 — update after this review.

---

## Config / policy compliance

| Policy | Round 03 |
|--------|----------|
| `forbid_accept_before_round: 8` | **ACCEPT forbidden** (round 3 of 10) |
| `require_measurable_breakthrough` | **Not met** |
| `forbid_accept_on_negative_result_only` | Round 03 is negative on assignment; hybrid marginalia insufficient |
| `early_accept_requires_two_independent_breakthrough_signals` | **Not satisfied** |
| `strict_no_early_accept` | **Honored** |

**NO ACCEPT before round 8** — reaffirmed. Round 03 does not warrant exception.

---

## Round 04 demands (mandatory — pick ≥1 path)

Round 04 must deliver **at least one** of the following with **pre-registered success criteria** recorded in `researcher_proposal.md` before runs:

### Path 1 — JASPER-style soft assignment (primary recommendation)

1. Port or reimplement parent TALON **JASPER** (joint soft incidence + Sinkhorn-style projection + MAP recovery; see `/workspace/rounds/round_06/`) in `code/benchmark_round04.py`.
2. **Forbidden:** greedy hard INCIDENCE-REFINE as the main novel claim; may remain ablation only.
3. **Pre-registered success (wrong40, level1 anchor, fraction 0.25, 15 rows):**
   - Mean snapshot MSE **≤ 0.85** (≥15% relative improvement vs Round 03 co-occurrence/refine ≈ **1.031**), **and**
   - Hard assignment overlap **≥ 0.35** (vs ~0.52 corrupted-H prior overlap),
   - Report **used** vs **true** observed-batch residual MSE (parent Round 06 audit).
4. Stretch goal (criterion **A**): parent mean gate `reaches_target` @ MSE **0.10** with ≥**25%** row reduction vs 60 rows under `wrong40` + level1—same gate as Rounds 02–03.

### Path 2 — ASSIGN-LOCK with measurable `wrong40` gain

5. Harden Round 02 stub: **permutation + incidence hiding** (and/or prefix-epoch protocol from parent Round 05), attacker model **without** true **H**.
6. **Pre-registered success:** @ `wrong40`, fraction 0.25, level1 anchor, mean attacker snapshot MSE **≤ 0.877** (≥15% relative vs Round 03 best Stage-2 barrier ≈ **1.031**) **or** criterion **A** parent gate pass as in Path 1 stretch goal.
7. Document defender vs attacker roles explicitly; if permutation-only fails again, **kill** ASSIGN-LOCK for remainder of run (one paragraph, no further stub reruns).

### Path 3 — Defense breakthrough (criterion **B**)

8. Revive SHIELD only with a **new** utility-feasible parameterization; demonstrate ≥**50%** T1p (and Stage-2) attack MSE reduction vs undefended with normalized utility ≤ **10%** on ≥1 grid cell.
9. If 0 feasible cells after redesign: **formal kill** in proposal (SHIELD removed from plan rounds 5–10).

### Path 4 — LEAK-CERT at T1p budget (criterion **C**)

10. Tier-specific rank/sensitivity bound at T1p partial row budget (**7 rows/epoch** per LASA-QTERM), proved ≥**2×** tighter than naive mean-only baseline on a defined leak metric in code.

### Deliverables (Round 04)

11. `code/benchmark_round04.py` → `artifacts/round04_metrics.json`, `logs/experiment_round04.log`.
12. `rounds/round_04/researcher_proposal.md` — **which path**, pre-registered thresholds, pass/fail vs A/B/C.
13. `rounds/round_04/revision_log.md` — pivot if Path 1 fails (do not spend Round 05 tuning failed greedy refine).
14. Extend `config_breakthrough` with `wrong20` and `unknown_random` booleans.

**Round 04 may not receive ACCEPT** even if a path shows marginal gains; acceptance remains blocked until round **8** and requires `require_measurable_breakthrough` on config definition.

---

## Scientist alignment

Proposal, revision log, and `honesty_notes` are **aligned** with metrics. The team correctly concludes `assignment_barrier_broken: false` and points to JASPER / ASSIGN-LOCK for Round 04—supervisor **endorses** that pivot. Round 03 closes the “graph + hard refine + DP anchor” hypothesis class under wrong incidence; it does **not** close the research run.

---

## Supervisor sign-off

| Field | Value |
|-------|-------|
| Round | 3 |
| Verdict | **REVISE_MAJOR** |
| Mechanisms tested | INCIDENCE-REFINE, co-occurrence wrong-H, DP-mean, HYBRID |
| Assignment barrier broken | **No** |
| Criterion A @ wrong40 (level1 + refine) | **Fail** |
| Criterion A @ oracle (level1 + co) | **Fail** |
| Criterion B | **Fail / N/A** |
| Criterion C | **Fail / N/A** |
| Round 02 oracle 75% replication | **Unchanged; not extended** |
| Accept | **No** (policy round ≥ 8 + substance) |
| Next round focus | **JASPER-style soft assignment** OR **ASSIGN-LOCK wrong40 gain** OR **defense B** OR **cert C** |
