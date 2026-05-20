# Round 09 — Supervisor Review

## Verdict

**REVISE_MAJOR**

Round 09 executes **PROBE-RAND** honestly: per-round random probe matrices \(A^{(e)}\), client gradient MSE at noise floor, and JASPER snapshot MSE **unchanged** under wrong40 when the attacker knows per-epoch \(A\) (**≈ 0.897**) or uses stale \(A^{(0)}\) (**≈ 1.002** vs baseline **≈ 0.894**). The proposal’s **“honest failure”** verdict is **confirmed** by supervisor re-run (`honest_failure: true`). This is **not** criterion **B** progress and does **not** supply the second independent signal required for early accept. **ACCEPT is forbidden.**

---

## Executive summary

Deliverables: `code/benchmark_round09.py`, `artifacts/round09_metrics.json`, `logs/experiment_round09.log`, `rounds/round_09/researcher_proposal.md`, `rounds/round_09/revision_log.md`. Supervisor re-ran `python3 code/benchmark_round09.py` (~16.9s); aggregate decode means match JSON within float tolerance.

| Claim | Round 09 | Supervisor |
|-------|----------|------------|
| PROBE-RAND breaks JASPER under wrong40 | **false** | **Confirmed false** |
| Clients train with published \(A^{(e)}\) | grad MSE **≈ 9.9×10⁻⁵** | **Confirmed** |
| Stale decode materially hurts attacker | **~12%** MSE increase | **Confirmed — far below B gate** |
| Oracle decode (cheating) | **≈ 0.897** ≈ baseline | **Confirmed — co-occurrence not destroyed** |
| Honest failure reported | **yes** | **yes — no overclaim** |

---

## PROBE-RAND failure — honest?

**Ruling: yes — failure is real, pre-registered lane, and correctly interpreted.**

### Aggregate (5 seeds, wrong40 @ 50%, `level1_estimate`)

| Condition | JASPER MSE mean | vs baseline **0.894** |
|-----------|----------------:|----------------------:|
| Round 07-style baseline (snapshot rows) | **0.894** | — |
| PROBE-RAND + `oracle_per_epoch` decode | **0.897** | **~0.3%** (cheating upper bound) |
| PROBE-RAND + `stale_first_epoch` | **1.002** | **~12%** worse — still **𝒪(1)** |
| PROBE-RAND + `pooled_lstsq` | **231.6** | Catastrophic — **naive defender artifact**, not practical protocol |
| Client grad MSE (PROBE-RAND) | **9.86×10⁻⁵** | Noise floor — utility preserved |

**Mechanism (accepted):** Randomizing \(A^{(e)}\) does not remove wrong40 assignment error as the dominant term; with correct per-epoch \(A\), decoded snapshots reproduce baseline leakage. Stale \(A^{(0)}\) slightly degrades decode but **not** to barrier targets (**0.10–0.15**). Threat model note in JSON is **correct**: a server that logs \(A^{(e)}\) collapses to `oracle_per_epoch`.

**Breakthrough criterion B check:**

| B requirement | Result |
|---------------|--------|
| ≥50% attack MSE increase | **~12%** (stale) — **Fail** |
| ≤10% utility loss | **Pass** (grad MSE negligible) |
| Joint B pass | **Fail** |

No cherry-picking detected: all five seeds reported; `verdict` block matches aggregates.

---

## Honesty / reproducibility audit

1. **Re-run fidelity** — Stale-decode means drift **<2%** vs submitted JSON (e.g. seed 3 stale **0.990** vs stored **0.993**); same failure class.
2. **Decode ablation grid** — Four attacker decodes plus baseline; pooled-lstsq explosion is **not** hidden.
3. **Alignment with Round 07** — Baseline JASPER **≈ 0.894** matches Round 07 wrong40 @ 50% scale.
4. **No hardcoded breakthrough pass** — `probe_rand_breaks_jasper_without_per_epoch_a: false` computed in script.
5. **Single-lane discipline** — SUBSPACE-MIX deferred per proposal; acceptable for one high-risk lane.

**Major gap vs Round 07 Round-9 demands:** Replication was to use **held-out seeds** `{5, 13, 17, 29, 31}`; Round 09 reused `{3, 7, 11, 19, 23}`. Negative result is still credible, but **not** the pre-registered independence test.

---

## Feasibility & resources

- Runtime **~17s** for 5 seeds × decode grid — acceptable.
- Threat model explicitly scoped (attacker lacks per-round \(A\) unless logging) — good scientific hygiene.

---

## Theory & assumptions

- **Hypothesis** (random \(A^{(e)}\) destroys cross-epoch co-occurrence) is **falsified** under stated attacker decodes.
- **Legacy single-probe** (`stale_first_epoch`) is a fair weak-attacker model; result shows it is **insufficient** for defense, not that PROBE-RAND is safe.
- **Oracle decode row** documents upper bound — team correctly labels cheating path.

---

## Breakthrough gates (`config.json`)

| Criterion | Round 09 | Ruling |
|-----------|----------|--------|
| **A** | Not attempted | **Fail / N/A** (wrong40 unchanged) |
| **B** | PROBE-RAND defense | **Fail** |
| **C** | Not attempted | **Fail / N/A** |
| `forbid_accept_on_negative_result_only` | Honest negative | **Blocks ACCEPT** |
| `require_measurable_breakthrough` | Not met | **Fail** |

---

## Critical issues

1. **PROBE-RAND killed as breakthrough** — Correct team verdict; must appear in Round 10 kill list.
2. **No second independent signal** for `early_accept_requires_two_independent_breakthrough_signals` — negative lane only.
3. **Held-out seed replication skipped** — Round 07 § Round 9 demand not satisfied.

## Major issues

4. **Round 08 integration not extended** — PROBE-RAND not wired into `QFL-PRIVACY-MAP` tier table (optional but weakens Round 10 “integrated framework” story).
5. **Pooled-lstsq headline** (**231.6**) is dramatic but **irrelevant** to SHARD threat model — keep in appendix/diagnostic only.

## Minor issues

6. `hard_overlap` still exported in per-decode rows — do not cite as MSE (Round 07 lesson).
7. GARD MSE under PROBE-RAND is unstable (**≈ 3–5** stale) — report JASPER as primary wrong40 attacker (consistent with prior rounds).

---

## Rubric scores (1–5)

| Dimension | Score | Note |
|-----------|------:|------|
| Novelty | 2 | Straightforward probe randomization test |
| Soundness | 5 | Honest negative; threat model explicit |
| Feasibility | 4 | Reproducible single lane |
| Impact | 2 | Closes PROBE-RAND path; no defense gain |
| Evaluability | 4 | Clear decode ablation JSON |

---

## Round 10 demands (carry from Rounds 07–08 + this review)

Round 10 default remains **REVISE_MAJOR** unless **ACCEPT** is justified on the **integrated privacy audit framework** with **two independent signals**:

| Acceptable pair | Content |
|-----------------|---------|
| **ABT-1 + oracle sparse quant** | Theorem doc + wrong40 falsification checks **and** GARD-SPARSE 75% protocol spec (from Round 08 map) |
| **T1 impossibility + cert coverage** | Parent cite + reproduced T1 floor **and** T1p cert covers empirical with explicit **C 2× kill** |

**PROBE-RAND:** Add to kill list; do not relitigate in Round 10 unless new external attacker (e.g. gradient-logging server) is pre-registered in proposal **before** runs.

**Mandatory before ACCEPT:**

1. Held-out seeds `{5, 13, 17, 29, 31}` on **map bundle** (and any accept-cited metric).
2. `rounds/round_10/revision_log.md` kill list (include PROBE-RAND, honest **C**, trace **C**, Snapshot-DP, ASSIGN-LOCK).
3. `rounds/round_10/acceptance_decision.md` with explicit two-signal pair and **0 Critical** issues.
4. `README.md` supervisor table complete.
5. **Forbidden:** ACCEPT on PROBE-RAND failure narrative alone; ACCEPT implying hidden \(A^{(e)}\) from gradient logs without new experiments.

---

## Actionable suggestions (prioritized)

1. Add **PROBE-RAND** tier row to `privacy_map.md` as **KILLED / not breakthrough** (one line + Round 09 cite).
2. Run **held-out seed** replication for Round 08 map + any Round 10 claim.
3. Do **not** spend Round 10 on SUBSPACE-MIX unless proposal pre-registers a **stricter external attacker** and departs from killed lanes.
4. In accept memo, quote stale **1.002** vs **0.894** only as “insufficient for B,” not as partial success.

---

## Supervisor sign-off

| Field | Value |
|-------|-------|
| Round | 9 |
| Verdict | **REVISE_MAJOR** |
| PROBE-RAND | **Honest failure — confirmed** |
| Criterion **B** | **Fail** |
| Accept | **No** |
| Round 10 | **Integrated audit + two-signal pair; kill list + held-out seeds** |
