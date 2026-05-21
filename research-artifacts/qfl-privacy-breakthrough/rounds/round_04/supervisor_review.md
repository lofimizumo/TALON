# Round 04 — Supervisor Review

## Verdict

**REVISE_MAJOR**

Round 04 is a **valid multi-path execution round**: JASPER-Q, ASSIGN-LOCK v2, and LEAK-CERT-T1p are implemented, reproducible, and honestly reported. **None** satisfy `config.json` breakthrough criteria **A**, **B**, or **C** under stressed assignment. JASPER-Q delivers a **large diagnostic gain** vs fixed-H GARD under `wrong40`, but snapshot MSE remains **≫ 0.10** and the parent row-reduction gate is **0/5 seeds** at all tested fractions. ASSIGN-LOCK v2 **fails** and is **killed** for the remainder of the run. LEAK-CERT-T1p **fails** criterion **C**. Policy forbids **ACCEPT** before round 8 regardless.

---

## Executive summary

Deliverables (`code/benchmark_round04.py`, `artifacts/round04_metrics.json`, `logs/experiment_round04.log`, `rounds/round_04/researcher_proposal.md`, `rounds/round_04/revision_log.md`) are **runnable and aligned** with proposal claims. Supervisor re-ran `python3 code/benchmark_round04.py` (~21s); criteria flags match submitted JSON (`A oracle=False wrong40=False C=False`).

**Round 04 mission (from Round 03 supervisor) — outcome:**

| Path | Round 04 action | Result |
|------|-----------------|--------|
| **Path 1** — JASPER-style soft assignment | JASPER-Q + T1p warm-start (65%) + entropy blend | **Partial diagnostic only** — wrong40 MSE **0.957** @ 25% rows; **not** barrier break; oracle path **regresses** |
| **Path 2** — ASSIGN-LOCK `wrong40` gain | Hungarian permutation recovery v2 | **Fail** — **kill** lane (no further stub reruns) |
| **Path 3** — Criterion **B** (SHIELD) | Not re-run (Round 02 kill) | **Fail / N/A** |
| **Path 4** — LEAK-CERT @ T1p budget | Tier bound @ 70 rows | **Fail** — ratio **1.89×**; cert invalid seed 19 |

Round 04 correctly closes the **“soft assignment + T1p prior without crypto”** hypothesis class as a **criterion-A breakthrough**. It does **not** close the research run.

---

## Breakthrough gates (`config.json`)

| Criterion | Round 04 JSON / proposal | Supervisor ruling |
|-----------|--------------------------|-------------------|
| **A** — ≥25% row reduction @ equal snapshot MSE (parent mean gate @ 0.10) | `A_row_reduction_25pct_wrong40_parent_gate_jasper_q`: **false**; `A_row_reduction_25pct_oracle_parent_gate_jasper_q`: **false** | **Fail** — wrong40 mean MSE **0.955** @ 20 rows; oracle JASPER-Q **0.853** @ 20 rows; **0/5** seeds pass any target {0.05, 0.10, 0.15} |
| **B** — ≥50% attack MSE cut @ ≤10% utility | Unchanged from Round 02 kill | **Fail / N/A** |
| **C** — cert ≥2× tighter than naive | `C_cert_2x_tighter_than_naive`: **false**; `criterion_c_pass_rate`: **0/5** | **Fail** — mean `naive/cert` **1.89**; cert **does not** upper-bound empirical on seed 19 |

Oracle conditional replication from Round 02 is **not extended**; JASPER-Q **worsens** oracle-assignment MSE vs GARD under `level1_estimate` (see audit §2).

---

## Verification audit

### 1. JASPER-Q @ fraction 0.25, `level1_estimate` — **DIAGNOSTIC, NOT BREAKTHROUGH**

From `jasper_q.aggregate` (supervisor re-run matches):

| Assignment | GARD (wrong **H**) | JASPER-Q | Hard overlap | Δ (JASPER − GARD) |
|------------|-------------------:|---------:|-------------:|------------------:|
| wrong20 | 1.334 | **0.902** | 0.740 | **−0.432** |
| wrong40 | 2.653 | **0.957** | 0.483 | **−1.696** |
| unknown_random | 2.277 | **1.010** | 0.120 | **−1.267** |
| oracle | 0.402 | **0.853** | 1.000 | **+0.451** (regression) |

**Ruling:** Under stressed assignment, JASPER-Q is a **strong wrong-H diagnostic module** (≈64% relative cut vs GARD on `wrong40`), but snapshot MSE **≫ 0.10** and **≫ 0.15**. This is **not** criterion **A** and **not** Round 03 Path 1 pre-registration (wrong40 mean **≤ 0.85** vs Round 03 co-occurrence **≈ 1.031**): achieved **0.957** — only **~7%** relative improvement vs Round 03 barrier, below the **≥15%** threshold. Overlap **0.483 ≥ 0.35** is met but **insufficient** without MSE gate pass.

**Mechanism attribution:** Gain tracks **T1p warm-start** (honest 70-row terminal channel per `honesty_notes`), not assignment recovery. Under `oracle` assignment, warm-start **hurts** vs GARD (**0.853** vs **0.402**) — terminal prior mismatches full Stage-2 incidence. **Do not** relabel JASPER-Q as “assignment barrier broken”; `assignment_barrier_broken: false` is **correct**.

### 2. Parent mean-MSE threat gate — **CONFIRMED FAIL**

`jasper_q.threat_reduction.wrong40_level1` @ fraction 0.25 (20 rows):

| target_mse | mean_mse | per_seed_pass_rate |
|------------|---------:|-------------------:|
| 0.05 | 0.955 | **0/5** |
| 0.10 | 0.955 | **0/5** |
| 0.15 | 0.955 | **0/5** |

`reaches_target: false` for all targets; `min_observed_rows: null`. Same failure @ fraction 0.15 (12 rows, mean **0.970**). Compare Round 02 oracle-conditional ~**0.844** @ wrong40 — still above gate, but JASPER-Q under level1 does **not** approach Round 02 oracle wins.

### 3. ASSIGN-LOCK v2 — **CONFIRMED FAIL; LANE KILLED**

`assign_lock_v2.aggregate` @ fractions {0.25, 0.15}:

| Metric | Mean |
|--------|-----:|
| Permutation recovery accuracy | **12.9%** |
| v1 broken LS MSE | 1.325 |
| v2 recovered LS / GARD MSE | **1.325** (identical) |
| Oracle-assignment GARD upper bound | 0.723 |

**Ruling:** Hungarian alignment on published row order **does not** recover true batch membership or reduce snapshot MSE. Round 03 demanded: if permutation-only fails again, **kill** ASSIGN-LOCK for rounds **5–10** with a written paragraph — **enforced below**. No further ASSIGN-LOCK stub reruns without **cryptographic permutation + incidence/metadata hiding** (new proposal path, not v2 gradient matching).

### 4. LEAK-CERT-T1p — **CONFIRMED FAIL (criterion C)**

`leak_cert_t1p.aggregate`:

| Metric | Value |
|--------|------:|
| Empirical T1p attack MSE (mean) | 0.439 |
| Cert tight upper (mean) | 0.529 |
| Naive mean broadcast | 1.000 |
| `naive/cert` ratio (mean) | **1.891** (< 2×) |
| Cert covers empirical | **4/5** seeds (seed **19**: emp **0.569** > cert **0.529**) |
| `qfl-terminal-snapshot` reference | 0.476 |

**Ruling:** Implementation is a useful **heuristic** tier bound, not a breakthrough certificate. Fails **2× tightening** and **valid upper bound on all seeds**. Do not spend Round 05 tuning LEAK-CERT constants without a new proof object.

### 5. Round 03 / Round 04 mandatory carry-over — **PARTIAL**

| Demand | Round 04 status |
|--------|-----------------|
| JASPER-style soft assignment (Path 1) | **Done** — diagnostic only |
| ASSIGN-LOCK hardening (Path 2) | **Done** — **failed; killed** |
| LEAK-CERT @ T1p budget (Path 4) | **Done** — **failed** |
| SHIELD / criterion **B** | **N/A** (prior kill) |
| `wrong20` / `unknown_random` in `config_breakthrough` | **Still missing** |
| **Used vs true** observed-batch residual MSE (Round 06 audit) | **Still missing** in JSON |
| Extend threat table to fraction **0.50** | **Not done** |

---

## Critical issues (Round 04 → carry to Round 05)

1. **No criterion-A breakthrough** — wrong40 JASPER-Q @ 25% rows is **~0.96 MSE**, **9.6×** above the 0.10 gate; relaxing to **0.15** still fails @ 25% rows (Round 05 may test **50%** rows explicitly).
2. **JASPER-Q confounds tiers** — T1p honest warm-start drives wrong-H gains; oracle path regression proves this is **not** generic Stage-2 assignment recovery. Round 05 must **decouple** terminal prior (wrong-H only) if JASPER remains an ablation.
3. **ASSIGN-LOCK killed** — permutation/Hungarian without hiding is **empirically dead**; cite this review and do not rerun v2.
4. **Criterion C dead at current bound** — 1.89× ratio and seed-19 coverage gap; need new math or abandon **C** for acceptance planning.

## Major issues

5. `config_breakthrough` still omits `A_row_reduction_25pct_wrong20_parent_gate` and `unknown_random` despite testing both regimes.
6. No **used-vs-true** residual split — overlap can look reasonable (~0.48) while snapshot MSE stays ~1.0; parent Round 06 lesson still unaddressed.
7. Fraction grid stops at **0.25** — supervisor Round 05 demand includes **0.50** (50% rows) for relaxed MSE **0.15** milestone on `wrong40`.
8. `README.md` supervisor column empty for round 4 — update after this review.

## Minor issues

9. Consider fraction **0.40** only if Path 2 (50% rows) shows monotonic MSE drop — parent Round 06 used 40% heavily.
10. LEAK-CERT `cert_rank_floor_70rows: 0` — dof term inactive; document if revisiting cert lane in later rounds.

### ASSIGN-LOCK formal kill (rounds 5–10)

**ASSIGN-LOCK** (permutation publish + gradient-only Hungarian recovery) is **removed** from the primary plan. Round 02 stub and Round 04 v2 show broken LS MSE **equals** recovered MSE; permutation recovery **~13%** accuracy. Any future “assignment hiding” must specify **secure aggregation / encrypted permutation / metadata suppression** and a new attacker model—**not** a third Hungarian variant.

---

## Config / policy compliance

| Policy | Round 04 |
|--------|----------|
| `forbid_accept_before_round: 8` | **ACCEPT forbidden** (round 4 of 10) |
| `require_measurable_breakthrough` | **Not met** |
| `forbid_accept_on_negative_result_only` | JASPER diagnostic + ASSIGN-LOCK/LEAK-CERT negatives insufficient |
| `early_accept_requires_two_independent_breakthrough_signals` | **Not satisfied** |
| `strict_no_early_accept` | **Honored** |

**NO ACCEPT before round 8** — reaffirmed. Round 04 does not warrant exception.

---

## Round 05 demands (mandatory — pick exactly one primary path)

Round 05 must deliver **one** primary path with **pre-registered success criteria** in `rounds/round_05/researcher_proposal.md` **before** runs. JASPER-Q hyperparameter sweeps and ASSIGN-LOCK reruns are **forbidden** unless listed as non-primary ablations.

### Path 1 — Snapshot-DP defense (criterion **B** primary)

1. Implement **Snapshot-DP**: calibrated noise / clipping on **published gradient snapshots** (Stage-2 and, if in scope, T1p terminal snapshots) with explicit \((\varepsilon, \delta)\) bookkeeping in code—not the Round 03 label-only `dp_mean` ablation.
2. Attacker: same LASA-QFL snapshot reconstruction stack **without** defender oracle channels.
3. **Pre-registered success:** ≥**50%** reduction in attacker snapshot MSE vs undefended baseline on **`wrong40` + `level1_estimate`** at fraction **0.25**, with reported utility loss ≤ **10%** (normalized metric defined in benchmark, same contract as Round 02 SHIELD grid).
4. If **0** feasible \((\varepsilon,\) clip, rank)\) cells: **formal kill** of Snapshot-DP for rounds 6–10 (one paragraph) and pivot to Path 2 or 3.

### Path 2 — Relaxed row-reduction milestone (`wrong40`, 50% rows)

5. Extend fraction grid to include **0.50** (30/60 observed rows). Method: best honest stack from Rounds 03–04 (**GARD co-occurrence** and/or **JASPER-Q with T1p warm-start disabled on oracle path**).
6. **Pre-registered success (intermediate, not full criterion A):** `wrong40` + `level1_estimate`, fraction **0.50**, mean snapshot MSE **≤ 0.15** on **≥4/5** seeds.
7. Stretch (criterion **A**): if MSE **≤ 0.10** at **0.50** fraction with **≥25%** row reduction vs full 60 rows, document as breakthrough candidate—but supervisor will still require stressed-assignment pass without oracle mean/T1p oracle membership cheats.

### Path 3 — Assignment barrier theorem (limits breakthrough)

8. Prove a formal **assignment barrier** proposition for the Stage-2 surrogate: under wrong-/unknown-incidence regimes and realistic mean anchor, no Lipschitz attacker map from published snapshots to batch membership recovery below a stated error floor without side information.
9. **Pre-registered success:** `paper/assignment_barrier.md` (or `literature/`) with lemma + proof sketch tied to simulator parameters (`n_samples`, `batch_size`, `wrong40` corruption model); benchmark JSON cites theorem IDs; no new mechanism required if proof is complete and falsifiable predictions match Round 04 aggregates (e.g., MSE floor **≥ 0.15** at 25% rows under level1).

### Deliverables (Round 05)

10. `code/benchmark_round05.py` → `artifacts/round05_metrics.json`, `logs/experiment_round05.log`.
11. `rounds/round_05/researcher_proposal.md` — **which path**, thresholds, pass/fail vs A/B/C.
12. `rounds/round_05/revision_log.md` — pivot rationale; **ASSIGN-LOCK kill** referenced.
13. Extend `config_breakthrough` with `wrong20`, `unknown_random`, and path-specific flags (e.g., `B_snapshot_dp_wrong40_50pct_cut`, `A_wrong40_mse_0.15_at_50pct_rows`, `assignment_barrier_theorem_stated`).

**Round 05 may not receive ACCEPT** even if Path 2 intermediate milestone passes; acceptance remains blocked until round **8** and requires `require_measurable_breakthrough` on the config definition (full **A**, **B**, or **C**).

---

## Scientist alignment

Proposal, revision log, and `honesty_notes` are **aligned** with metrics. The team correctly reports `assignment_barrier_broken: false` and labels JASPER-Q gains as **not** criterion-A grade. Supervisor **rejects** any Round 05 plan that treats JASPER-Q wrong40 MSE **~0.96** as a breakthrough without gate pass. Round 04 closes soft-assignment + T1p-warm-start + permutation-recovery + current LEAK-CERT bound as **primary** lanes.

---

## Supervisor sign-off

| Field | Value |
|-------|-------|
| Round | 4 |
| Verdict | **REVISE_MAJOR** |
| Mechanisms tested | JASPER-Q, ASSIGN-LOCK v2, LEAK-CERT-T1p |
| JASPER-Q vs GARD wrong40 @ 0.25 | **Diagnostic yes; breakthrough no** |
| Assignment barrier broken | **No** |
| ASSIGN-LOCK | **Killed** (rounds 5–10) |
| Criterion A @ wrong40 (JASPER-Q) | **Fail** |
| Criterion A @ oracle (JASPER-Q) | **Fail** (regression vs GARD) |
| Criterion B | **Fail / N/A** |
| Criterion C | **Fail** |
| Accept | **No** (policy round ≥ 8 + substance) |
| Next round focus | **Snapshot-DP defense** OR **`wrong40` MSE ≤ 0.15 @ 50% rows** OR **assignment barrier theorem** |
