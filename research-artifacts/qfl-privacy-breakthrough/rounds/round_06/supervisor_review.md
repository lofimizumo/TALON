# Round 06 — Supervisor Review

## Verdict

**REVISE_MAJOR**

Round 06 is an **honest dual-path execution round**: Path 2 (relaxed `wrong40` @ 50% rows) and Path 3 (`ABT-1` assignment barrier document) are implemented and reproducible. **Path 2 fails decisively** — **0/5** seeds meet mean snapshot MSE **≤ 0.15** under the pre-registered gate (best attacker JASPER-Q mean **≈ 0.896**). Path 3 delivers **scientific value** as a limits object (theorem + falsifiable P1 check at 25% rows) but is **not** a `config.json` breakthrough. Policy forbids **ACCEPT** before round 8 regardless.

---

## Executive summary

Deliverables present: `code/benchmark_round06.py`, `artifacts/round06_metrics.json`, `logs/experiment_round06.log`, `rounds/round_06/researcher_proposal.md`, `rounds/round_06/revision_log.md`, `paper/assignment_barrier_theorem.md`. Supervisor re-ran `python3 code/benchmark_round06.py` (~5.4s); `A_wrong40_mse_0.15_at_50pct_rows: false` and `path2_relaxed_milestone_pass: false` match submitted JSON (minor float drift on JASPER mean **0.8966** vs stored **0.908** — same fail).

**Round 06 mission (Round 05 demands) — outcome:**

| Check | Result |
|-------|--------|
| Path 2 — `wrong40` + `level1_estimate` @ fraction **0.50** | **Fail** — **0/5** seeds **< 0.15** |
| Best attacker @ 50% (min GARD, JASPER) | JASPER-Q mean **≈ 0.896** (GARD **≈ 1.156**) |
| Oracle T1p warm-start disabled | **Yes** (`t1p_warm_blend=0`) |
| Used-vs-true residual MSE exported | **Yes** |
| T1p audit `level1_mean_recovery` | **Yes** (mean **≈ 0.438**) |
| Path 3 — `ABT-1` document + benchmark cite | **Yes** |
| Theorem P1 @ 25% rows (JASPER mean **≥ 0.15**) | **Holds** (observed **≈ 0.953**) |
| Snapshot-DP grid | **Absent** (killed R05) — **correct** |

---

## Breakthrough gates (`config.json`)

| Criterion | Round 06 JSON / proposal | Supervisor ruling |
|-----------|--------------------------|-------------------|
| **A** — ≥25% row reduction @ parent mean gate **0.10** | `A_row_reduction_25pct_wrong40_parent_gate`: **false** | **Fail / N/A** (not primary) |
| **A (relaxed)** — `A_wrong40_mse_0.15_at_50pct_rows` | **false** (0/5 seeds) | **Fail** — see Path 2 audit |
| **B** — defense @ ≤10% utility | **false** (killed lane) | **Fail / N/A** |
| **C** — cert ≥2× tighter than **naive mean-only** | **false** | **Fail / N/A** |

**NO ACCEPT before round 8** — reaffirmed. A negative Path 2 + a limits theorem sketch does **not** satisfy `require_measurable_breakthrough`.

---

## Verification audit

### 1. Path 2 gate — **CONFIRMED FAIL**

From `path2_primary` (supervisor re-run):

| Attacker | Mean MSE @ 50% | Seeds pass @ ≤0.15 | Gate `passes_gate` |
|----------|---------------:|-------------------:|:------------------:|
| GARD co-occurrence | **1.156** | **0/5** | **false** |
| JASPER-Q | **0.896** | **0/5** | **false** |
| Best (JASPER) | **0.896** | **0/5** | **false** |

Per-seed JASPER MSE spans **0.889–0.916** — roughly **6×** above the **0.15** intermediate target and still **≫ 0.10** full criterion **A**. Increasing rows from Round 04’s 25% (**~0.96**) to 50% yields only **~7%** relative improvement, not barrier break.

**Ruling:** Pre-registered Path 2 milestone is **failed**. `assignment_barrier_broken: false` is **correct**. Do **not** relabel **~0.90** MSE as “near gate” without seed-level pass.

### 2. GARD vs JASPER protocol asymmetry — **CRITICAL**

Path 2 compares attackers on **different incidence universes**:

| Lane | Config source | `n_epochs` | `n_samples` | `full_rows` | `observed_rows` @ 50% |
|------|---------------|------------|-------------|-------------|----------------------:|
| GARD | `Stage2Config()` default | **5** | **48** | **60** | **30** |
| JASPER-Q | `stage_cfg_from_qfl(QflConfig)` | **10** | **32** | **80** | **40** |

Proposal text says **“30 / 60 rows”** but JASPER-Q JSON reports **40 / 80**. At the same nominal fraction **0.50**, JASPER receives **33% more observed batch rows** than GARD. Any cross-attacker “best-of” comparison is **not apples-to-apples** until both lanes share `Stage2Config` dimensions (`n_samples`, `n_epochs`, `batch_size`, noise).

**Supervisor ruling:** Path 2 fail verdict stands (JASPER still **≫ 0.15** even with extra rows), but **Round 06 Path 2 headline numbers are not protocol-fair**. Round 07+ must **unify** Stage-2/QFL incidence construction before claiming attacker superiority.

### 3. Used-vs-true residual — **DELIVERED; INTERPRET WITH CARE**

Aggregate JASPER: `residual_mse_used_h` **≈ 0.175**, `residual_mse_true_h` **≈ 0.182**, `ratio_true_over_used` **≈ 1.04** — attacker fits **wrong** incidence nearly as well as **true** incidence on published means, while snapshot MSE stays **~0.90**. This **supports** Lemma 2 / decoupling in `ABT-1`, not Path 2 success.

**Anomaly (Major):** GARD seed **7** reports `residual_mse_used_h` **≈ 6×10⁻¹⁵** with `residual_mse_true_h` **≈ 0.323** (`ratio_true_over_used` **≈ 3.2×10¹¹**). Indicates numerical degeneracy or rank-collapse fit on used-\(H\) — flag in JSON, do not cite as typical.

### 4. Oracle ablation — **PARTIAL PASS**

`oracle_jasper_ablation` @ 50%: `t1p_warm_blend=0` on all seeds (**good**). Mean MSE **≈ 0.761** — better than wrong40 JASPER but still **5×** above **0.15**. Confirms Round 04/05 lesson: disabling warm-start on oracle is necessary but **insufficient** for assignment breakthrough.

### 5. T1p audit — **PASS (non-primary)**

`ShardAttacker.level1_mean_recovery` + `QtermAttack.T1P`: mean **≈ 0.438** (Round 05 flat-mean ref **0.427**). Lane is audit-only; does not affect Path 2 gate.

### 6. Assignment barrier theorem (`ABT-1`) — **VALUE: LIMITS / FALSIFICATION, NOT BREAKTHROUGH**

| Aspect | Assessment |
|--------|------------|
| Document present | **Yes** — `paper/assignment_barrier_theorem.md` |
| Proof status | **Sketch only** — Lemmas 1–2 + boxed floor not formal for deployment |
| P1 @ 25% rows | **Supported** — `observed_jasper_mean_mse_at_25pct` **≈ 0.953** **≥ 0.15** |
| P2 @ 50% rows | **Supported (failure of break)** — mean **≈ 0.896** **≥ 0.15** |
| P3 (used/true decoupling) | **Partially illustrated** — ratio **≈ 1.04** with high snapshot MSE |

**Supervisor ruling on theorem value:**

- **Positive:** Names the right impossibility mechanism (wrong incidence → rank/co-occurrence deficiency → MSE floor). Gives **falsifiable** predictions aligned with Rounds 04–06 aggregates. Useful for paper **limits** section and stopping JASPER hyperparameter churn.
- **Not breakthrough:** No proof tied to a stochastic channel model; constants (**0.15**) are simulator-calibrated; **does not** pass relaxed Path 2 or criterion **A**. `theorem_abt1_linked: true` is bibliographic only.

Treat **ABT-1** as **Round 06’s durable output** alongside an honest Path 2 kill — not as substitute for measurable **A**, **B**, or honest **C**.

### 7. Round 05 mandatory carry-over — **MOSTLY DONE**

| Round 05 demand | Round 06 status |
|-----------------|-----------------|
| Path 2 @ 50% rows | **Done** — **failed** |
| `A_wrong40_mse_0.15_at_50pct_rows` export | **Done** — **false** |
| Used-vs-true residual | **Done** |
| T1p `level1_mean_recovery` | **Done** |
| Oracle warm-start off | **Done** |
| Theorem `ABT-1` | **Done** (sketch) |
| Unified GARD/JASPER incidence grid | **Fail** |
| `README.md` supervisor column | **Still empty** — update after reviews |

---

## Critical issues (Round 06 → carry to Round 07+)

1. **Path 2 gate failed** — **0/5** seeds @ MSE **≤ 0.15**; best mean **≈ 0.896** @ 50% rows.
2. **Attacker protocol mismatch** — GARD **30/60** vs JASPER **40/80** at same fraction; invalidates strict cross-attacker comparison until fixed.
3. **Assignment barrier not broken** — theorem **predicts** floor; experiment **confirms** floor — opposite of breakthrough.

## Major issues

4. **ABT-1 is not publication-grade proof** — upgrade or label explicitly “simulator conjecture with empirical falsification.”
5. **GARD seed-7 residual blow-up** — investigate rank reporting vs fit degeneracy.
6. **JASPER still tier-confounded on `wrong40`** — `t1p_warm_blend=0.65` + `recover_t1p_snapshots` (Round 04 stack); gains vs GARD do not isolate assignment recovery.

## Minor issues

7. Float drift between runs (~1%) — acceptable; gate outcome unchanged.
8. `config_breakthrough` flags for `wrong20` / `unknown_random` still **false** / untested — expected.

---

## Path 2 formal status (rounds 7–10)

With **0/5** seeds at **≤ 0.15** @ 50% rows, the **relaxed Path 2 milestone** is **not met**. Further “stretch to 0.15” sweeps on the **same** JASPER-Q v4 stack without a **new observation model** or **unified protocol** are **low priority**.

**Allowed:** mechanism changes that address barrier (e.g. unified incidence, decoupled terminal prior, new defense class). **Forbidden:** claiming Path 2 “almost passed” at **~0.90** MSE.

---

## Config / policy compliance

| Policy | Round 06 |
|--------|----------|
| `forbid_accept_before_round: 8` | **ACCEPT forbidden** (round 6 of 10) |
| `require_measurable_breakthrough` | **Not met** |
| `forbid_accept_on_negative_result_only` | Path 2 fail + theorem sketch **insufficient** |
| `strict_no_early_accept` | **Honored** |

---

## Round 07 demands (mandatory — pre-register before runs)

Round 07 already executed scientist-side; these are the **supervisor requirements** that round should have satisfied (audit in Round 07 review).

1. **Unify incidence grid** — GARD and JASPER-Q v-next must share `n_samples`, `n_epochs`, `batch_size`, and `full_rows` for every fraction in the sweep.
2. **Fraction sweep** — `{0.25, 0.35, 0.50, 0.65, 0.75}` on `wrong40` and `oracle` with `level1_estimate`; report parent mean-MSE gate @ **{0.05, 0.10, 0.15}**.
3. **JASPER structural upgrade** — document what changed vs Round 04 (multi-epoch T1p, spectral graph, conditional warm-start); **no** criterion **A** claim without gate pass.
4. **LEAK-CERT audit lane** — if criterion **C** is attempted, use **`config.json` naive mean-only baseline** for the **2×** ratio; any alternate “trace-inflated naive” must be labeled **non-primary** and cannot satisfy **C** alone.
5. **Deliverables:** `code/benchmark_round07.py`, metrics, log, `revision_log.md`, `supervisor_review.md` (this round was missing — corrected in Round 07).
6. **Extend `config_breakthrough`** — export relaxed gates separately from standard **A**; do **not** overwrite parent **A** semantics.

**Round 06 may not receive ACCEPT** even if theorem P1 holds.

---

## Rubric scores (1–5)

| Dimension | Score | Note |
|-----------|------:|------|
| Novelty | 3 | First explicit barrier document + used/true metric |
| Soundness | 2 | Path 2 cross-attacker protocol mismatch |
| Feasibility | 5 | ~5–7s reproducible |
| Impact | 3 | Honest kill of 50% milestone; theorem useful for limits narrative |
| Evaluability | 4 | JSON thorough; row-count asymmetry hurts |

---

## Scientist alignment

Proposal, revision log, and `honesty_notes` **match** supervisor metrics on Path 2 fail and theorem role. Team correctly does **not** claim breakthrough. Supervisor **rejects** reframing **~0.90** MSE as Path 2 success.

---

## Supervisor sign-off

| Field | Value |
|-------|-------|
| Round | 6 |
| Verdict | **REVISE_MAJOR** |
| Path 2 (`wrong40` MSE ≤ 0.15 @ 50%) | **Fail** (0/5 seeds) |
| Path 3 (`ABT-1`) | **Delivered** — limits value, **not** breakthrough |
| Criterion A / B / C | **Fail / N/A / N/A** |
| Accept | **No** (policy + substance) |
| Next round focus | **Unified-protocol JASPER sweep + honest LEAK-CERT vs mean-only naive** |
