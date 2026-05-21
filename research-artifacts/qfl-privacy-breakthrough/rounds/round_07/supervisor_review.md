# Round 07 — Supervisor Review

## Verdict

**REVISE_MAJOR**

Round 07 delivers **JASPER-Q v7** (multi-epoch T1p warm-start, spectral kNN graph, fraction sweep) and **LEAK-CERT-T1p v2** with reproducible artifacts. **JASPER-Q still fails criterion A** and all Round 07 pre-registered relaxed assignment gates (best `wrong40` mean MSE **≈ 0.888** @ 75% rows — **≫ 0.10–0.15**). **Criterion C passes only against a trace-inflated naive baseline** (`naive_trace/cert` **≈ 14.8×**); it **fails** against `config.json`’s **mean-only naive** (**≈ 0.93×**, not ≥2). Supervisor rules criterion **C** as **inflated naive gaming**, **not** an honest breakthrough. **ACCEPT is forbidden** before round 8; **ACCEPT remains forbidden** in rounds 8–10 unless **C** is re-earned on the **contract baseline** **and** at least one of **A** or **B** passes.

---

## Executive summary

Deliverables: `code/benchmark_round07.py`, `artifacts/round07_metrics.json`, `logs/experiment_round07.log`, `rounds/round_07/researcher_proposal.md`, `rounds/round_07/revision_log.md`. Supervisor re-ran `python3 code/benchmark_round07.py` (~48s); gate flags match JSON: `A_std=False`, relaxed A **false**, `C_cert_2x_tighter_than_naive_trace=true` (trace baseline only).

**Round 07 mission — outcome:**

| Lane | Result |
|------|--------|
| JASPER-Q v7 fraction sweep `wrong40` / `oracle` | **Diagnostic only** — monotonic MSE drop with more rows; **no** gate pass |
| Relaxed A (MSE ≤ **0.15**, ≥50% row reduction) | **Fail** |
| Relaxed `wrong40` (MSE ≤ **0.10**, ≥40% reduction) | **Fail** |
| Standard **A** @ **0.10**, ≥25% reduction | **Fail** |
| LEAK-CERT v2 criterion **C** (trace naive) | **Passes 5/5** — **rejected** as breakthrough (wrong baseline) |
| LEAK-CERT v2 vs **mean-only naive** | **Fail** (**≈ 0.93×** < 2×) |
| Assignment barrier broken | **false** — consistent with `ABT-1` |

---

## Breakthrough gates (`config.json`)

| Criterion | Round 07 JSON | Supervisor ruling |
|-----------|---------------|-------------------|
| **A** (standard) | `A_row_reduction_25pct_wrong40_parent_gate_0.10`: **false** | **Fail** — best wrong40 **≈ 0.888** @ 60 rows; **0/5** seeds @ any target |
| **A** (relaxed pre-reg) | `A_relaxed_mse_0.15_reduction_50pct`: **false** | **Fail** |
| **A** (`wrong40` @ 0.10, 40% rows) | `wrong40_mse_0.10_reduction_40pct`: **false** | **Fail** |
| **B** | Not attempted (R05 kill) | **Fail / N/A** |
| **C** — cert ≥2× tighter than **naive mean-only** | `C_cert_2x_tighter_than_naive_trace`: **true** only | **Fail (breakthrough)** — see §4 |

**Criterion C breakthrough ruling:** **NOT ACCEPTABLE** on trace-inflated naive alone. Round 04 supervisor already killed tuning LEAK-CERT constants without a new proof object; Round 07 **widens** the naive baseline (`cert_naive_trace_inflated`) to manufacture **14.8×** while the **contract** baseline fails. This is **inflated naive gaming**, not a **≥2× tighter proved bound** in the sense of `config.json`.

---

## Verification audit

### 1. JASPER-Q v7 @ `wrong40`, `level1_estimate` — **DIAGNOSTIC, NOT BREAKTHROUGH**

Aggregate `jasper_q_v7_multi_epoch_spectral` (supervisor re-run):

| Fraction | Observed rows | Mean MSE | vs Round 06 JASPER @ 0.50 |
|----------|--------------:|---------:|---------------------------:|
| 0.25 | 20 | **0.955** | **≈ 0.953** (theorem ref) |
| 0.50 | 40 | **0.907** | **≈ 0.896** |
| 0.75 | 60 | **0.888** | (new — best) |

**Ruling:** More rows **help** (~7% relative 0.25→0.75) but MSE remains **~6–9×** above **0.15** and **~9×** above **0.10**. `parent_mean_mse_gate.reaches_target: false` for **all** targets **{0.05, 0.10, 0.15}** on both `wrong40` and `oracle`. **JASPER-Q v7 does not break the assignment barrier**; it **refines** the floor estimated by `ABT-1`.

**Tier confound (Major):** `wrong40` still uses `t1p_warm_blend=0.65` and multi-epoch T1p recovery on the **honest terminal channel**. GARD co-occurrence on wrong-\(H\) remains worse (**≈ 1.11** @ 0.75), but that is a **wrong-graph** baseline — not evidence of Stage-2 assignment recovery at production MSE.

**Oracle regression vs GARD (carry Round 04):** @ 0.75, GARD **≈ 0.002**, JASPER **≈ 0.682** — spectral + MAP stack still **far** from co-occurrence GARD under correct assignment.

### 2. Incidence grid — **IMPROVED VS ROUND 06, STILL VERIFY**

Round 07 uses `stage_cfg_from_qfl` for JASPER sweep (`n_samples=32`, `n_epochs=10` → **80** full rows). GARD wrong-\(H\) in the same benchmark shares that construction for v7 runs — **better than Round 06’s 48/60 vs 32/80 split**. Document this explicitly in Round 08+; any legacy GARD numbers from Round 06 must **not** be compared directly.

### 3. `jasper_q_v7_hard_overlap` in aggregates — **MISLABELED METRIC**

`hard_overlap` returns **batch membership overlap fraction** (~0.41–0.48), not snapshot MSE, but appears under `aggregate[].snapshot_mse_mean`. **Do not** cite **~0.45** as MSE improvement. Reporting bug — fix in Round 08 JSON schema.

### 4. LEAK-CERT-T1p v2 / criterion **C** — **TRACE-INFLATED NAIVE GAMING**

`leak_cert_t1p_v2.aggregate` (supervisor re-run):

| Metric | Mean | Notes |
|--------|-----:|-------|
| Empirical T1p MSE | **0.424** | Honest attack path |
| Cert tight upper (`cert_tight`) | **1.080** | Nearly **constant** across seeds |
| Naive **mean broadcast** | **1.000** | Contract baseline |
| Naive **trace-inflated** | **16.000** | `rank_term = trace_var × n / rank × 2` |
| `naive_broadcast / cert` | **≈ 0.93** | **Fails** 2× tightening |
| `naive_trace / cert` | **≈ 14.81** | **Passes** 2× — **non-contract** |
| `cert_covers_empirical` @ 0.95 | **5/5** | Coverage with slack |

**Mechanism (code audit):**

```311:322:research-artifacts/qfl-privacy-breakthrough/code/benchmark_round07.py
def cert_naive_trace_inflated(
    e_bar: np.ndarray,
    snapshots: np.ndarray,
    true_rank: int,
) -> float:
    broadcast = cert_naive_mean_only(e_bar, snapshots)
    pop_var = float(np.var(snapshots))
    n = snapshots.shape[0]
    trace_var = snapshot_trace_variance(snapshots)
    rank_term = trace_var * n / max(true_rank, 1)
    return float(max(broadcast, pop_var * 2.0, rank_term * 2.0))
```

With `n=32`, `true_rank=4`, `trace_var≈1`, `rank_term` **dominates** at **~16**, while cert **~1.08** (`cert_trace_slack` + floors). The **14.8×** ratio is **mostly baseline inflation**, not a tighter certificate than Round 04’s **~0.53** cert (**~1.89×** vs mean-only).

**Additional issues:**

- `cert_t1p_trace_upper` includes `max(cert, pop_var * 0.62, …)` — **calibrated** to cover empirical (~0.42) with headroom, not derived tight bound.
- `criterion_c_2x_tighter_than_naive_broadcast`: **false** on **all** seeds in JSON — team’s own flag contradicts `config.json` **C** if read literally.
- Round 07 proposal pre-registration of trace naive for **C** is **post-hoc gaming** relative to parent `breakthrough_definition`.

**Supervisor ruling:** **Criterion C is FAIL for breakthrough purposes.** Trace-inflated naive may be recorded as **diagnostic** `C_trace_inflated_pass` but **must not** trigger `require_measurable_breakthrough` or early accept. **No ACCEPT before round 8 unless C is honest** — condition **not met**.

### 5. Round 06 carry-over — **PARTIAL**

| Demand | Round 07 status |
|--------|-----------------|
| Unified-protocol sweep | **Partial** — v7 internally consistent; cross-round compare needs care |
| Fraction extension | **Done** 0.25–0.75 |
| Oracle warm-start off | **Done** (`blend=0`) |
| Honest **C** vs mean-only | **Fail** — trace gaming |
| `supervisor_review.md` | **Missing at run time** — this document |
| `README.md` status table | **Stale** |

---

## Critical issues (Round 07 → carry to Rounds 8–10)

1. **No criterion A breakthrough** — best wrong40 **≈ 0.888**; **0/5** seeds @ **0.15** or **0.10** with required row reduction.
2. **Criterion C is not honest** — **14.8×** is an artifact of `cert_naive_trace_inflated`; **mean-only** ratio **< 1×** vs cert.
3. **LEAK-CERT constant cert across seeds** — suggests slack/floor tuning, not seed-adaptive bound (Round 04 failure mode persists).
4. **JASPER-Q still terminal-confounded on `wrong40`** — cannot satisfy `early_accept_requires_two_independent_breakthrough_signals` with **C** alone even if fixed.

## Major issues

5. **Aggregate mislabels `hard_overlap` as `snapshot_mse_mean`** — correct before any external table.
6. **Relaxed gates failed** — proposal post-run admits; do not promote v7 as milestone pass.
7. **GARD @ 0.35 wrong40** mean **≈ 17.7** (outlier seed 11) — report stability / clipping in defense planning if GARD remains a baseline.

## Minor issues

8. Runtime **~48s** — acceptable.
9. `qfl_terminal_snapshot_t1p_mse_reference` **≈ 0.476** — empirical T1p **beats** reference; cite but do not over-claim privacy safety.

---

## Criterion C — formal status (rounds 8–10)

**LEAK-CERT-T1p v2 with trace-inflated naive** is **removed** as a **primary acceptance lane** until:

1. **Primary 2× ratio** uses `cert_naive_mean_only` only (per `config.json`).
2. Cert is **seed-adaptive** and **not** dominated by a single global `cert_trace_slack` multiplier.
3. **Proof sketch** links Ky-Fan tail to identifiable row budget — not `max(·)*1.08` calibration.
4. **Coverage** on all seeds without `pop_var * 0.62` floors.

**Forbidden:** claiming Round 07 **C pass** in executive summaries, README, or accept memos.

---

## Config / policy compliance

| Policy | Round 07 |
|--------|----------|
| `forbid_accept_before_round: 8` | **ACCEPT forbidden** (round 7 of 10) |
| `require_measurable_breakthrough` | **Not met** (honest **C** fails) |
| `early_accept_requires_two_independent_breakthrough_signals` | **Not met** |
| `forbid_accept_on_negative_result_only` | v7 diagnostic + fake **C** insufficient |
| **User policy:** no ACCEPT unless **C** honest | **Violated if ACCEPT considered** — **reject** |

---

## Rounds 8–10 demands (mandatory)

Rounds **8**, **9**, and **10** are the **final window**. Each round needs `researcher_proposal.md` **before** runs and a supervisor review **after**. No **ACCEPT** until round **8** minimum; rounds **9–10** are for **confirmation**, not new primary pivots unless Round 8 fails both honest **C** and **A**.

### Round 8 — **Honest certificate OR defense (pick one primary)**

1. **Path C-honest (recommended if cert continues):** Re-implement LEAK-CERT as **v3** with:
   - **Mandatory** `naive_broadcast/cert ≥ 2` on **all** seeds (the only **C** metric in `config.json`).
   - **Forbidden** as primary gate: `cert_naive_trace_inflated`, `pop_var * 0.62` floors, global slack tuned to pass 5/5.
   - Report trace-inflated ratio as **secondary** `C_trace_diagnostic` only.
   - If **0/5** seeds pass mean-only 2×: **formal kill** LEAK-CERT for rounds **9–10** (one paragraph in `revision_log.md`).
2. **Path A-final (parallel or alternate primary):** One **unified-protocol** attacker stack on `wrong40` + `level1` with **T1p warm-start ablation grid** `{0, 0.35, 0.65}` — pre-register: **≥4/5** seeds with MSE **≤ 0.15** at **≥50%** row reduction **without** warm-start **OR** with warm-start **and** MSE **≤ 0.10** at **≥25%** reduction (pick one hypothesis; no dual claiming).
3. Export `config_breakthrough.C_cert_2x_tighter_than_naive_mean_only` — distinct from trace flag.
4. Fix `hard_overlap` naming in JSON aggregates.

### Round 9 — **Replication + independence**

5. **Replicate** Round 8 primary pass on **held-out seeds** `{5, 13, 17, 29, 31}` (5 seeds) without hyperparameter changes — pre-register pass rule: **≥4/5** same gate as Round 8.
6. If Round 8 was **C-honest**, Round 9 must show **B** or **A** progress **or** document cross-tier impossibility tying **ABT-1** to T1p cert (limits paper path).
7. If Round 8 was **A**, Round 9 must include **defense or cert** second signal for `early_accept_requires_two_independent_breakthrough_signals`.

### Round 10 — **Integration, kill list, accept memo**

8. **Written kill list** in `rounds/round_10/revision_log.md`: Snapshot-DP (R05), ASSIGN-LOCK (R04), trace-inflated **C** (R07 unless R8 rescues mean-only **C**), Path 2 @ 0.15 (R06–07).
9. `README.md` supervisor table complete rounds **1–10**.
10. **Accept memo** (`rounds/round_10/acceptance_decision.md`) allowed **only if**:
    - round index **≥ 8**;
    - **honest** criterion **C** (**mean-only** 2×, all seeds covered) **OR** criterion **A** or **B** per `config.json`;
    - **two independent signals** if before round 10;
    - supervisor **0 critical** issues on final replication.
11. **Forbidden in Round 10:** ACCEPT on trace-inflated **C** alone; ACCEPT on JASPER diagnostic without gate pass; ACCEPT on `ABT-1` sketch alone.

### Cross-cutting (Rounds 8–10)

12. **No** Snapshot-DP σ/ε grids (R05 kill).
13. **No** ASSIGN-LOCK variants (R04 kill).
14. **No** new naive baselines for **C** without supervisor pre-approval in proposal.
15. Cite `rounds/round_06/supervisor_review.md` (protocol fairness) and this review (**C** gaming) in all claims.

---

## Rubric scores (1–5)

| Dimension | Score | Note |
|-----------|------:|------|
| Novelty | 3 | v7 structural extensions; cert v2 variant |
| Soundness | 2 | **C** fails contract baseline; overlap mislabel |
| Feasibility | 4 | Full sweep ~48s reproducible |
| Impact | 2 | Tighter wrong40 floor; kills fake **C** narrative |
| Evaluability | 3 | Rich JSON undermined by baseline switch |

---

## Scientist alignment

Proposal **correctly** reports relaxed gate failures and **does not** claim **A**. **Misalignment:** post-run **C pass** on trace naive is framed as success — supervisor **rejects** that interpretation. Team must align with `config.json` mean-only wording before Round 8.

---

## Supervisor sign-off

| Field | Value |
|-------|-------|
| Round | 7 |
| Verdict | **REVISE_MAJOR** |
| JASPER-Q v7 | **Diagnostic** — best wrong40 MSE **≈ 0.888** @ 75% rows |
| Criterion A (all gates) | **Fail** |
| Criterion C (honest / mean-only) | **Fail** |
| Criterion C (trace-inflated) | **Pass metric only — not breakthrough** |
| Accept | **No** (policy + dishonest **C**) |
| Rounds 8–10 | **Honest C or A + replication + kill list** |
