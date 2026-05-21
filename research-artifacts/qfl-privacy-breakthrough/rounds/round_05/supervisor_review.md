# Round 05 — Supervisor Review

## Verdict

**REVISE_MAJOR**

Round 05 is an **honest criterion-B execution round**: Snapshot-DP (Gaussian/Laplace noise on batch-mean snapshots before gradient publish) is implemented, grid-searched, and reproducible. **Criterion B fails decisively** — **0/420** grid cells meet normalized utility ≤ **10%**; **0** jointly satisfy utility + ≥**50%** attack MSE increase on any channel. The round correctly parallels Round 02 QFL-SHIELD’s utility wall and satisfies Round 04 Path 1’s **formal kill** condition for Snapshot-DP. Policy forbids **ACCEPT** before round 8 regardless.

---

## Executive summary

Deliverables present: `code/benchmark_round05.py`, `artifacts/round05_metrics.json`, `logs/experiment_round05.log`, `rounds/round_05/researcher_proposal.md`, `rounds/round_05/revision_log.md`. Supervisor re-ran `python3 code/benchmark_round05.py` (~265s); `B_defense_50pct_mse_increase_at_10pct_utility: false` matches submitted JSON.

**Round 05 mission (Round 04 Path 1 — Snapshot-DP / criterion B) — outcome:**

| Check | Result |
|-------|--------|
| Snapshot-DP implemented on batch means | **Yes** |
| Utility ≤ 10% on ≥1 grid cell | **Fail** — **0/420** feasible |
| ≥50% attack MSE increase @ feasible utility | **Fail** — no feasible cell |
| Pre-registered wrong40 + level1 + fraction 0.25 (Stage-2) | **Yes** |
| Formal kill Snapshot-DP (0 feasible cells) | **Triggered** |

Undefended means (supervisor re-run): T1p **0.435**, SHARD Stage-2 wrong40 GARD **1.025**, JASPER-Q **0.953**. Batch-mean noise can **raise** T1p attack MSE (max **+242%** on seed 3, Gaussian σ=0.5) but only at utility loss **≈350%** normalized — far outside the 10% budget. At the lowest-utility grid point (Laplace σ=0.02, any ε), utility loss remains **≈188%** with Stage-2 attack increase **<1%**.

**Snapshot-DP is killed for rounds 6–10** unless the team proposes a **new utility definition** and a **new defense class** with pre-registered thresholds (not another σ/ε sweep).

---

## Breakthrough gates (`config.json`)

| Criterion | Round 05 JSON / proposal | Supervisor ruling |
|-----------|--------------------------|-------------------|
| **A** — ≥25% row reduction @ parent mean gate | Not primary; secondary blocked | **Fail / N/A** |
| **B** — defense vs attack @ ≤10% utility | `B_defense_50pct_mse_increase_at_10pct_utility`: **false** | **Fail** — see utility & attack audit |
| **C** — cert ≥2× tighter than naive | Not attempted | **Fail / N/A** |

**Config wording note:** `config.json` breakthrough text says defense “**reduces** attacker snapshot MSE,” which would mean a **better** attack (lower MSE). Round 05 correctly scores **attack MSE increase** (higher MSE = worse attack). The **code metric direction is right**; `config.json` prose and `honesty_notes` “aligned with config.json” are **wrong** — fix in Round 06 docs.

---

## Verification audit

### 1. Utility metric — **HONEST BUT INFEASIBLE FOR B**

Implementation: `utility_snapshot_dp` (`benchmark_round05.py:314–335`) reuses Round 02 `normalized_utility_loss` contract:

```314:335:research-artifacts/qfl-privacy-breakthrough/code/benchmark_round05.py
def utility_snapshot_dp(
    ...
    for e in range(3):
        batches = make_epoch_batches(true_snapshots.shape[0], 4, seed + e)
        for batch in batches:
            g_true = simulate_gradients(true_snapshots, [batch], surrogate)[1][0]
            g_pub = simulate_gradients_snapshot_dp(
                true_snapshots, [batch], surrogate, mechanism, epsilon, sigma, seed + e, r05
            )[1][0]
            losses.append(float(np.mean((g_true - g_pub) ** 2)))
            base_losses.append(float(np.mean(g_true**2)))
    raw = float(np.mean(losses))
    return raw, raw / max(float(np.mean(base_losses)), 1e-12)
```

**Ruling:**

- Metric is **computed**, not injected; normalization vs undefended gradient energy on the **same** `SurrogateQFL` is consistent with Round 02 SHIELD.
- **3-epoch proxy** on **N=32** QFL snapshots is a **narrow** training-utility stand-in — acceptable if documented; not comparable to full 10-epoch FL training.
- **Minimum** observed `utility_loss_fraction_normalized` ≈ **1.88** (Laplace σ=0.02) — **18.8×** the 10% budget. No cherry-picking: **0/420** cells pass `meets_utility_constraint`.
- When `sigma > 0`, `dp_noise_scale` **ignores** ε (`benchmark_round05.py:127–128`) — ε grid is redundant for most high-σ cells; not dishonest but wasteful.

### 2. Attack implementation — **MIXED FIDELITY**

#### T1p (`run_t1p_attack_mse`, `benchmark_round05.py:338–386`)

**Issue (Major):** Level-1 recovery is **not** `ShardAttacker.level1_mean_recovery` (Round 02 SHIELD used the production attacker stack). Code uses:

```375:375:research-artifacts/qfl-privacy-breakthrough/code/benchmark_round05.py
    e_bar = np.mean([g for grads_e in batch_grads for g in grads_e], axis=0)
```

This is a **flat mean over all published batch gradients**, not SHARD’s iterative mean recovery. T1p MSE numbers are **internally consistent** but **not** comparable to Round 01/02 QFL-SHIELD T1p baselines without relabeling.

**Positive:** Defense noise is applied in `simulate_gradients_snapshot_dp` on the **attacker-visible** publish path; no oracle snapshot mean is passed into `QtermAttack`.

#### SHARD Stage-2 (`run_shard_stage2_attack_mse`, `benchmark_round05.py:389–418`)

**Aligned with Round 04 pre-registration:** `wrong40`, fraction **0.25**, `level1_estimate`, co-occurrence GARD via `run_stage2_with_m_obs` + `select_graph_map`.

**Saturation (Critical for B on this channel):** Undefended mean MSE ≈ **1.025** (already near random-guess scale for wrong-H). A **50% increase** requires defended MSE ≳ **1.54**; observed increases cap ≈ **+45%** only at utility **≫10%**, and **<1%** at lowest noise. **Criterion B on SHARD Stage-2 is structurally unreachable** under this stressed assignment without changing the attack baseline or metric.

#### JASPER-Q (`run_jasper_attack_mse`, `benchmark_round05.py:421–450`)

Uses Round 04 **T1p warm-start** (65% blend) inside `run_jasper_q_with_m_obs` — same tier confound Round 04 flagged. Defense only perturbs `m_obs`; warm-start still injects honest partial terminal structure. JASPER attack MSE **barely moves** (<3% increase) across the grid.

#### Noise mechanism (`batch_mean_noise_cache`, `benchmark_round05.py:145–169`)

- Noise is keyed by **sorted batch membership** — identical noise for repeated compositions; **not** per-publish independent DP slices.
- **No** \((\varepsilon,\delta)\) ledger, clipping, or composition accounting — Round 04 demanded explicit bookkeeping; Round 05 is **heuristic noise**, correctly disclaimed in `honesty_notes` but **not** Snapshot-DP in the DP-SGD sense (same class of caveat as Round 03 `dp_mean`).

### 3. Criterion B grid — **CONFIRMED FAIL**

From `snapshot_dp_grid.summary` (supervisor re-run):

| Aggregate | Value |
|-----------|------:|
| Grid cells | 420 |
| Feasible utility (≤10%) | **0** |
| B-pass cells (utility + 50% increase, any channel) | **0** |
| `breakthrough_B_50pct_mse_increase_at_10pct_utility` | **false** |

**Strongest privacy signal (fails utility):** seed 3, Gaussian σ=0.5 — T1p attack increase **+242%**, utility **+350%**.

**Closest to utility budget (still fails):** Laplace σ=0.02 — utility **+188%**, T1p increase **≈0.6%**, Stage-2 increase **≈0%**.

Proposal headline “seed 19, Gaussian ε=8, σ=0.5 → **+268%**” is **not supported** by JSON (seed 19 at σ=0.5 ≈ **+145%**). Max T1p increase is seed 3 (+242%). **Minor reporting slip** — does not change fail verdict.

### 4. Round 04 mandatory carry-over — **PARTIAL**

| Round 04 demand | Round 05 status |
|-----------------|-----------------|
| Path 1 Snapshot-DP primary | **Done** |
| Pre-registered B on wrong40 @ 0.25 + level1 | **Done** (Stage-2 channel) |
| Formal kill if 0 feasible cells | **Done** (below) |
| \((\varepsilon,\delta)\) bookkeeping in code | **Fail** |
| `researcher_proposal.md` before runs | **Present** (timing not auditable post hoc) |
| Extend `config_breakthrough` flags | **Partial** — only B + secondary booleans |
| ASSIGN-LOCK kill referenced | **Done** in revision log |

---

## Critical issues (Round 05 → carry to Round 06)

1. **Criterion B not met** — **0** feasible utility cells; **0** B-pass cells. Snapshot-DP on batch means **cannot** satisfy the project utility gate at any tested \((\varepsilon,\sigma)\) in this simulator.
2. **Stage-2 B metric saturated** — wrong40 + level1 GARD undefended MSE ≈ **1.0** makes “+50% increase” the wrong pre-registered target for SHARD; even large noise barely moves MSE while destroying utility.
3. **T1p attack stack inconsistent with Round 02** — flat gradient mean vs `level1_mean_recovery` invalidates cross-round T1p comparisons.

## Major issues

4. **Not formal DP** — no sensitivity/clipping/composition proof; ε grid cosmetic when σ dominates.
5. **JASPER-Q defense eval confounds T1p warm-start** — oracle terminal channel remains in attacker stack.
6. **`config.json` criterion B prose inverted** vs implemented increase metric; fix documentation before any external claim.
7. **Proposal numeric slip** — +268% T1p claim vs JSON ~+145% (seed 19).

## Minor issues

8. Grid size 420 cells × ~265s — acceptable; do **not** expand σ grid under kill policy.
9. `README.md` supervisor column needs round 5 entry after this review.

---

## Snapshot-DP formal kill (rounds 6–10)

Per Round 04 Path 1 item 4: with **0** feasible utility cells, **Snapshot-DP** (batch-mean Gaussian/Laplace grid on this surrogate) is **removed** from the primary plan for rounds **6–10**.

**Forbidden without a new proposal path:** further σ/ε grid sweeps, renaming Round 03 `dp_mean` anchor as “Snapshot-DP,” or claiming T1p MSE bumps at utility **>100%** as partial B success.

**Allowed:** a **new** defense class (e.g. secure aggregation, gradient clipping + calibrated DP-SGD with **task-loss** utility) with fresh pre-registration — not Snapshot-DP tuning.

---

## Config / policy compliance

| Policy | Round 05 |
|--------|----------|
| `forbid_accept_before_round: 8` | **ACCEPT forbidden** (round 5 of 10) |
| `require_measurable_breakthrough` | **Not met** |
| `forbid_accept_on_negative_result_only` | Round 05 is a valid negative on **B**; insufficient alone |
| `strict_no_early_accept` | **Honored** |

**NO ACCEPT before round 8** — reaffirmed.

---

## Round 06 demands (mandatory — no criterion B pass)

Round 06 must pursue **exactly one** primary path from Round 04 (Snapshot-DP is **killed**). Pre-register in `rounds/round_06/researcher_proposal.md` **before** runs.

### Path 2 — Relaxed row-reduction milestone (recommended)

1. Extend fraction grid to **0.50** (30/60 rows). Stack: **GARD co-occurrence** and/or **JASPER-Q with T1p warm-start disabled on `oracle` assignment** (Round 04 demand).
2. **Pre-registered success:** `wrong40` + `level1_estimate`, fraction **0.50**, mean snapshot MSE **≤ 0.15** on **≥4/5** seeds.
3. Export `A_wrong40_mse_0.15_at_50pct_rows` in `config_breakthrough`.
4. Report **used-vs-true** observed-batch residual MSE (parent Round 06 audit) — mandatory this round.

### Path 3 — Assignment barrier theorem (alternate)

5. `literature/assignment_barrier.md` or `paper/assignment_barrier.md`: lemma + proof sketch tied to `wrong40` corruption and level1 anchor; falsifiable prediction MSE floor **≥ 0.15** at 25% rows.
6. Benchmark cites theorem ID; no new mechanism required if proof complete.

### Cross-cutting (Round 06)

7. **Fix T1p attacker** if any lane still reports T1p: use `ShardAttacker.level1_mean_recovery` for parity with Round 02, or document intentional downgrade.
8. Clarify `config.json` criterion **B** text to “attack MSE increase ≥50%” (or “defender reduces attack success by ≥50%”) — align prose with code.
9. `code/benchmark_round06.py` → `artifacts/round06_metrics.json`, `logs/experiment_round06.log`, `revision_log.md`.
10. Extend `config_breakthrough` with `wrong20`, `unknown_random` booleans if those regimes are run.

**Round 06 may not receive ACCEPT** even if Path 2 intermediate milestone passes.

---

## Rubric scores (1–5)

| Dimension | Score | Note |
|-----------|------:|------|
| Novelty | 2 | Batch-mean noise is standard; first systematic grid in this run |
| Soundness | 3 | Code runs; attack/utility contracts partially inconsistent |
| Feasibility | 4 | Full grid reproducible ~265s |
| Impact | 2 | Negative result closes B lane — valuable, not breakthrough |
| Evaluability | 4 | JSON grid thorough; config/prose mismatch hurts |

---

## Scientist alignment

Proposal and revision log **match** supervisor metrics on B fail and kill rationale. Minor correction needed on +268% headline. Team correctly does **not** claim breakthrough. Supervisor **rejects** any Round 06 plan that spends budget tuning Snapshot-DP σ/ε.

---

## Supervisor sign-off

| Field | Value |
|-------|-------|
| Round | 5 |
| Verdict | **REVISE_MAJOR** |
| Primary path | Snapshot-DP (criterion B) |
| Criterion B | **Fail** (0/420 feasible) |
| Snapshot-DP status | **Killed** (rounds 6–10) |
| Criterion A / C | **Fail / N/A** |
| Accept | **No** (policy round ≥ 8 + substance) |
| Next round focus | **`wrong40` MSE ≤ 0.15 @ 50% rows** OR **assignment barrier theorem** |
