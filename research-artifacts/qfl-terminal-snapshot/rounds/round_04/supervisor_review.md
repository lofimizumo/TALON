# Round 04 — Supervisor Review (Final Acceptance Gate)

## Verdict

**ACCEPT_WITH_MINOR**

Round 04 closes the acceptance gate for this research run: **LASA-QTERM** (alias **Q-SNAP-T**) packages Round 03 science into a reproducible, tier-honest artifact (production API, consolidated benchmark, method/scope papers, tutorial, dual-track metrics). All user acceptance criteria for a **scoped** outcome are met. The primary `config.json` goal—individual snapshot recovery at **strict T1** (epoch terminals only, 0 within-epoch intermediates) within 2× SHARD—is **correctly not met** and is **replaced** by Proposition T1 plus labeled positive tiers (T1p, T1b@80).

**Run acceptance:** Recommend **stop** under `stop_mode: until_acceptance` for the scoped QFL terminal-snapshot objective. Optional future rounds (T1p@80 honest, real QFL stack) are **non-blocking**.

---

## Executive summary

| Scoped tier | Round 04 outcome (5 seeds) | Acceptance |
|-------------|---------------------------|------------|
| **T1** (10 epoch-mean terminals, 0 intermediates) | `lasa_qterm_T1` MSE **0.987** (smooth) / **1.000** (MNIST); vs SHARD **0.483** / **0.605** → ratio **2.04×** / **1.65×**; `meets_2x_at_tier_budget: false` | **Impossibility accepted** (`paper/impossibility_t1.md`) |
| **T1p honest** (last \(p{=}7\) rows/epoch, 70 terminal rows) | MSE **0.446** / **0.635**; `imputation_free: true`; no `pad_partial` in `QtermAttack` | **Positive bracket** (non-primary; partial trajectory leak) |
| **T1b @ 80 rows** (B=1 per-client, matched budget) | MSE **0.160** / **0.111** vs SHARD@80 **0.677** / **0.574** (ratio **0.24×** / **0.19×**) | **Individual recovery** without within-epoch steps; **non-primary** channel |
| **T2** (SHARD Stage 2, 80 intermediate rows) | Oracle + budget-80 subsample present | **Required baseline** — pass |

---

## Acceptance criteria audit

| Criterion | Result | Evidence |
|-----------|--------|----------|
| T1 impossibility rigorous | **Pass** | `paper/impossibility_t1.md` — reduction \(\mathcal{A}(S)=\mathcal{B}(\bar{s})\), explicit \(\Delta^\top\mathbf{1}=0\) non-uniqueness; graph priors do not add linear constraints |
| Positive tiers honest, reproducible, no imputation cheat | **Pass** | `QtermAttack._recover_t1p` → `partial_honest_disaggregate` only; `imputation_free: true` in metrics; `pad_partial_for_shard` excluded from production (`paper/method.md`) |
| SHARD baseline present | **Pass** | `ShardAttacker.level2_disaggregate` in `benchmark_round04.py`; T2 + T2@80 in `acceptance_table` |
| Individual snapshots on defensible weak tier | **Pass** | **T1p** (70 terminal rows, 0 intermediates) and **T1b@80** (80 terminal rows, 0 intermediates) recover spread rows (`snapshot_row_std` ≫ passive); Hungarian MSE ≤2× SHARD at tier-appropriate budgets |
| Scoped claims documented | **Pass** | `paper/scope.md`, `paper/method.md`, `tutorial/tutorial.md`, `acceptance_table` in `artifacts/round04_metrics.json` |
| Runnable experiments | **Pass** | `logs/experiment_round04.log`; supervisor re-ran `python3 code/benchmark_round04.py --smooth-only` successfully |
| No within-epoch intermediates (T1/T1p/T1b labels) | **Pass** | `observed_intermediate_batch_gradients_mean: 0` for LASA-QTERM methods |
| QFL surrogate required | **Pass** | `SurrogateQFL` throughout |
| `max_critical_issues` = 0 | **Pass** | No metric injection, imputation headline, or B=1@320 conflation in acceptance paths |

---

## Round 03 → Round 04 delta

| Round 03 gap / minor | Round 04 status |
|---------------------|-----------------|
| No unified method name | **Fixed** — LASA-QTERM / Q-SNAP-T in code, JSON, papers |
| Fragmented benchmarks | **Fixed** — `code/benchmark_round04.py` + `acceptance_table` |
| No production wrapper | **Fixed** — `code/qterm_attack.py` (`QtermAttack`, `QtermTier`) |
| Tutorial stub | **Fixed** — `tutorial/tutorial.md` |
| MNIST optional | **Fixed** — default dual track in R04 |
| Misleading `meets_2x` on T1 | **Fixed** — `meets_2x_at_tier_budget: false` for T1 rows; `primary_t1_path: impossibility` |
| `config.json` tier gates | **Partially fixed** — `supervisor_status` + `acceptance_tiers` added (this round) |
| Honest partial @ 80 rows (B>1) | **Open** (minor, non-blocking) |
| SHARD `matching_acc` vs Hungarian MSE | **Open** (minor; documented) |
| Real QFL beyond SurrogateQFL | **Open** (out of scope) |

---

## T1 impossibility (proof audit)

**Assessment:** **Sound** and **aligned** with simulator. Epoch terminals are `np.mean(grads_e, axis=0)` per epoch; T1 uses 10 rows, 0 intermediates. Empirical T1 MSE tracks passive broadcast (~1.0 MNIST; ~0.99 smooth with small graph spread, `snapshot_row_std` ~0.24 vs ~10⁻¹⁵ passive).

**Headline discipline:** Do **not** treat MNIST T1 ratio **1.65×** as recovery—`snapshot_row_std` ~10⁻⁴ shows collapsed rows; ratio reflects SHARD oracle MSE > passive floor.

---

## Tier evaluations (Round 04 metrics)

### T1 — strict epoch terminal

| Track | LASA-QTERM T1 MSE | SHARD T2 MSE | Rows (T1 / SHARD) | `meets_2x_at_tier_budget` |
|-------|------------------:|-------------:|-------------------|---------------------------|
| smooth | 0.987 | 0.483 | 10 / 80 intermediate | false |
| mnist | 1.000 | 0.605 | 10 / 80 intermediate | false |

**Ruling:** Accepted as **negative identifiability result**, not failed attack.

### T1p — honest partial (defensible weak tier)

| Track | MSE | Rows | vs SHARD full | Notes |
|-------|----:|-----:|--------------:|-------|
| smooth | 0.446 | 70 | 0.92× | Approaches SHARD; 7/8 batch rows/epoch |
| mnist | 0.635 | 70 | 1.05× | Borderline vs full oracle; still honest partial |

**Ruling:** Valid **individual recovery** path under **weaker-than-T2, stronger-than-T1** observation (terminal-row leak without within-epoch *published* intermediates). Tier-labeled non-primary.

### T1b @ 80 — matched row budget (defensible weak tier)

| Track | LASA-QTERM T1b@80 | SHARD@80 | Ratio |
|-------|------------------:|---------:|------:|
| smooth | 0.160 | 0.677 | 0.24× |
| mnist | 0.111 | 0.574 | 0.19× |

**Ruling:** Strong **individual recovery** at **equal 80-row budget** with **0** within-epoch intermediates. Channel is per-client B=1 (not epoch aggregate)—correctly **non-primary** for the strict user goal in `config.json`.

### T2 — SHARD oracle

Present on both tracks; `matching_acc_mean = 0` with good Hungarian MSE—same as Round 03; use Hungarian MSE as primary (documented in `paper/scope.md`).

---

## Production code audit

| Check | Result |
|-------|--------|
| `QtermAttack` tier routing | T1 / T1p / T1b separated; required inputs enforced |
| T1p imputation | **None** — `partial_epoch_gradients` + `partial_honest_disaggregate` |
| T1 production pick | `graph_term_map` (Round 03 winner); meta cites impossibility |
| Budget accounting | `observed_terminal_gradient_rows` / `observed_intermediate_batch_gradients` logged per method |
| Padded partial | Not in `QtermAttack` (benchmark_round03 only) |

---

## Honesty / reproducibility

| Check | Result |
|-------|--------|
| Metrics from arrays | **Pass** — Hungarian / fixed-order MSE in benchmark |
| Acceptance table tier labels | **Pass** — explicit notes per row |
| Researcher proposal | **Pass** — states T1 not met; impossibility path |
| MNIST `level1_mean_rel_error` huge | **Info** — surrogate encode vs simulated grads; does not affect tier rankings |
| SHARD convergence warnings | **Info** — seed 19 smooth; MSE still reported |

**Cheating verdict:** **No fraud.** Round 02–03 integrity fixes hold in Round 04 packaging.

---

## Rubric (1–5)

| Criterion | Score | Note |
|-----------|------:|------|
| Novelty | 4 | Impossibility + tier taxonomy + packaged LASA-QTERM |
| Soundness | 4 | Proof + budget-honest tables; T1p heuristic assignment |
| Feasibility | 5 | Single benchmark, dual tracks, production API |
| Impact | 4 | Closes false epoch-only hope; charts honest escalation paths |
| Evaluability | 5 | `acceptance_table`, JSON verdict block, tutorial |

---

## Issues

| Tag | Issue |
|-----|-------|
| **Minor** | Honest **partial @ 80 rows** with B=4 (no B=1, no imputation) still not run |
| **Minor** | SHARD `matching_acc = 0%` vs good Hungarian MSE — threshold audit deferred |
| **Minor** | SurrogateQFL only; no deployed QFL/VQC stack |
| **Minor** | `rounds_planned: 8` vs acceptance now — recommend updating stop policy or archiving optional rounds |
| **Info** | T1b full (320 rows) excluded from primary claims (`non_acceptance_for_primary_t1` in meta) |

**Critical issues:** **0**

---

## Config & user-goal compliance

| `config.json` gate | Round 04 |
|--------------------|----------|
| `require_runnable_experiments` | **Yes** |
| `require_shard_stage2_baseline` | **Yes** |
| `require_no_intermediate_gradients` (within-epoch, for weak tiers) | **Yes** (T1/T1p/T1b counts) |
| `require_individual_snapshot_recovery` | **Yes** at T1p / T1b@80 (tier-scoped); **No** at T1 (proved) |
| `snapshot_mse_vs_shard_stage2_oracle` @ T1 budget | **No** (expected) |
| Same @ T1b@80 / comparable | **Yes** (exceeds ≤2×) |
| `forbid_aggregate_only_endpoint` | **Not violated** — tiered attacks + impossibility |
| `forbid_claim_without_qfl_surrogate` | **Yes** |
| Primary goal (strict T1 ≈ SHARD) | **Not met** — documented impossibility |

---

## Scoped acceptance decision

The run **ACCEPTs** Round 04 and the **LASA-QTERM** package under this scope:

1. **T1 impossible (proved)** — primary scientific outcome for epoch-terminal-only QFL under LASA linearity.
2. **T1p honest** — imputation-free partial; 70-row budget disclosed; non-primary.
3. **T1b @ 80 terminal rows** — individual recovery at matched budget; B=1 channel disclosed as non-primary.
4. **SHARD T2 oracle** — required upper bound.
5. **No imputation or cross-tier headline cheats** in production or acceptance table.

**Not accepted:** Global claim that strict T1 (10 epoch means) recovers individuals within 2× SHARD.

---

## Verdict summary

| Item | Status |
|------|--------|
| **Supervisor verdict** | **ACCEPT_WITH_MINOR** |
| **Method package** | **LASA-QTERM** (`code/qterm_attack.py`) |
| **T1** | **Impossibility accepted** |
| **T1p / T1b@80** | **Accepted positive brackets** (tier-labeled) |
| **Imputation / headline cheat** | **Cleared** |
| **Primary `config.json` goal @ T1 only** | **Not met** (documented) |
| **Recommend stop run** | **Yes** (scoped acceptance complete) |
