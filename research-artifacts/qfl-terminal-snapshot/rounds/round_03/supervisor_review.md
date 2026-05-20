# Round 03 — Supervisor Review

## Verdict

**ACCEPT_WITH_MINOR**

Round 03 satisfies the Round 02 gate for **scoped acceptance**: formal **T1 impossibility** is proved and aligned with the simulator; **T1p honest** and **T1b @ 80 rows** are budget-labeled and separated from padded/imputed paths; **no headline cheat** from B=1 row inflation or mean-imputed partials. The primary `config.json` goal (individual snapshots at **strict T1** within 2× SHARD) is **correctly not claimed**—it is replaced by a negative identifiability result plus positive, tier-explicit brackets.

**Do not** read `meets_2x_shard_at_t1_budget: true` on the MNIST track as recovery success: T1 MSE ≈ 1.0 is passive-scale noise; smooth T1 remains **2.04×** worse than SHARD and matches the proof.

---

## Executive summary

| Scoped tier | Round 03 outcome | Acceptance |
|-------------|------------------|------------|
| **T1** (10 epoch-mean terminals, 0 within-epoch intermediates) | Proof in `paper/impossibility_t1.md`; best attack GRAPH-TERM MSE **0.987** (smooth) / **≈1.0** (MNIST) vs SHARD **0.483** / **0.605** | **Accepted as impossibility** (not ≤2× recovery) |
| **T1p honest** (last \(p\) minibatch rows/epoch, no imputation) | Best `partial_honest_last_7`: MSE **0.441** / **0.586**, **70** terminal rows, `imputation_free: true` | **Positive bracket**; not T1; row budget > T1 |
| **T1b** (B=1 per-client terminals @ **80** rows) | `b1_client_budget80_terminal` MSE **0.160** / **0.111** vs `shard_budget80` **0.677** / **0.574** (ratio **0.24×** / **0.19×**) | **Positive at matched rows**; **non-acceptance** for primary weak-terminal goal (stronger FL channel) |
| **T2** (SHARD Stage-2, 80 intermediate rows) | Oracle + budget subsample; matching_acc **0%** strict, Hungarian MSE primary | Baseline **required** — present |
| **T1p-pad** (imputed early rows) | Reported separately; excluded from honest headline | Upper bound only — **not** a cheat if not used in claims |

---

## Round 02 issue resolution

| Round 02 (REVISE_MAJOR) issue | Round 03 status |
|------------------------------|-----------------|
| Headline B=1 conflation | **Fixed** — `headline_by_tier` only; no global `best_terminal_method` |
| `pad_partial_for_shard` dishonesty | **Fixed** — `partial_honest_*` in headline tier; padded methods labeled `imputation_free: false` with `imputed_early_rows_per_epoch` |
| Comparable budget ignored | **Fixed** — `budget80_matched` (80 B=1 terminal vs 80 SHARD intermediate, epoch-major subsample) |
| `graph_lambda` unused | **Fixed** — ridge scaling in `graph_term_map`; ablation selects λ (mean **10.0** smooth) |
| Formal T1 impossibility missing | **Fixed** — `paper/impossibility_t1.md` (Proposition + proof + simulator mapping) |
| Graph MAP degeneracy | **Already fixed R02**; R03 confirms T1-limited spread (~0.27 row-std smooth) |

---

## T1 impossibility (proof audit)

**Claim:** Stacked epoch terminals \(c^{(e)} = A^{(e)}\bar{s}\) identify \(\bar{s}\) only; individual deviations \(s_i - \bar{s}\) lie in a joint nullspace.

**Assessment:** **Sound** under stated assumptions (LASA linearity, one scalar terminal = batch mean of linear snapshots, full-rank \(A^{(e)}\) on \(\bar{s}\)).

**Simulator alignment:** In `benchmark_round03.py`, each epoch publishes `terminal_gradients.append(np.mean(grads_e, axis=0))` and T1 uses `terminal_batch_grads = [[c] for c in terminal_gradients]` with **0** `observed_intermediate_batch_gradients`. This matches the paper’s observation map \(\mathcal{A}(S) = \mathcal{B}(\bar{s})\).

**Empirical consistency:** Best T1 MSE ≈ passive (**1.0** MNIST; **0.987** smooth ≈ **2.04×** SHARD). New T1 attacks (`active_probe_graph_terminal`, `cross_epoch_consistency_terminal`) do not violate the proof—they re-spread around \(\bar{s}\) without new linear constraints on \(\delta_i\).

---

## Tier evaluations

### T1 (strict epoch terminal)

| Track | Best T1 method | MSE | vs SHARD full | Rows |
|-------|----------------|----:|-------------:|-----:|
| smooth | `graph_term_terminal` | 0.987 | 2.04× | 10 |
| mnist | `graph_term_terminal` | 1.000 | 1.65×* | 10 |

\*Ratio below 2× on MNIST reflects oracle MSE > passive floor, **not** individual recovery (row-std ≈ 10⁻⁴).

**Ruling:** **T1 tier ACCEPTED as impossibility outcome.** Does **not** satisfy `snapshot_mse_vs_shard_stage2_oracle` at T1 row budget; must not be relabeled as success in `config.json` without tier scoping.

### T1p honest (partial trajectory, imputation-free)

`partial_honest_disaggregate` uses only observed last-\(p\) rows; **no** `pad_partial_for_shard` on the headline path. Metrics carry `imputation_free: true`.

| Track | Best honest partial | MSE | Rows | vs SHARD full |
|-------|---------------------|----:|-----:|--------------:|
| smooth | `partial_honest_last_7` | 0.441 | 70 | 0.91× |
| mnist | `partial_honest_last_7` | 0.586 | 70 | 0.97× |

**Ruling:** Valid **T1p** positive result. This is **partial minibatch trajectory leak** (7 of 8 batch rows/epoch), not strict T1. Padded upper bounds are worse than honest at small \(p\) on smooth (e.g. p=1: honest 0.807 vs padded 0.540 MSE)—correctly shows imputation was inflating R02 claims.

**Gap (minor):** No **honest partial @ 80 rows** with B>1 (equal row budget to SHARD without N-client channel)—listed for Round 04+.

### T1b (B=1 per-client terminal @ matched budget)

| Comparison | smooth MSE | mnist MSE | Rows | Within-epoch intermediates |
|------------|----------:|----------:|-----:|---------------------------:|
| `shard_budget80_intermediate` | 0.677 | 0.574 | 80 | 80 (subsampled) |
| `b1_client_budget80_terminal` | 0.160 | 0.111 | 80 | 0 |
| Ratio b1 / shard @80 | **0.24×** | **0.19×** | equal | — |

**Ruling:** **T1b @ 80 rows** is a clean budget match and meets ≤2× at equal rows, but each row is \(A^{(e)} s_i\) (unmixed per-client)—a **stronger publication channel** than one epoch aggregate. Correctly tagged `non_acceptance_path` in `headline_by_tier`. Full T1b (320 rows) remains a separate identifiability demonstration (MSE ≈ 0.0015 smooth).

### T2 (SHARD oracle)

- `SurrogateQFL` + vendored `ShardAttacker` via `code/_paths.py` — **pass**
- Smooth SHARD MSE **0.483**, 80 intermediate rows, `shard_max_iter=200` — **pass**
- `matching_acc_mean = 0.0` while Hungarian MSE is good — **documented**; not fraud (`true_snapshots` used for matching metric only in oracle path)

---

## Config & threat-model compliance

| `config.json` / user constraint | Round 03 |
|----------------------------------|----------|
| `require_runnable_experiments` | **Yes** — `logs/experiment_round03.log`, `artifacts/round03_metrics.json` (smooth + `--mnist`) |
| `require_shard_stage2_baseline` | **Yes** — T2 oracle + budget-80 subsample |
| `require_no_intermediate_gradients` (within-epoch) | **Yes** for T1/T1b labels; T1p uses terminal-row leak only (0 intermediate count) |
| `require_individual_snapshot_recovery` | **No at T1** (proved impossible); **Yes at T1b/T1p/T2** with tier labels |
| `forbid_claim_without_qfl_surrogate` | **Yes** — `SurrogateQFL` throughout |
| `forbid_aggregate_only_endpoint` | **Not violated** — impossibility + tiered attacks documented |
| `snapshot_mse_vs_shard_stage2_oracle` @ comparable budget | **No for T1**; **Yes for T1b@80** (non-primary); **Borderline/yes for T1p@70** (different budget) |
| Approach SHARD | **Yes** — oracle, partial honest, B=1 disaggregation grounded in SHARD machinery |

---

## Honesty / reproducibility audit

| Check | Result | Evidence |
|-------|--------|----------|
| Metric injection | **Pass** | `hungarian_snapshot_mse` / `fixed_order_snapshot_mse` from arrays |
| T1 row accounting | **Pass** | 10 terminal rows, 0 intermediates |
| B=1 row accounting | **Pass** | 80 or 320 terminal rows logged |
| Partial honest vs padded | **Pass** | `imputation_free` flags; padded excluded from `T1p_partial_honest` headline |
| Impossibility ↔ code | **Pass** | Paper §Simulator reference ↔ `terminal_batch_grads = [[c]]` |
| Researcher proposal honesty | **Pass** | States T1 not met; impossibility path (`researcher_proposal.md`) |
| Headline misleading flag | **Minor fail** | MNIST `meets_2x_shard_at_t1_budget: true` while MSE ≈ 1.0 — should be overridden when `impossibility_accepted` |

**Cheating verdict:** **No fraud.** Round 02 integrity risks (B=1 headline, imputed partial) are **resolved**.

---

## Rubric (1–5)

| Criterion | Score | Note |
|-----------|------:|------|
| Novelty | 4 | Formal T1 nullspace + tier taxonomy completes identifiability story |
| Soundness | 4 | Proof matches simulator; budget-80 pairing explicit; T1p still assignment-heuristic |
| Feasibility | 5 | Reproducible benchmark, dual tracks, full JSON |
| Impact | 4 | Closes false hope for epoch-only terminals; charts honest partial / B=1 paths |
| Evaluability | 5 | `headline_by_tier`, row counts, imputation flags |

---

## Issues

| Tag | Issue |
|-----|-------|
| **Minor** | `config.json` `acceptance` block lacks explicit **tier gates** (T1 impossibility vs T1p/T1b success paths). |
| **Minor** | `headline_by_tier.T1_epoch_terminal.meets_2x_shard_at_t1_budget` true on MNIST is **misleading** alongside `impossibility_accepted: true`. |
| **Minor** | SHARD `matching_acc = 0%` vs good Hungarian MSE — needs threshold audit (Round 04). |
| **Minor** | Honest **partial @ 80 rows** (B=4, no B=1, no imputation) not yet run. |
| **Info** | `stop_mode: until_acceptance` — project may continue for T1p/T1b refinement even after T1 closure. |

**Critical issues:** **0** (Round 02 critical set cleared).

---

## Scoped acceptance decision

The project **can ACCEPT** Round 03 under this scope:

1. **T1 impossible (proved)** — primary scientific deliverable for the strict epoch-terminal model.
2. **T1p honest partial trajectory** — imputation-free; labeled non-T1; 70-row budget disclosed.
3. **T1b B=1 per-client @ matched 80-row budget** — positive at equal rows; labeled non-primary channel.
4. **No imputation cheats** in headline claims — padded partials and B=1@320 excluded from T1/T1p honest acceptance paths.

**Not accepted (by design):** Global claim that weak epoch-only terminals recover individuals within 2× SHARD.

---

## Round 04 suggestions (optional, non-blocking)

1. Amend `config.json` with `acceptance_tiers`: `{ "T1": "impossibility_or_2x", "T1p": "...", "T1b": "..." }`.
2. Set `meets_2x_shard_at_t1_budget` false whenever `impossibility_accepted` is true.
3. Run honest partial with **exactly 80** observed terminal rows (e.g. last rows across epochs without B=1).
4. JASPER-TERMINAL on honest partial; SHARD matching audit.

---

## Verdict summary

| Item | Status |
|------|--------|
| **Supervisor verdict** | **ACCEPT_WITH_MINOR** |
| **T1 (10 rows)** | **Impossibility accepted** |
| **T1p honest** | **Accepted bracket** (70 rows, no imputation) |
| **T1b @ 80 rows** | **Accepted bracket** (matched budget, non-primary) |
| **Imputation / B=1 headline cheat** | **Cleared** |
| **QFL surrogate + SHARD** | **Compliant** |
| **Primary config goal @ T1 only** | **Not met** (expected; documented) |
| **Recommend stop entire run?** | **No** — optional rounds for T1p@80 and assignment-aware terminals |
