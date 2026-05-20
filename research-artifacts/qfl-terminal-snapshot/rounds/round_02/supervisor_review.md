# Round 02 — Supervisor Review

## Verdict

**REVISE_MAJOR**

Round 02 is a strong methodological response to Round 01: graph degeneracy is fixed, observation tiers are explicit, SHARD diagnostics are hardened, and the team is honest that **strict T1 (epoch means only) still fails** acceptance. The run does **not** satisfy `config.json` binding gates for the stated user goal—recovering individual LASA-VQC snapshots under a **defensible, budget-comparable** no-within-epoch-intermediate model. **Do not ACCEPT.** Continue to Round 03 with budget-normalized tiers, an imputation-free partial attack, and claim language scoped to T1 / honest partial leak—not B=1 row inflation.

---

## Executive summary

| Question | Answer |
|----------|--------|
| Is `b1_client_terminal` (MSE ≈ 0.0015, 0 intermediate, 320 rows) a valid no-within-epoch-intermediate QFL model? | **Partially valid as tier T1b**, not as acceptance for the primary “epoch terminal / weaker-than-SHARD” goal. It is **not** within-epoch minibatch leakage, but it is **per-sample published gradient rows** (32× per epoch), i.e. a **different, stronger** FL channel than one epoch summary per round. |
| Is B=1 equivalent to per-sample intermediate leakage? | **No** for within-epoch SGD steps (one publication per sample per epoch). **Yes in spirit** for identifiability: each row is \(g = A^{(e)} s_i\) with no batch mixing—closer to **N client updates per round** than to **one aggregated terminal**. |
| Does strict T1 meet acceptance? | **No** — best GRAPH-TERM MSE ≈ 0.985 vs SHARD ≈ 0.483 (~2.04× worse). |
| Is `partial_terminal_last_7_rows` (MSE ≈ 0.468, 0 intermediate counted) honest? | **No for acceptance** — attack **pads** missing early minibatch rows with \(A^{(e)}\bar{s}\) before calling full SHARD (`pad_partial_for_shard`). Metrics under-count leakage. |
| Graph MAP fix? | **Yes (implementation)** — row spread and MSE now differ from passive; **No (science goal)** — still far above 2× SHARD; `graph_lambda` is unused in `graph_term_map`. |

**Headline risk:** `artifacts/round02_metrics.json` selects `b1_client_terminal` as `best_terminal_method` and reports `terminal_mse_over_shard_mse ≈ 0.003`. That ratio is **not** comparable observation budget and must not be read as meeting `snapshot_mse_vs_shard_stage2_oracle`.

---

## Critical question: `b1_client_terminal`

### What the code does

```303:326:research-artifacts/qfl-terminal-snapshot/code/benchmark_round02.py
    # B=1 per-client terminal: one gradient per sample per epoch (no within-epoch steps)
    b1_attacker = ShardAttacker(
        dim_g=DIM_G,
        n_samples=N_SAMPLES,
        batch_size=1,
        ...
    )
    for e in range(N_EPOCHS):
        perm = np.random.default_rng(seed + 9100 + e).permutation(N_SAMPLES)
        batches_b1 = [[int(i)] for i in perm]
        a_e, grads_e = simulate_epoch_gradients(true_snapshots, batches_b1, surrogate)
        ...
    s_b1 = b1_attacker.level2_disaggregate(
        e_bar_b1, b1_coeff, b1_grads, true_snapshots
    )
```

Each epoch emits **N = 32** gradients (batch size 1 → \(g^{(e,k)} = A^{(e)} s_{\pi(k)}\)). SHARD’s B=1 fast path solves them directly and matches across epochs (`vendor/shard_sim/attacker.py:368–446`). **Within-epoch intermediate count = 0** is accurate: there is no multi-step local trajectory per sample before publication.

### Threat-model classification

| Model | Rows per epoch | Within-epoch intermediates | Linear measurement |
|-------|----------------|--------------------------|--------------------|
| **T1** (epoch terminal) | 1 (mean) | 0 | \(A^{(e)}\bar{s}\) only |
| **T1b** (B=1 client terminal) | **N = 32** | 0 | \(A^{(e)} s_i\) per row |
| **T2** (SHARD oracle, B=4) | **K = 8** batch rows | 80 total (all counted intermediate) | \(A^{(e)} \bar{s}^{(e,k)}_{\text{batch}}\) |

**T1b is defensible** as “each of N clients publishes exactly one gradient per federated round, no intra-client minibatch telemetry.” It is **not defensible** as the same observation class as Round 01 T1 or as “strictly weaker than SHARD Stage 2”:

- Row budget: **320 terminal rows** vs SHARD **80** intermediate rows (4× more).
- Information per row: **unmixed** per-sample constraints vs **B=4** batch means that require assignment.

So MSE ≈ 0.0015 demonstrates **identifiability when the server sees per-client gradients**, not that **epoch-averaged terminals** suffice. It is **not** a cheat in the sense of smuggling hidden intermediate rows, but it **is** a tier change that the acceptance criterion explicitly couples to **comparable observation budget**.

### Cheat vs valid?

- **Not** equivalent to leaking all within-epoch minibatch steps under B=4 (those would be 7×8 = 56 early rows per epoch if defined as pre-final batches).
- **Is** equivalent to running SHARD with batch size 1 and a full client roster each round—a **stronger channel** than the user’s primary “terminal-only” framing in `config.json` goal text.

**Supervisor ruling:** Document T1b as a **separate positive result**; do **not** use it to claim global acceptance.

---

## `partial_terminal_last_7_rows` (MSE ≈ 0.468, 0 intermediate)

Observed: **70** terminal rows (7 last minibatch gradients × 10 epochs). Reported `observed_intermediate_batch_gradients = 0`.

Attack path:

```328:339:research-artifacts/qfl-terminal-snapshot/code/benchmark_round02.py
        partial_grads = partial_epoch_gradients(batch_gradients, rows_per_epoch=p)
        padded = pad_partial_for_shard(
            coeff_matrices, partial_grads, e_bar_term, k_full
        )
        ...
        s_p = attacker.level2_disaggregate(
            e_bar_p, coeff_matrices, padded, true_snapshots
        )
```

```112:124:research-artifacts/qfl-terminal-snapshot/code/terminal_attacks.py
def pad_partial_for_shard(...):
    """Impute missing early-minibatch rows with ``A^(e) @ e_bar`` so SHARD sees K rows."""
    ...
        imputed = [coeff_matrices[e] @ e_bar for _ in range(missing)]
        padded.append(imputed + list(partial_e))
```

For \(p=7\), **one row per epoch is synthetic** from Level-1 mean \(\bar{s}\) (already recovered to ~3×10⁻⁴ rel error). SHARD then runs on **full K = 8 rows/epoch**. This is a **hybrid mean-imputation + partial-leak** attack, not a pure terminal observer.

**Supervisor ruling:** Useful **upper-bound bracketing** (shows last rows carry most signal). **Fails** `require_no_intermediate_gradients` as an honest end-to-end threat model unless imputed rows are disclosed as **derived side information** and excluded from budget comparisons. Do not treat partial-7 as meeting acceptance without an imputation-free baseline.

---

## Strict T1 (epoch means only)

| Method | Mean MSE (smooth) | Terminal rows | Intermediate rows |
|--------|------------------:|--------------:|------------------:|
| SHARD Stage-2 oracle | 0.483 | 0 | 80 |
| Passive mean | 1.000 | 10 | 0 |
| GRAPH-TERM (best λ sweep) | 0.985 | 10 | 0 |
| GRAPH-RANK-TERM | 0.989 | 10 | 0 |

T1 **confirms H1**: stacked epoch terminals identify \(\bar{s}\) only; graph prior adds Fiedler spread (~0.27 row-std) but cannot approach SHARD without extra row budget. Ratio vs oracle: **~2.04× worse** (not ≤ 2×).

MNIST track: graph spread ~10⁻⁴ row-std; GRAPH-TERM ≈ passive (1.0). Tiering story holds; graph arm still not informative on encoded snapshots.

---

## Graph MAP fix (Round 01 major issue)

| Check | Round 01 | Round 02 |
|-------|----------|----------|
| `graph_term_map` vs passive row-std | ~10⁻¹⁵ | ~0.267 |
| MSE vs passive | 1.0 | 0.985 |
| λ ablation changes MSE | N/A (degenerate) | **No** — `graph_lambda` unused in body of `graph_term_map` (`terminal_attacks.py:32–63`) |
| Wrong-graph ablation | N/A | Differs trivially from rank-truncated graph |

**Fix verdict:** Degeneracy **resolved**; graph is now a real **prior template**, not new measurements. **Not** a path to ≤2× SHARD at T1 budget.

---

## Honesty / reproducibility audit

| Check | Result | Evidence |
|-------|--------|----------|
| Metric injection | **Pass** | MSE from `hungarian_snapshot_mse`; no hardcoded wins |
| Terminal intermediate count (T1 / graph) | **Pass** | Only `[[c]]` epoch terminals; 0 intermediates |
| B=1 intermediate count | **Pass** | 0 within-epoch; 320 terminal rows correctly logged |
| Partial “0 intermediate” semantics | **Fail (labeling)** | SHARD fed padded full trajectories; 10 imputed rows/epoch hidden from budget |
| SHARD oracle uses truth only for logging | **Pass** | `true_snapshots` for matching acc only |
| SurrogateQFL / vendored SHARD | **Pass** | `benchmark_round02.py`, `_paths.py` |
| Runnable experiment | **Pass** | `logs/experiment_round02.log`, JSON artifact |
| Researcher honesty on acceptance | **Pass** | Proposal states T1 fails; pivots tier language (`researcher_proposal.md:53–59`) |
| Headline JSON vs acceptance | **Fail** | `best_terminal_method = b1_client_terminal` implies success under goal wording |

**Cheating verdict:** No fraud detected. The main integrity issue is **observation-budget conflation** (B=1 and imputed partial) in headline metrics—not fabricated MSE.

---

## Acceptance criteria mapping (`config.json`)

| Criterion | Round 02 |
|-----------|----------|
| `require_individual_snapshot_recovery` | **No** at T1; **Yes** at T1b (different tier); **Partial** at T1p with imputation |
| `snapshot_mse_vs_shard_stage2_oracle` ≤ 2× at **comparable** budget | **No** for T1 (~2.04×); **N/A** for B=1 (4× row budget); **Borderline** for partial-7 with dishonest padding |
| `require_no_intermediate_gradients` | **Yes** for T1/T1b row labels; **Violated in spirit** for partial padding |
| `require_shard_stage2_baseline` | **Yes** (200 iter; matching still 0% strict) |
| `require_runnable_experiments` | **Yes** |
| `forbid_aggregate_only_endpoint` | **Not violated** as success claim |
| `forbid_claim_without_qfl_surrogate` | **Yes** |
| `max_critical_issues` | **> 0** (budget + partial imputation + headline) |

**User goal alignment:** “QFL individual snapshots **without intermediate gradients** (as in SHARD Stage 2 pipeline)” is met only if interpreted as **no within-epoch steps** **and** the delivered method is **weaker than full SHARD**. Round 02 proves the opposite for the main tier: **T1 fails**; success requires **more rows** (T1b) or **imputation** (T1p).

---

## Rubric scores (1–5)

| Criterion | Score | Note |
|-----------|------:|------|
| Novelty | 4 | Clear tier taxonomy (T1 / T1b / T1p); impossibility story sharpening |
| Soundness | 3 | B=1 and partial tiers need budget-normalized evaluation; λ ablation null |
| Feasibility | 5 | Runnable, reproducible, good diagnostics |
| Impact | 3 | Negative T1 + positive T1b bracket the problem; not yet a deployable weak attack |
| Evaluability | 4 | JSON, fixed-order MSE, row counts; headline needs tier-specific ratios |

---

## Issues

| Tag | Issue |
|-----|-------|
| **Critical** | **Acceptance not met** for primary T1 / weak-terminal goal; headline uses B=1 (320 rows) vs SHARD (80 rows). |
| **Critical** | **`partial_terminal_last_*` imputation** — `pad_partial_for_shard` feeds synthetic early rows; “0 intermediate” misleads budget accounting. |
| **Critical** | **`config.json` comparable-budget clause** ignored in `headline.terminal_mse_over_shard_mse`. |
| **Major** | **`graph_lambda` unused** in `graph_term_map`; ablation table is non-informative. |
| **Major** | **SHARD matching_acc = 0%** despite good MSE — strict metric vs Hungarian MSE; audit threshold (Round 03). |
| **Minor** | MNIST graph still ~passive; smooth-only graph success may not transfer. |
| **Minor** | Proposal “97% matching” for B=1 is seed-dependent (one seed 84.375% on smooth). |

---

## Actionable suggestions (Round 03, prioritized)

1. **Split headlines by tier** — Report `terminal_mse / shard_mse` separately for T1 (10 rows), T1p (observed rows only), T1b (N×E rows), with **rows-per-sample** normalization (e.g. MSE vs SHARD at equal total row budget).
2. **Imputation-free partial attack** — Run SHARD only on observed last-\(p\) rows (no `pad_partial_for_shard`) or joint estimator that does not require full K; report degradation vs padded upper bound.
3. **Formal T1 impossibility** — Proposition: stacked \(c^{(e)} = A^{(e)}\bar{s}\) leaves \(s_i - \bar{s}\) in joint nullspace under stated noise model.
4. **Fix or remove λ sweep** — Wire `graph_lambda` into `graph_term_map` or drop ablation claims.
5. **JASPER-TERMINAL / assignment-aware partial** — Target honest partial row budget without N-client equivalence.
6. **Clarify B=1 FL mapping** — One paragraph: when N clients × 1 gradient/round is realistic vs when only epoch aggregate is published (reject conflation).

---

## Round 03 gate (supervisor)

Advance to **ACCEPT** or **ACCEPT_WITH_MINOR** only when **all** hold:

1. At least one attack at the **primary T1 budget** (≈ E epoch terminal rows, no within-epoch intermediates, no per-client N-fold row inflation) achieves Hungarian snapshot MSE **≤ 2 × SHARD Stage-2 MSE** on smooth **and** MNIST, **or**
2. A written impossibility result (explicit assumptions) covers T1 and is accepted as the scientific outcome **with** budget-normalized positive results for T1b/T1p clearly labeled **non-acceptance** paths.

Until then: **REVISE_MAJOR** — Round 02 is valuable science, not a stopping round.

---

## Verdict summary

| Item | Status |
|------|--------|
| **Supervisor verdict** | **REVISE_MAJOR** |
| **B=1 valid tier?** | Yes (T1b), not cheat on within-epoch intermediates |
| **B=1 meets user/config acceptance?** | **No** (stronger row budget; different FL channel) |
| **Partial-7 honest weak terminal?** | **No** (mean-imputed padding) |
| **Strict T1** | **Fails** (~2× worse than SHARD) |
| **Graph fix** | **Implementation fixed; goal not met** |
| **Recommend stop?** | **No** (`stop_mode: until_acceptance`) |
