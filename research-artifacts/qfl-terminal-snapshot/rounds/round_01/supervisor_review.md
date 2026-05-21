# Round 01 — Supervisor Review

## Verdict

**REVISE_MAJOR**

Round 01 is an honest identifiability baseline, not a pass on `config.json` acceptance. Metrics are computed, not injected; the negative result is real. The run still fails every binding acceptance gate (individual recovery, ≤2× SHARD Stage-2 MSE, no aggregate-only success). Round 02 must add observation tiers that can break the mean-collapse floor and stress-test graph/assignment claims.

---

## Executive summary

The scientist framed Round 01 correctly: terminal epoch summaries under LASA linearity identify the dataset mean, not per-sample snapshots \(S\). Code uses `SurrogateQFL` and `ShardAttacker` from the parent vendor path; terminal attacks report `observed_intermediate_batch_gradients = 0`. SHARD Stage-2 oracle MSE ≈ 0.44 vs terminal MSE ≈ 1.00 (ratio **2.25× worse**, not within 2×). Skeptical audit: **MSE = 1.0 is not cheating**—it is the Hungarian-aligned ceiling for mean-broadcast recovery on unit-normalized snapshots. **Graph methods are not meaningfully implemented**: `graph_term_map` / `graph_rank_terminal` collapse numerically to passive broadcast (row spread ~10⁻¹⁴) because the mean anchor weight (`sqrt(100)`) dominates the Laplacian prior. SHARD oracle is a weak upper bound here (0% matching accuracy, residuals not converging at 50 iterations). No TALON-style prototype-only success is claimed. Proceed to Round 02 with stronger observation models and ablations; do not accept aggregate/mean recovery as meeting the QFL individual-snapshot goal.

---

## Honesty / reproducibility audit

| Check | Result | Evidence |
|-------|--------|----------|
| Hardcoded terminal MSE | **Pass** | No literal `1.0` assignment in `code/benchmark_round01.py` or `code/terminal_attacks.py`; values come from `hungarian_snapshot_mse()` |
| MSE = 1.0 plausibility | **Pass (explained)** | Independent repro: constant mean broadcast on `make_smooth_snapshots()` yields MSE ≈ 1.0 for all seeds; equals per-dimension variance (~1) after global std normalization (`benchmark_round01.py:74–76`, `106–114`) |
| Terminal intermediate count | **Pass** | Terminal path uses `terminal_batch_grads = [[c] for c in terminal_gradients]` only (`benchmark_round01.py:161–163`); JSON/logs show 0 intermediate rows for terminal methods |
| SHARD oracle leakage via `true_snapshots` | **Pass** | `ShardAttacker.level2_disaggregate` uses `true_snapshots` only for matching-accuracy logging (`vendor/shard_sim/attacker.py:191–192`); recovery uses batch gradients only |
| SurrogateQFL requirement | **Pass** | `SurrogateQFL` instantiated in `benchmark_round01.py:123–129`; simulator field in `artifacts/round01_metrics.json` |
| Aggregate-only endpoint | **Pass (failure, not cheat)** | Passive mean broadcast is explicit baseline; proposal states acceptance **not met** (`researcher_proposal.md:87`) |
| Graph vs passive differentiation | **Fail (implementation)** | `graph_term_map` output row-std ~10⁻¹⁴ vs truth ~0.93; max row diff from passive ~10⁻¹⁴ (supervisor repro) |
| Runnable experiment | **Pass** | `logs/experiment_round01.log` shows full 5-seed run; JSON written with per-seed breakdown |
| Cherry-picked seeds | **Minor risk** | Fixed `SEEDS = [3,7,11,19,23]` only; no failure reported but N=5 is thin |

**Cheating verdict:** No evidence of metric injection or smuggled intermediate gradients. The suspicious uniformity of **1.000000** across methods is a consequence of (a) normalization, (b) mean collapse, and (c) graph MAP degeneracy—not fraud.

---

## Feasibility & resources

- **Compute:** ~5 s total for 5 seeds (`runtime_sec_mean` ≈ 1.1 per SHARD run)—well within budget.
- **Dependencies:** Parent `vendor/shard_sim` resolves via `code/_paths.py`; benchmark ran successfully in supervisor repro.
- **Scope:** Round 01 is appropriately narrow (T1 vs T2 oracle). Risk: staying on synthetic smooth snapshots too long without MNIST/real QFL geometry (`revision_log.md:28–29`).
- **SHARD oracle budget:** 50 iterations insufficient for stable Stage-2 on several seeds (matching 0%, residual flat ~20–60 in log). Oracle MSE may be pessimistic or unstable; still far below terminal 1.0.

---

## Theory & assumptions

- **H1 (terminal → mean only):** Supported empirically: `level1_terminal_mean_rel_error` ≈ 3.5×10⁻⁴ matches full-batch Level-1 (`artifacts/round01_metrics.json`).
- **H2 (graph spreads individuals):** **Not tested fairly** in code. With `weight = np.sqrt(100.0)` in `terminal_attacks.py:38–40`, the anchor term forces \(S \approx \mathbf{1}\bar{s}^\top\) regardless of `graph_lambda` (supervisor sweep: λ ∈ {0.01, 0.5, 5, 50} all give row_std = 0, MSE = 1.0).
- **H3 (TALON non-transfer):** Correctly cited; no prototype-only success claimed.
- **Oracle chain graph:** Documented as side information (`benchmark_round01.py:291–292`) but batches are re-permuted each epoch (`make_epoch_batches`)—graph prior on index order does not align with FL batch geometry. Honest negative, but graph arm is currently a **null experiment**.
- **Identifiability proof:** Literature notes gap (`literature/qfl_terminal_snapshot.md:48`); Round 01 provides empirical bound only.

---

## Rubric scores (1–5)

| Criterion | Score | Note |
|-----------|------:|------|
| Novelty | 3 | Clear QFL snapshot threat tiering; Round 01 mostly confirms expected linear nullspace |
| Soundness | 4 | Threat model and metrics coherent; graph arm implementation undermines H2 test |
| Feasibility | 4 | Runnable, vendored SHARD path, realistic Round 02 extensions listed |
| Impact | 2 | Negative result only; no method beats mean floor |
| Evaluability | 4 | JSON + logs + seed-level rows; needs ablations and convergence diagnostics |

---

## Issues

| Tag | Issue |
|-----|-------|
| **Critical** | **Acceptance failed:** `snapshot_mse_vs_shard_stage2_oracle` requires ≤2× **or better**; observed `terminal_to_shard_mse_ratio` = **2.25** (terminal worse). |
| **Critical** | **No individual snapshot recovery:** All terminal methods ≡ mean broadcast; Hungarian MSE ≈ 1.0 is the mean-collapse ceiling, not partial success. |
| **Critical** | **QFL goal not satisfied:** User mandate rejects TALON-style aggregate/prototype success; Round 01 delivers aggregate mean only. |
| **Major** | **Graph methods degenerate:** `graph_term_map` / `graph_rank_terminal` numerically identical to `passive_mean_broadcast` (`terminal_attacks.py:38–43`, `59–66`). |
| **Major** | **SHARD oracle under-converged:** 0% matching accuracy all seeds; 50 iter residuals high (`logs/experiment_round01.log`). Weakens “approach Stage-2” comparison. |
| **Major** | **Misleading headline ratio:** “2.25×” reads like nearness; it means terminal is **2.25× worse** than oracle. Clarify in Round 02 tables. |
| **Minor** | Only synthetic smooth rank-4 snapshots; oracle-favorable graph yet graph arm still useless—confounds “graph can’t help” vs “implementation can’t help”. |
| **Minor** | Hungarian metric masks per-index error when order is known synthetically—add fixed-order MSE diagnostic. |

---

## Actionable suggestions (Round 02, prioritized)

1. **Fix graph MAP experiment** — Rebalance or decouple mean anchor vs `graph_lambda` (`terminal_attacks.py:38–43`); report row-diversity diagnostic and MSE vs passive before claiming graph failure. Include **permuted / wrong graph** ablation (`revision_log.md:27`).
2. **Add observation tier T1b: B=1 client terminals** — One gradient per sample per epoch (still no within-epoch intermediates); test whether per-client terminals lift identifiability without full SHARD row budget.
3. **Partial terminal trajectory sweep** — Leak budget: last minibatch row per epoch, 1 vs 2 vs K−1 rows; plot MSE vs observation count to bracket SHARD gap.
4. **Harden SHARD oracle** — Increase `max_iter`, report convergence (residual, matching acc); optionally separate “SHARD saturated” vs “SHARD budget-limited” columns so oracle is a fair ceiling.
5. **MNIST / real SurrogateQFL snapshots** — Replace or augment `make_smooth_snapshots` (`benchmark_round01.py:58–77`) per `config.json` domain notes.
6. **Assignment-aware terminal candidate** — JASPER-TERMINAL or soft incidence (Round 03 preview) once graph ablation is honest.
7. **Metrics clarity** — Report `terminal_mse / shard_mse` explicitly as “× worse than oracle”; add non-Hungarian MSE when index order is known.
8. **Short identifiability note** — Formalize T1: stacked \(c^{(e)} = A^{(e)}\bar{s}\) leaves \(s_i - \bar{s}\) in joint nullspace; cite in proposal before Round 03 theory push.

---

## Acceptance criteria mapping (`config.json`)

| Criterion | Round 01 |
|-----------|----------|
| `require_individual_snapshot_recovery` | **No** |
| `snapshot_mse_vs_shard_stage2_oracle` ≤ 2× | **No** (2.25× worse) |
| `require_no_intermediate_gradients` (terminal tier) | **Yes** for declared terminal methods |
| `require_shard_stage2_baseline` | **Yes** (present; convergence weak) |
| `require_runnable_experiments` | **Yes** |
| `forbid_aggregate_only_endpoint` | **Not violated as success claim** |
| `forbid_claim_without_qfl_surrogate` | **Yes** |

**Stop rule:** `stop_mode: until_acceptance` — continue; Round 01 does not warrant `ACCEPT` or `ACCEPT_WITH_MINOR`.

---

## Round 02 gate (supervisor)

Do not advance claim language to “terminal approaches SHARD” until:

- At least one terminal-tier method achieves **Hungarian snapshot MSE ≤ 2 × SHARD Stage-2 MSE** on the same benchmark, **or**
- A written impossibility result (assumptions explicit) covers the tested tiers with the same metrics.

Until then, treat MSE ≈ 1.0 as the **documented failure floor**, not a rounding artifact.
