# Round 03 — Researcher Proposal: T1 Impossibility & Budget-Honest Evaluation

## Supervisor Round 02 demands addressed

| Priority | Action | Status |
|----------|--------|--------|
| Formal T1 impossibility | `paper/impossibility_t1.md` (Proposition + proof) | **Done** |
| Imputation-free partial | `partial_honest_disaggregate` (no `pad_partial_for_shard` in headline tier) | **Done** |
| Budget-matched comparison | 80 B=1 terminal rows vs 80 SHARD intermediate rows | **Done** |
| Tier-specific headlines | `headline_by_tier` in `artifacts/round03_metrics.json` | **Done** |
| Wire `graph_lambda` | Ridge scaling in `graph_term_map`; ablation now selects λ | **Done** |
| New T1 attacks (B>1, no intermediates) | `active_probe_graph_terminal`, `cross_epoch_consistency_terminal` | **Done** (still T1-limited) |
| No misleading global “best terminal” | Removed; per-tier ratios only | **Done** |

## Scientific outcome (honest)

**Primary T1 acceptance (`config.json` at epoch-terminal budget):** **Not met** — accepted via **impossibility** instead.

- Smooth: best T1 method GRAPH-TERM MSE **0.987** vs SHARD **0.483** → **2.04×** worse (10 terminal rows).
- MNIST: T1 MSE ≈ **1.0** (passive scale); graph/active/cross-epoch cannot lift individuals.

**T1b at matched 80-row budget:** B=1 terminals MSE **0.16** vs subsampled SHARD **0.68** (ratio **0.24×**) — meets 2× at equal rows but is a **stronger FL channel** (per-client unmixed rows), labeled **non-acceptance** for the primary goal.

**T1p honest (imputation-free):** Last 7 minibatch rows/epoch, 70 terminal rows — MSE **0.44** vs SHARD **0.48** on smooth (**0.91×**); no synthetic early rows. This is **partial trajectory leak**, not strict T1.

## Methods (Round 03)

| Method | Tier | Rows (typical) |
|--------|------|----------------|
| Passive / GRAPH-TERM / active-probe / cross-epoch | T1 | 10 |
| `partial_honest_last_p` | T1p honest | 10p |
| `partial_padded_last_p` | T1p upper bound | 10p (+ hidden imputation) |
| `b1_client_terminal_full` | T1b | 320 |
| `b1_client_budget80_terminal` | T1b @ 80 rows | 80 |
| `shard_stage2_oracle` | T2 | 80 intermediate |
| `shard_budget80_intermediate` | T2 @ 80 rows | 80 |

## Experiment

- Script: `code/benchmark_round03.py`
- Artifacts: `artifacts/round03_metrics.json`, `logs/experiment_round03.log`
- Proof: `paper/impossibility_t1.md`
- Seeds: 3, 7, 11, 19, 23 (smooth + MNIST)

## Headline table (smooth, 5 seeds)

| Tier | Best method | Mean MSE | vs SHARD full | Meets 2× at tier budget? |
|------|-------------|----------|---------------|---------------------------|
| T1 | graph_term_terminal | 0.987 | 2.04× | **No** (impossibility) |
| T1p honest | partial_honest_last_7 | 0.441 | 0.91× | Yes vs full SHARD* |
| T1b full | b1_client_terminal_full | 0.0015 | 0.003× | N/A (320 rows) |
| budget-80 | b1_client_budget80 | 0.160 | 0.24× vs shard_b80 | Yes @ 80 rows |
| T2 | shard_stage2_oracle | 0.483 | 1.0× | Oracle |

\*70 observed terminal rows/epoch-slice; not comparable to 10-row T1.

## Verdict

Round 03 closes the **T1 identifiability** question under LASA linearity: epoch-averaged terminals cannot recover \(s_i - \bar{s}\). **Accept T1 impossibility** as the scientific result; continue rounds on **honest partial** and **assignment-aware** attacks without inflating row budget or imputing gradients.
