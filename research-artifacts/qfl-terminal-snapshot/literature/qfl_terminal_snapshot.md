# QFL terminal snapshot recovery — literature & observation taxonomy

## Observation tiers (this run)

| Tier | Server sees | Individual \(s_i\) identifiable? |
|------|-------------|----------------------------------|
| T0 | Passive final model only | No (TALON class moments only under probes) |
| T1 | Terminal epoch summary \(c^{(e)} = A^{(e)}\bar{s}\) per probe round | **Mean \(\bar{s}\) only** (H1) |
| T1b | B=1 per-client terminal: \(g^{(e,i)} = A^{(e)} s_i\) (one row per sample per epoch) | **Yes** (Round 02: MSE ≈ 0.0015, 0 within-epoch intermediates) |
| T1p | Partial terminal: last \(p\) of \(K\) minibatch rows per epoch | **Partial** (Round 03 honest \(p{=}7\): MSE ≈ 0.44 vs SHARD 0.48, 70 rows, no imputation) |
| T1p-pad | Partial + mean-imputed early rows (`pad_partial_for_shard`) | Upper bound only (Round 03: padded better than honest at small \(p\)) |
| T2 | All intermediate \(g^{(e,k)}\) (SHARD Stage 2) | Yes under rank + assignment (vendored attacker) |
| T3 | T2 + snapshot inversion to \(x\) | Input recovery (SHARD L3) |

Round 01 implements **T1** attacks vs **T2** oracle. Round 02 adds **T1b**, **T1p**, graph ablations, and hardened SHARD diagnostics. Round 03 adds **formal T1 impossibility** (`paper/impossibility_t1.md`), **imputation-free partial**, and **80-row budget-matched** SHARD vs B=1 comparisons with **tier-specific headlines**.

## LASA linear structure (SurrogateQFL)

\[
g^{(e,k)} = A^{(e)} \bar{s}^{(e,k)}, \quad
c^{(e)} = \frac{1}{K}\sum_k g^{(e,k)} = A^{(e)} \bar{s}.
\]

Stacking \(c^{(e)}\) across epochs constrains the **same** \(\bar{s}\) under noiseless linearity; it does not constrain individual deviations \(s_i - \bar{s}\) without batch-mean rows \(\bar{s}^{(e,k)}\) with \(K>1\) batches per epoch.

## Parent TALON bridge (negative result)

- Terminal active probes identify **class aggregate** geometry in a linear head, not per-client feature vectors.
- **Implication:** Mechanisms that only stack terminal summaries need **extra structure** (graph, assignment, low-rank manifold, or per-client terminals) to lift to \(S\).

## Method families (candidates)

| Family | Needs beyond T1? | Round 01 |
|--------|------------------|----------|
| Mean broadcast | No | Baseline |
| Graph MAP (GARD-style) | Oracle graph on indices | `graph_term_terminal` (Round 02: Fiedler spread, not mean-only MAP) |
| Graph low-rank subspace | Rank + graph | `graph_rank_terminal` |
| SHARD Stage 2 | All batch rows + assignment | Oracle |
| Partial terminal SHARD | Last \(p\) rows/epoch, 0 early intermediates | `partial_terminal_last_*` |
| JASPER / soft assignment | Partial rows or soft incidence | Round 03+ |
| B=1 client terminals | Per-sample \(g\) each round | `b1_client_terminal` (Round 02) |

## Citations

- SHARD pipeline: `vendor/shard_sim/attacker.py`, `surrogate_model.py`
- Parent rounds: `/workspace/rounds/round_05`, `round_06` (GARD, JASPER assignment barriers)
- Snapshot inversion survey: `/workspace/literature/qfl_snapshot_inversion.md`
- Bridge doc: `literature/parent_talon_bridge.md`

## Round 02 findings

- **Graph degeneracy (Round 01):** Mean-only MAP has a constant Laplacian nullspace; anchor RHS is row-constant, so \(\lambda\) cannot spread rows. Fix uses Fiedler-mode prior + mean re-projection (still T1-limited; MSE ≈ 0.985 vs passive 1.0).
- **T1b identifiability:** B=1 terminals give linear measurements \(A^{(e)} s_i\) per client per epoch—SHARD B=1 path recovers individuals without within-epoch intermediates.
- **Partial trajectory:** MSE vs terminal-row count brackets SHARD; last 7/8 rows ≈ full oracle MSE on smooth synthetic data.

## Round 03 findings

- **T1 impossibility:** Stacked epoch terminals \(c^{(e)} = A^{(e)}\bar{s}\) depend only on \(\bar{s}\); all \(s_i - \bar{s}\) lie in a joint nullspace (proof in `paper/impossibility_t1.md`). Best T1 MSE ≈ 0.987 vs SHARD 0.483 (**2.04×**) on smooth — not an under-powered SHARD, a different observation class.
- **graph_lambda** now affects Fiedler spread; ablation is informative (best λ often 10.0 on smooth).
- **Honest partial** (`partial_honest_last_7`): MSE ≈ 0.44, no imputed rows; can beat padded-at-small-\(p\) but needs 70 row budget.
- **Budget 80:** `b1_client_budget80` MSE ≈ 0.16 vs `shard_budget80` ≈ 0.68 (ratio 0.24×) — 2× met at equal rows, but B=1 is per-client channel (non-acceptance for primary T1 goal).
- **New T1 attacks** (active-probe graph, cross-epoch consistency): do not break impossibility; MSE remains ≈ 0.99–1.0 on MNIST.

## Round 04 — LASA-QTERM acceptance package

- **Method name:** LASA-QTERM (alias Q-SNAP-T); production `code/qterm_attack.py`.
- **Papers:** `paper/method.md`, `paper/scope.md`.
- **Benchmark:** `code/benchmark_round04.py` — smooth + MNIST, `acceptance_table` in `artifacts/round04_metrics.json`.
- **Tutorial:** `tutorial/tutorial.md` (QFL-focused; parent TALON as related work).

## Open gaps

- Assignment-aware terminal tier without full row budget or per-client inflation.
- Honest **partial @ 80 rows** (equal SHARD budget, B>1, no imputation).
- Coupled **terminal head updates + LASA snapshots** in real QFL stacks.
- SHARD strict matching_acc vs Hungarian MSE (0% matching at good MSE).
