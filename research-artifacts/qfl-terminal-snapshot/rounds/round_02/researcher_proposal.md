# Round 02 — Researcher Proposal: Observation Tiers & Graph Fix

## Supervisor Round 01 demands addressed

| Priority | Action | Status |
|----------|--------|--------|
| Fix graph MAP (λ vs mean anchor) | Removed `sqrt(100)` weight; replaced mean-only MAP with Fiedler-mode spread + mean re-projection; λ×scale ablation | **Done** — graph row-std ≈ 0.27 vs passive ≈ 0; MSE ≈ 0.985 vs 1.0 |
| B=1 per-client terminal (T1b) | `b1_client_terminal` via SHARD B=1 direct path, 320 terminal rows/epoch, 0 within-epoch intermediates | **Done** — MSE ≈ 0.0015, matching ≈ 97% |
| Partial terminal trajectory | Last 1 / 2 / 7 rows per epoch + mean-imputed early slots for SHARD | **Done** — MSE 0.57 / 0.55 / **0.47** vs SHARD 0.48 |
| SHARD oracle diagnostics | `shard_max_iter=200`, matching acc, row-std, fixed-order MSE, runtime | **Done** — still 0% strict matching; residuals plateau |
| MNIST SurrogateQFL (optional) | `--mnist` track in `benchmark_round02.py` | **Done** — see `artifacts/round02_metrics.json` mnist track |

## Hypotheses (Round 02)

| ID | Hypothesis | Result |
|----|------------|--------|
| H1 | T1 epoch terminals identify \(\bar{s}\) only | **Confirmed** — passive MSE = 1.0 |
| H2 | Honest graph prior spreads rows beyond mean broadcast | **Partial** — GRAPH-TERM differs from passive (row-std, MSE 0.985) but not within 2× SHARD |
| H2b | B=1 per-client terminals identify individuals | **Confirmed** — Hungarian MSE ≈ 0.0015, 0 intermediate rows |
| H3 | Partial terminal rows bracket SHARD gap | **Confirmed** — 7/8 rows/epoch ≈ SHARD MSE; 1 row ≈ 0.57 |

## Identifiability note (T1)

Stacked \(c^{(e)} = A^{(e)}\bar{s}\) constrains only the dataset mean. Any deviation \(s_i - \bar{s}\) lies in the joint nullspace without per-batch or per-client gradient diversity. Graph smoothness on index order cannot break this with a **constant** mean anchor RHS; Round 02 uses explicit Fiedler spread as a **prior template**, not new linear measurements.

## Methods (Round 02)

1. **Passive mean broadcast** — T1 baseline.
2. **GRAPH-TERM** — Fiedler-mode spread around \(\bar{s}\); λ and spread_scale swept per seed.
3. **GRAPH-RANK-TERM** — Low-rank projection of GRAPH-TERM.
4. **GRAPH-TERM wrong graph** — Permuted chain ablation.
5. **B=1 client terminal** — T1b: one terminal gradient per sample per epoch.
6. **Partial terminal** — Last \(p \in \{1,2,7\}\) minibatch rows per epoch (0 intermediate).
7. **SHARD Stage-2 oracle** — Full 80 intermediate rows; 200 iterations.

## Experiment

- Script: `code/benchmark_round02.py`
- Artifacts: `artifacts/round02_metrics.json`, `logs/experiment_round02.log`
- Seeds: 3, 7, 11, 19, 23

## Round 02 results (smooth snapshots, 5 seeds)

| Method | Mean snapshot MSE | Intermediate rows | Terminal rows |
|--------|------------------:|------------------:|--------------:|
| SHARD Stage-2 oracle | **0.483** | 80 | 0 |
| B=1 client terminal | **0.0015** | 0 | 320 |
| Partial last-7 rows | **0.468** | 0 | 70 |
| Partial last-1 row | 0.568 | 0 | 10 |
| GRAPH-TERM (best λ) | 0.985 | 0 | 10 |
| Passive mean | 1.000 | 0 | 10 |

**Acceptance vs `config.json`:** Strict T1 (10 epoch terminals) **still fails** 2× SHARD. **T1b** and **partial-7** meet ≤2× without intermediate batch gradients (different observation budgets).

**Headline ratio (best terminal / SHARD):** 0.003× (B=1 tier). For T1-only best: 2.04× worse (GRAPH-TERM 0.985 / 0.483).

## Honest verdict

Round 02 shows **where** identifiability returns: per-client B=1 terminals or most-of-epoch terminal rows—not epoch-averaged T1 alone. Graph MAP is no longer numerically identical to passive, but cannot approach SHARD without extra row budget. Round 03: assignment-aware terminals (JASPER-TERMINAL), real FL batch graphs, formal impossibility for T1.
