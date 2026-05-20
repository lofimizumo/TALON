# Round 05 — Researcher Proposal: Snapshot-DP defense (criterion B)

## Mission

Pivot from Round 04 assignment mechanisms to **client-side Snapshot-DP**: add Gaussian or Laplace noise to **batch-mean snapshots** before gradient publish. Measure privacy–utility tradeoff against three attacker channels:

| Attacker | Setting |
|----------|---------|
| **SHARD Stage-2** | Co-occurrence GARD, `wrong40`, fraction 0.25, `level1_estimate` |
| **JASPER-Q** | Round 04 joint soft GARD + T1p warm-start, same assignment/anchor |
| **LASA-QTERM T1p** | `QtermAttack` T1p, 7 partial rows/epoch, N=32 QFL track |

**Grid:** mechanisms `{gaussian, laplace}` × ε ∈ `{0.5,1,2,4,8,16}` × σ ∈ `{0.02,…,0.5}` × seeds `{3,7,11,19,23}`.

**Criterion B (primary):** attack snapshot MSE **increase** ≥ **50%** vs undefended **and** normalized utility loss ≤ **10%** on ≥1 grid cell (any attacker channel).

**Secondary (relaxed):** under best B-feasible config, JASPER-Q `wrong40` parent gate @ MSE **0.15** with ≥**50%** row reduction.

## Methods (`code/benchmark_round05.py`)

1. **Defense** — per-batch mean \(\bar s_B\) → \(\bar s_B + \mathcal{N}(0,\sigma^2)\) or Laplace; gradients from noisy mean via `SurrogateQFL`.
2. **Utility** — normalized batch-gradient MSE vs undefended (3-epoch proxy, Round 02).
3. **Privacy metric** — \((\mathrm{MSE}_\mathrm{def} - \mathrm{MSE}_\mathrm{undef}) / \mathrm{MSE}_\mathrm{undef}\) (higher = worse attack).

## Pass/fail vs `config.json`

| Criterion | Result | Pass? |
|-----------|--------|-------|
| **B** | 0/420 cells meet utility ≤10%; 0 meet B jointly | **Fail** |
| **Secondary** | No B-feasible config → not evaluated | **Fail** |

## Numeric results (honest)

### Undefended baselines (mean over seeds)

| Channel | Mean snapshot MSE |
|---------|------------------:|
| T1p | 0.427 |
| SHARD Stage-2 (wrong40 GARD) | 1.025 |
| JASPER-Q (wrong40) | 0.948 |

### Best privacy signal (fails utility)

| Config | T1p attack ↑ | Utility loss (norm.) |
|--------|-------------:|---------------------:|
| Gaussian ε=8, σ=0.5 (seed 19) | **+268%** | 285% |
| Closest-to-B (Laplace ε=4, σ=0.02) | +0.6% | **188%** (still ≫10%) |

Snapshot-DP **can** inflate T1p attack MSE at high σ, but **no** grid point satisfies the ≤10% utility gate. Minimum observed utility loss ≈ **188%** (Laplace σ=0.02). SHARD/JASPER Stage-2 attack MSE barely moves (<12% increase) at any feasible noise level tested.

## Conclusions

1. **Criterion B:** **Not met** — utility–privacy tradeoff is unfavorable for batch-mean DP on this surrogate; same class of failure as Round 02 QFL-SHIELD (0 feasible utility cells).
2. **Best defense config (operational):** none for B; **closest tradeoff** = Laplace ε=4, σ=0.02 (lowest utility harm in grid, still 18.8× budget).
3. **Strongest attack degradation:** Gaussian σ=0.5 (ε irrelevant when σ>0) on T1p only.
4. **Secondary row gate:** **Not met** (depends on B-feasible config).

## Round 06 preview

- Do **not** tune Snapshot-DP σ grid further without a new utility parameterization (e.g. loss on downstream task, clipping-aware sensitivity).
- Revisit **criterion A** with conditional T1p warm-start (Round 04 recommendation) or formal kill Snapshot-DP for rounds 6–10.

## Artifacts

- `code/benchmark_round05.py`
- `artifacts/round05_metrics.json`
- `logs/experiment_round05.log`
- `rounds/round_05/revision_log.md`

## Answers

1. **B pass?** **No.**
2. **Best defense config for B?** **None.** Closest: **Laplace, ε=0.5, σ=0.02.** Strongest T1p harm: **Gaussian, σ=0.5.**
