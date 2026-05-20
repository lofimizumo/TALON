# LASA-QTERM — Scope and observation-tier table

This document fixes **what each tier means** for acceptance review. Method details: `paper/method.md`. T1 proof: `paper/impossibility_t1.md`.

## Primary user goal (`config.json`)

Recover **individual** LASA-VQC snapshots in QFL **without within-epoch intermediate gradients**, approaching SHARD Stage-2 fidelity under a **strictly weaker** observation channel—**at comparable row budget** when claiming numeric parity.

## Tier table

| Tier | Code / enum | Server observes | Within-epoch intermediates | Individual \(s_i\) from data alone? | Acceptance role |
|------|-------------|-----------------|----------------------------|-------------------------------------|-----------------|
| **T1** | `QtermTier.T1` | One epoch-averaged \(c^{(e)} = A^{(e)}\bar{s}\) per round | **0** | **No** — identifies \(\bar{s}\) only | **Impossible** (Proposition T1); best attack ≈ passive scale |
| **T1p** | `QtermTier.T1P` | Last \(p\) minibatch rows per epoch (honest) | **0** (early rows not published) | **Partial** — needs \(p>1\) batch means | **Non-primary** path; imputation-free; 70 rows default |
| **T1b** | `QtermTier.T1B` | B=1 per-client \(g^{(e,i)} = A^{(e)} s_i\) | **0** | **Yes** | **Non-primary** — stronger FL channel (N rows/round) |
| **T2** | (oracle) | All \(E \times K\) intermediate \(g^{(e,k)}\) | **80** (default sim) | **Yes** (SHARD Stage 2) | **Upper bound** baseline |

### T1 — impossible for primary goal

Stacked terminals depend only on \(\bar{s}\). Any \(\Delta\) with \(\Delta^\top \mathbf{1} = 0\) is in the joint nullspace. Graph priors without extra rows cannot break this (likelihood unchanged).

**Scientific acceptance for T1:** Document impossibility + report best LASA-QTERM T1 MSE vs SHARD with explicit **10 vs 80** row mismatch (~2.04× on smooth).

### T1p — honest partial (not strict T1)

- **Honest:** `partial_honest_disaggregate` — no `pad_partial_for_shard`.
- **Upper bound only:** padded partial (labeled, not in `QtermAttack` production).

### T1b — per-client terminals (not epoch aggregate)

- Full: 320 terminal rows (\(N \times E\)).
- Budget-80: matched to SHARD 80-row subsample — can meet ≤2× **at equal rows** but is **not** the primary “epoch terminal only” deployment.

### T2 — SHARD Stage 2 oracle

Full intermediate trajectory; Hungarian MSE is the reference. Strict `matching_acc` may remain 0% while MSE is good (documented).

## Budget-matched comparison (required headline)

| Comparison | Row budget | Smooth (5 seeds, R04) | MNIST (5 seeds, R04) |
|------------|------------|------------------------|----------------------|
| SHARD subsampled | 80 intermediate | see `round04_metrics.json` | same |
| LASA-QTERM T1b @80 | 80 terminal | same | same |
| Ratio B1/SHARD @80 | equal | report `b1_mse_over_shard_budget80` | same |

Do **not** divide T1 (10 rows) MSE by SHARD (80 rows) and call it acceptance.

## What LASA-QTERM claims

| Claim | Supported? |
|-------|------------|
| T1 individual recovery from epoch terminals only | **No** (impossibility) |
| T1p honest partial can approach SHARD on smooth at 70 rows | **Empirical** (tier-labeled) |
| T1b B=1 recovers individuals without within-epoch steps | **Yes** (stronger channel) |
| Production code is tier-explicit and imputation-free for T1p | **Yes** (`qterm_attack.py`) |

## Out of scope (this artifact)

- Real quantum hardware noise beyond SurrogateQFL Gaussian
- SHARD L3 input inversion (Tier 3)
- Overwriting parent TALON prototypes / class-aggregate results
- Claiming `config.json` primary acceptance from T1b or T1p alone

## Round 03 gaps closed in Round 04

| Gap | Round 04 deliverable |
|-----|----------------------|
| No packaged method name | **LASA-QTERM** / Q-SNAP-T |
| No production API | `code/qterm_attack.py` |
| Papers split | `paper/method.md`, `paper/scope.md` |
| Benchmark fragmentation | `code/benchmark_round04.py` + acceptance table |
| Tutorial stub | `tutorial/tutorial.md` |
| MNIST track | Always on in R04 benchmark |

## Supervisor gate (unchanged logic)

**ACCEPT** or **ACCEPT_WITH_MINOR** when:

1. T1 impossibility is written and accepted **and** tier headlines are budget-honest, **or**
2. Primary T1 achieves ≤2× SHARD at T1 row budget on smooth **and** MNIST.

Current outcome: path (1) — impossibility + tier-separated positives.
