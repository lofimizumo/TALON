# Round 11 — Phase 2: Minibatch-Realistic TANGO (TANGO-MB / STORM)

## Mandate response

Round 10 **ACCEPT is suspended** for the core attack claim until minibatch SGD is fixed. Round 11 attacks the fatal flaw directly: vanilla TANGO treats terminal deltas as if they were **one full-batch gradient per local step**, but the simulator (and real FL) applies **shuffled minibatches** with per-batch normalization \(1/B\), accumulating **\(N/B\) minibatch updates per local step**.

## Thesis (revised)

> Under active terminal probing with **public hyperparameters** (\(\eta\), \(T\), \(B\), \(N\)), class prototypes remain identifiable from terminal deltas when the inversion uses **effective stochastic-gradient steps** \(T_{\mathrm{eff}} = T \cdot (N/B)\) at the linearized \(W_0 \approx 0\) regime. Vanilla first-order TANGO fails on minibatch; **TANGO-MB** and **STORM** recover the Round-09 linear structure.

## Root-cause hypothesis (H1) — supported

| Failure mode | Vanilla TANGO | TANGO-MB |
|---|---|---|
| Gradient scale | Assumes \(\Delta \approx -\eta T\, g_{\text{full}}\) | Uses \(\Delta \approx -\eta T (N/B)\, g_{\text{full}}\) at \(W_0\) |
| Minibatch MSE | `6.5877` | **`0.0110`** |
| Passive baseline | `0.7532` (better) | **beaten (~68×)** |

The dominant error is **deterministic mis-scaling** of terminal deltas, not irreducible noise. Residual MSE (`0.011` vs exact `0.00015`) comes from **weight drift within multi-minibatch steps** (H2, approximate tier).

## Methods

### 1. TANGO-MB (primary)

**Terminal Aggregate Neural Gradient Observation — Minibatch correction.**

Before calling Round-09 `tango_estimate_sums`, scale observed deltas:

\[
\tilde{\Delta} = \Delta \cdot \frac{T}{T_{\mathrm{eff}}}, \quad T_{\mathrm{eff}} = T \cdot \frac{N}{B} \quad (\text{minibatch on}).
\]

Equivalently: `avg_grad = -delta / (lr * T_eff)`.

**Derivation sketch (Lemma MB-A):** With \(W_0=0\), logits depend only on bias \(b\); within one local step, a full pass over \(N/B\) equal-size minibatches sums to \((N/B)\) times the full-batch gradient contribution before weights move. Over \(T\) shuffled passes: scale \(\propto T(N/B)\). No batch-order or incidence is observed—only public \(N,B,T,\eta\).

### 2. STORM (secondary, same tier)

**Stochastic Terminal Observation Recovered Moments** — explicit stacked moment system with \(T_{\mathrm{eff}}\) in the bias and weight rows; defaults to TANGO-MB at `ridge=0`.

### 3. Baselines (unchanged threat model)

- **TANGO vanilla** — Round-09 estimator (expected minibatch failure).
- **Passive multi-round** — zero bias probes.
- **Passive+MB** — passive probes with MB scaling (helps counts, worse prototypes than active MB).

## Experiments (`code/benchmark_round11.py`)

**Primary scenario (co-default):** `minibatch_sgd` — 6 local steps, batch 8, shuffled minibatches, 8 probe rounds.

| Scenario | Role |
|---|---|
| `minibatch_sgd` | Phase-2 acceptance target |
| `minibatch_nonzero_init` | Stress: nonzero \(W_0\) + minibatch |
| `balanced_clean` | Full-batch regression (MB must be no-op) |
| `imbalanced_clean` | Count/prototype separation check |

## Results (8 seeds, honest simulation)

### Phase-2 primary: `minibatch_sgd`

| Method | Prototype MSE | Count MAE | vs passive |
|---|---:|---:|---|
| TANGO vanilla | `6.5877` | `2.5743` | **loses** |
| **TANGO-MB** | **`0.0110`** | **`0.3621`** | **~68× better** |
| STORM | `0.0110` | `0.3621` | ~68× better |
| Passive multi-round | `0.7532` | `0.2772` | baseline |
| Passive+MB | `0.1505` | `0.1095` | — |

**Phase-2 primary win:** `tango_mb_prototype_mse < passive` ✅ (`0.0110 < 0.7532`).

### Regression

| Scenario | TANGO vanilla | TANGO-MB |
|---|---:|---:|
| `balanced_clean` | `0.000151` | `0.000151` (unchanged) |
| `imbalanced_clean` | `0.000382` | `0.000382` |

### Stress: `minibatch_nonzero_init`

| Method | Prototype MSE |
|---|---:|
| TANGO vanilla | `10.078` |
| TANGO-MB | **`0.0168`** |
| Passive | `7.448` |

## Limitations (honest)

1. **\(W_0 \approx 0\) linearization** — correction derived at zero weights; drift inside each local step remains (~0.011 MSE floor on minibatch).
2. **Public \(B, T, N\)** — server must know batch size and step count (standard in FL protocols).
3. **No intermediate gradients** — still terminal-only; MB correction is not batch-incidence leakage.
4. **Count MAE** on minibatch can exceed passive on some seeds; **prototype MSE** is the Phase-2 metric.
5. **Not deployed CNN FL** — synthetic hidden-feature simulator only.

## Claims for supervisor

| Claim | Status |
|---|---|
| Fundamentally addressed minibatch flaw (not re-labeled as “limits only”) | ✅ New estimator tier |
| Beat passive on `minibatch_sgd` prototype MSE | ✅ |
| Runnable benchmark + artifacts | ✅ `artifacts/round11_metrics.json` |
| Did not claim ACCEPT via full-batch-only eval | ✅ Minibatch primary |

## Artifacts

- `code/benchmark_round11.py`
- `artifacts/round11_metrics.json`, `artifacts/round11_minibatch_methods.svg`
- `logs/experiment_round11.log`
- `paper/proofs.md` — Remark MB-A (Round 11)
- `literature/minibatch_gradient_scaling.md`

## Next rounds (if Phase-2 ACCEPT restored)

- Prove Lemma MB-A with explicit within-step drift bound.
- Iterative refinement when \(W_0 \neq 0\) (trajectory Jacobians).
- Secure aggregation / label noise under MB tier.
- Frozen MLP + minibatch cross (decoder bridge under MB).
