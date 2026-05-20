# Phase 2 — Scoped Acceptance Package (Round 14)

This document defines what **scoped Phase-2 ACCEPT** means for TALON/TANGO under minibatch SGD in the bundled simulator. It is the stakeholder-facing fence for claims, failures, and open external-validation gaps.

---

## 1. What TANGO-MB claims (in scope)

Under **terminal-only** client observations (no intermediate minibatch gradients, no batch-order oracle):

| Assumption | Detail |
|---|---|
| **Lemma MB-A regime** | Linear softmax **head**, **\(W_0 = 0\)**, PyTorch-style local SGD with **\(1/B\)** per-minibatch gradient normalization |
| **Effective steps** | Terminal weight/bias deltas scaled by \(T_{\mathrm{eff}} = T \cdot (N/B)\), not raw local step count \(T\) |
| **Active probing** | Server-chosen bias probes across rounds; full-rank design on stacked terminal moments |
| **Count path** | `estimate_counts_mb`: least-perturbed probe round for bias moments (not joint stack over aggressive rounds) |
| **Targets recovered** | Class counts \(n_c\), class sums \(S_c\), hidden prototypes \(\mu_c = S_c/n_c\) — **aggregate** leakage |
| **Primary metric** | Prototype MSE and count MAE reported **separately**; aggregates use **median + IQR** over 8 seeds |

**Empirical primary (`minibatch_sgd`, Round 14):** see `artifacts/round14_metrics.json` → `phase2_primary`. TANGO-MB must beat passive multi-round on prototype MSE every seed (`phase2_primary_win`).

**Honest decomposition:** `passive_mb_scale_only` isolates Lemma MB-A bookkeeping; active probing adds further gain (~14× over scaling-only on primary, not ~68× conflated with passive vanilla).

---

## 2. What TANGO-MB does **not** claim (out of scope)

| Exclusion | Rationale |
|---|---|
| General deployed CNN / FedAvg FL | Simulator is synthetic hidden features / frozen MLP head |
| Representation drift during local training | Frozen-feature experiments partial only |
| Individual sample recovery | Theorem 2 / JASPER barrier; terminal aggregates insufficient |
| SHARD-equivalent Tier-3 attack | Requires intermediate batch-gradient rows + incidence |
| Secure aggregation robustness | Masked global sums hide per-client probe structure (see benchmark secure-agg note) |
| Batch-order / permutation oracle | Minibatch order not observed or inverted |
| Exact tier under minibatch | Residual prototype MSE ~0.01 vs ~\(1.5\times 10^{-4}\) full-batch exact |

---

## 3. What failed (documented negatives)

Round 13–14 tested harder inversion beyond \(T_{\mathrm{eff}}\) scaling. **None beat TANGO-MB** on primary `minibatch_sgd`.

| Estimator | Mechanism | Primary proto MSE (typical) | Verdict |
|---|---|---:|---|
| **TANGO-JOINT** | Uniform-weight joint LS on stacked bias + weight moments | ~0.28 (R14; distinct from DOPT) | **Failed** |
| **TANGO-DOPT** | D-opt round weights \(w_r \propto 1/\|p^{(r)}-\mathbf{1}/C\|\) | ~0.28 | **Failed**; may differ slightly from JOINT |
| **TANGO-COUPLED** | Fixed-point bias Jacobian + joint invert | ~0.27 | **Failed**; unstable counts on some seeds |
| **trajectory_midpoint** (R13) | \(W_{\mathrm{mid}}\) bias correction | ~0.27 | **Failed** |
| **tango_mb_drift2** (R12) | Ad hoc \(T_{\mathrm{eff}}\) inflation | ~0.025 | **Failed** vs MB 0.011 |

**Interpretation (Lemma MB-JOINT):** Aggressive probe rounds break a shared-\(n_c\) linear bias model; joint stacking **harms** counts relative to single-round selection. These are **informative negatives**, not limits-section substitutes.

---

## 4. SHARD baseline (`level3_invert`)

**Config gate:** `require_baseline_comparison_vs_shard` → `vendor/shard_sim/attacker.py level3_invert`.

**Round 14 cross:** `code/shard_cross_round14.py` (synthetic snapshots, no MNIST download).

| Method | Observation tier | Target | Metric (R14 synthetic cross) |
|---|---|---|---|
| SHARD L1–L3 | Tier 3 — all intermediate batch gradients | Individual inputs | `level3_reconstruction_mse` (see JSON) |
| TANGO-MB | Tier 1 — terminal deltas only | Class prototypes | `prototype_mse` median on `minibatch_sgd` |
| SHARD on terminal-only | **N/A** | — | Cannot reconstruct batch-mean snapshot rows |

**Tier conclusion:** SHARD and TANGO-MB solve **different problems** under **different observations**. Phase-2 scoped ACCEPT does **not** claim TANGO beats SHARD at individual reconstruction; it claims defensible **aggregate** recovery when intermediate gradients are **not** leaked.

---

## 5. Phase-2 acceptance checklist

| Criterion | Scoped ACCEPT? | Evidence |
|---|---|---|
| `require_minibatch_primary_win` | **Yes** (simulator) | `phase2_primary.phase2_primary_win` |
| `forbid_primary_eval_on_fullbatch_only` | **Yes** | Primary = `minibatch_sgd` |
| `no_limits_section_as_substitute_for_fix` | **Yes** | Failed estimators implemented + measured |
| `require_baseline_comparison_vs_shard` | **Yes (tier table)** | `shard_baseline_cross` in R14 JSON |
| `require_stage2_or_full_method_improvement` | **Partial** | Terminal-tier aggregate win; not SHARD individual tier |
| `max_critical_issues: 0` | **No** | Deployed FL / representation drift remain external |
| **Holistic supervisor ACCEPT** | **Conditional** | Scoped simulator claims only; see §2 |

**Supportable verdict:** **Scoped Phase-2 ACCEPT** for Lemma MB-A regime + honest negatives + SHARD tier cross. **Not** holistic ACCEPT for deployment-grade FL or SHARD-individual parity.

---

## 6. Stakeholder wording (safe)

> Under terminal-only observation, linear head, \(W_0=0\), and correct minibatch gradient scaling (Lemma MB-A), TANGO-MB recovers class prototypes and counts on the primary synthetic minibatch benchmark, with median/IQR reporting. Joint, coupled, and D-opt-weighted extensions **fail** empirically. SHARD-style individual recovery requires strictly stronger (intermediate-gradient) observations and is **not** claimed.
