# TALON/TANGO Manuscript Draft (Round 12)

Working title: **Beyond SHARD: Tiered Gradient Leakage and Identifiability Limits Under Active Terminal Probing**

Formal proofs: `paper/proofs.md`. LaTeX fragment: `paper/method.tex`.

---

## Abstract

Federated learning often leaks only **terminal** client model updates, not intermediate minibatch gradients required by SHARD-style reconstruction. We introduce **TALON** (observation tiers) and **TANGO** (active terminal probing) for recovering class counts, class sums, and hidden prototypes under a linear-head, first-order model. We prove aggregate identifiability in the full-batch exact regime and **Lemma MB-A** for minibatch divisor correction at \(W_0=0\), with **Lemma MB-B** drift bounds tied to code. Round 12 shows **TANGO-MB** on primary `minibatch_sgd`: prototype MSE **0.011** (median **0.011**) vs passive **0.753**; count MAE **0.100** vs passive **0.295**; scaling-only passive+MB **0.284** — most gain is **active probing**, not bookkeeping alone. **STORM** is removed; **TANGO-MB-drift2** and **stack-MB-ridge** add differentiated tiers. Count and prototype metrics are reported **separately**. Individual sample recovery remains impossible (Theorem 2).

---

## 1. Limits (lead section)

### 1.1 Minibatch SGD (Phase-2 primary)

| Method | Prototype MSE (mean / median) | Count MAE |
|---|---:|---:|
| TANGO vanilla | 6.588 / — | — |
| Passive multi-round | 0.753 / 0.751 | 0.295 |
| Passive + MB scaling only | **0.284** / 0.282 | — |
| **TANGO-MB (active)** | **0.011** / **0.011** | **0.100** |
| TANGO-MB-drift2 | 0.025 / — | — |

Vanilla TANGO mis-scales terminal deltas (\(T\) vs \(T_{\mathrm{eff}}=T(N/B)\)). TANGO-MB fixes Lemma MB-A; residual \(\sim 0.01\) MSE vs exact \(1.5\times 10^{-4}\) is bounded by within-step drift (Lemma MB-B; mean \(\|W_M-W_0\|_F \approx 0.11\) in `artifacts/round12_metrics.json`).

**Active vs scaling:** passive+MB improves over passive (\(0.28\) vs \(0.75\)) but active TANGO-MB adds **\(\approx 26\times\)** further prototype gain on the primary scenario.

### 1.2 Dual metrics

Count recovery uses the least-perturbed probe round for bias moments under MB; prototype recovery uses full active stack. Do not merge into a single “attack success” score.

### 1.3 Scope

Synthetic hidden-feature / frozen-MLP simulator; not deployed CNN FL; not secure aggregation; SHARD L3 baseline still required at project level (`config.json`).

---

## 2. Contributions (Rounds 09–12)

1. TALON tier ladder vs SHARD intermediate-gradient requirement.
2. TANGO active probing with separated metrics.
3. Formal proofs + Lemma MB-A/B/Iter (`paper/proofs.md`).
4. Reproducible benchmarks through Round 12 (`code/benchmark_round12.py`).
5. Broader stress: label noise, terminal noise, frozen MLP + minibatch, median/IQR reporting.

---

## 3. Experiments (Round 12 headlines)

Source: `artifacts/round12_metrics.json`, log `logs/experiment_round12.log`.

**Primary `minibatch_sgd` (8 seeds):** see table §1.1.

**`minibatch_nonzero_init`:** TANGO-MB mean **0.017**; **tango_mb_iter** (one Jacobian step, \(\alpha=0.15\)) mean **0.013** vs vanilla **10.08**.

**Robustness:** `minibatch_label_noise` (8% flips), `minibatch_terminal_noise` (\(\sigma=0.002\)), `frozen_mlp_minibatch` — see JSON scenario headlines.

---

## 4. Conclusion

Minibatch failure mode is addressed by principled \(T_{\mathrm{eff}}\) scaling plus explicit drift and count estimators. Phase-2 primary prototype and count gates pass on the simulator; global ACCEPT still requires supervisor sign-off and SHARD comparison per project config.
