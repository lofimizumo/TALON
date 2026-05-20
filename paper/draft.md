# TALON/TANGO Manuscript Draft (Round 13)

Working title: **Beyond SHARD: Tiered Gradient Leakage and Identifiability Limits Under Active Terminal Probing**

Formal proofs: `paper/proofs.md`. LaTeX fragment: `paper/method.tex`.

---

## Abstract

Federated learning often leaks only **terminal** client model updates, not intermediate minibatch gradients required by SHARD-style reconstruction. We introduce **TALON** (observation tiers) and **TANGO** (active terminal probing) for recovering class counts, class sums, and hidden prototypes under a linear-head, first-order model. We prove aggregate identifiability in the full-batch exact regime and **Lemma MB-A** for minibatch divisor correction at \(W_0=0\), with **Lemma MB-B** drift bounds (mean \(\|W_M-W_0\|_F \approx 0.112\)). Round 13 retests **TANGO-MB** on primary `minibatch_sgd`: prototype MSE **0.011** (median **0.011**, IQR **0.005**) vs passive **0.753**; count MAE **0.100** vs passive **0.295**. **TANGO-JOINT**, **TANGO-COUPLED**, and trajectory-midpoint estimators **do not** beat TANGO-MB (means **0.27–0.28**). Honest scaling-only ablation **`passive_mb_scale_only`** yields **0.151** (median **0.140**) — active gain over scaling is **\(\approx 14\times\)**, not the inflated **\(\approx 26\times\)** from Round 12’s coupled `passive_mb`. Individual sample recovery remains impossible (Theorem 2).

---

## 1. Limits (lead section)

### 1.1 Minibatch SGD (Phase-2 primary)

| Method | Prototype MSE (mean / median / IQR) | Count MAE |
|---|---:|---:|
| TANGO vanilla | 6.588 / — | — |
| Passive multi-round | 0.753 / 0.751 | 0.295 |
| **passive_mb_scale_only (R11)** | **0.151** / **0.140** / 0.032 | — |
| **TANGO-MB (active)** | **0.011** / **0.011** / 0.005 | **0.100** |
| TANGO-JOINT | 0.283 / 0.280 / 0.038 | 0.195 |
| TANGO-COUPLED | 0.272 / 0.276 | — |

Vanilla TANGO mis-scales terminal deltas (\(T\) vs \(T_{\mathrm{eff}}=T(N/B)\)). TANGO-MB fixes Lemma MB-A; residual \(\sim 0.01\) MSE vs exact \(1.5\times 10^{-4}\) correlates with within-step drift (Lemma MB-B; mean \(\|W_M-W_0\|_F \approx 0.112\) in `artifacts/round13_metrics.json`).

**Active vs scaling (honest):** scaling-only **0.151** vs passive **0.753**; active TANGO-MB adds **\(\approx 14\times\)** over scaling-only (not \(\approx 26\times\) using Round 12 coupled passive_mb **0.284**).

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
