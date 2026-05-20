# TALON/TANGO Manuscript Draft (Round 10)

Working title: **Beyond SHARD: Tiered Gradient Leakage and Identifiability Limits Under Active Terminal Probing**

Formal proofs: `paper/proofs.md`. LaTeX fragment: `paper/method.tex`.

---

## Abstract

Federated learning often leaks only **terminal** client model updates, not the intermediate minibatch gradients required by SHARD-style individual reconstruction. We introduce **TALON**, an observation-tier framework, and **TANGO**, an **active terminal probing** attack that recovers class counts, class sums, and hidden prototypes under a linear-head, full-batch, first-order model. We prove aggregate identifiability (Theorem 1) and individual non-identifiability (Theorem 2) with assumptions matching `theorem_scope.exact`. Experiments on synthetic hidden-feature FL show strong prototype recovery in the exact regime (prototype MSE $\approx 1.5\times 10^{-4}$) but **honest failure** under minibatch SGD (TANGO MSE $\approx 6.59$ vs passive $\approx 0.75$). **Count recovery and prototype recovery must be reported separately**—passive baselines can outperform TANGO on counts. A Round-10 decoder probe links recovered hidden prototypes to synthetic pixel class means when TANGO succeeds; frozen nonlinear features preserve Tier-1 accuracy. We do **not** claim individual sample recovery from terminal-only observations.

---

## 1. Limits (lead section for reviewers)

### 1.1 Minibatch SGD breaks the default estimator

| Scenario | TANGO prototype MSE | Passive prototype MSE | TANGO count MAE |
|---|---:|---:|---:|
| balanced_clean (exact) | 0.000151 | 0.143 | 0.0419 |
| minibatch_sgd | **6.5877** | **0.7532** | 2.5743 |

Under minibatch training, the first-order full-batch mapping in Lemma A is invalid. Active TANGO is **worse** than passive multi-round on prototypes ($\approx 0.11\times$ gain). The exact theorems apply only to full-batch training; minibatch FL requires corrected estimators (future work).

### 1.2 Prototype win $\neq$ count win

On `balanced_clean`, passive count MAE (0.0152) beats TANGO (0.0419) while TANGO dominates prototype MSE. On `large_10class_30dim`, passive count MAE (0.0038) beats TANGO (0.0294). Summaries must not imply “active dominates passive on all metrics.”

### 1.3 No individual recovery

Theorem 2: within-class zero-sum perturbations leave terminal observations unchanged. Empirically, `high_within_class_variance` individual MSE $\approx 0.983$ vs within-class floor $\approx 0.837$.

### 1.4 Evaluation scope

Synthetic simulator with fixed (or frozen) hidden features and linear head. Not a deployed CNN FL system; not secure aggregation.

---

## 2. Contributions

1. Observation tier ladder (TALON) separating terminal, partial intermediate, and SHARD tiers.
2. TANGO active probing algorithm with separated count/prototype metrics.
3. Formal proofs (Appendix `paper/proofs.md`).
4. Reproducible benchmarks Rounds 08–10 with baselines and negative results.
5. Decoder bridge experiment (Round 10): decodable pixel means track recovered hidden prototypes when Tier-1 succeeds.

---

## 3. Method summary

See `paper/method.tex` and `tutorial/tutorial.md` §3–4.

---

## 4. Experiments (headline numbers)

**Round 09** (`artifacts/round09_metrics.json`): baselines, stress tests, theorem scope labels.

**Round 10** (`artifacts/round10_metrics.json`):

- **Decoder probe:** pixel class-mean MSE via $D\hat\mu_c$ when TANGO accurate; degrades under minibatch.
- **Frozen MLP features:** nonlinear $\phi(x)$ fixed; only head trains—Tier-1 prototype MSE remains low.

---

## 5. Related work pointer

SHARD (vendored `vendor/shard_sim`) requires intermediate gradients. TANGO targets strictly weaker observations and weaker privacy targets (aggregates, not individuals).

---

## 6. Conclusion

Active terminal probing identifies class-level hidden prototypes and aggregate moments in an exact linear full-batch regime, with formal proofs and explicit limits. Minibatch SGD, count–prototype asymmetry, and individual non-identifiability must remain visible in every summary table and the abstract.
