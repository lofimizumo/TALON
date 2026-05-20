# Appendix: Formal Proofs for TALON/TANGO

This appendix provides complete proofs for the two main identifiability results used in the manuscript. Assumption lists match `theorem_scope.exact` in `artifacts/round09_metrics.json` / `artifacts/round10_metrics.json`.

---

## Notation and Setup

- Client has \(N\) samples \(\{(x_i, y_i)\}_{i=1}^N\) with fixed hidden features \(x_i \in \mathbb{R}^d\), labels \(y_i \in \{1,\ldots,C\}\).
- Class count \(n_c = |\{i : y_i = c\}|\), class sum \(S_c = \sum_{i:y_i=c} x_i\), total sum \(S = \sum_{c=1}^C S_c\).
- Class prototype \(\mu_c = S_c / n_c\) (defined when \(n_c > 0\)).
- Linear softmax head: logits \(z_i = W^\top x_i + b\), probabilities \(\pi_{i,c} = \mathrm{softmax}(z_i)_c\).
- Cross-entropy loss \(\mathcal{L} = -\sum_i \log \pi_{i,y_i}\).
- Federated round \(r\): server sets **known** initial bias \(b^{(r)} \in \mathbb{R}^C\), **zero** initial weights \(W_0 = 0\).
- Client runs \(T\) **full-batch** gradient steps with learning rate \(\eta\); server observes terminal deltas \(\Delta W^{(r)} = W_T - W_0\), \(\Delta b^{(r)} = b_T - b^{(r)}\).
- Server knows \(\eta, T, N\) and all \(b^{(r)}\). Define one-step average gradients (recovered from deltas):
  \[
  \bar g^{(r)}_c = -\frac{1}{\eta T}\,\frac{\partial \mathcal{L}}{\partial b_c}\Big|_{\text{terminal}}, \qquad
  \bar h^{(r)}_c = -\frac{1}{\eta T}\,\frac{\partial \mathcal{L}}{\partial W_{\cdot,c}}\Big|_{\text{terminal}}.
  \]
- Probe probabilities \(p^{(r)}_c = \mathrm{softmax}(b^{(r)})_c\).

---

## Lemma A (First-order full-batch terminal gradients)

**Assumptions:** linear head, \(W_0 = 0\), full-batch updates, first-order terminal observation (ignore higher-order terms in \(\eta T\)).

**Claim:** For each round \(r\) and class \(c\),
\[
N\,\bar g^{(r)}_c = p^{(r)}_c\, S - S_c, \qquad
N\,\bar h^{(r)}_{c,k} = p^{(r)}_c\, S_k - S_{c,k} \quad (k = 1,\ldots,d).
\]

**Proof.** With \(W_0 = 0\), initial logits equal \(b^{(r)}\), so initial probabilities are \(p^{(r)} = \mathrm{softmax}(b^{(r)})\), independent of \(x\).

For one full-batch step, standard softmax CE derivatives give
\[
\frac{\partial \mathcal{L}}{\partial b_c} = \sum_{i=1}^N (\pi_{i,c} - \mathbf{1}[y_i = c]), \qquad
\frac{\partial \mathcal{L}}{\partial W_{k,c}} = \sum_{i=1}^N (\pi_{i,c} - \mathbf{1}[y_i = c])\, x_{i,k}.
\]
Because \(W_0 = 0\), \(\pi_{i,c} = p^{(r)}_c\) for all \(i\) at initialization. Hence
\[
\frac{\partial \mathcal{L}}{\partial b_c} = N p^{(r)}_c - n_c, \qquad
\frac{\partial \mathcal{L}}{\partial W_{k,c}} = p^{(r)}_c S_k - S_{c,k}.
\]
Terminal deltas scale accumulated gradients by \(\eta T\) under first-order full-batch training, yielding the stated \(\bar g, \bar h\) identities. ∎

---

## Lemma B (Count identifiability from bias deltas)

**Assumptions:** Lemma A; known \(p^{(r)}\); at least one round with invertible bias-to-gradient map; \(n_c \ge 1\).

**Claim:** \(n_c\) is uniquely determined from \(\{\bar g^{(r)}_c, p^{(r)}\}_{r,c}\).

**Proof.** From Lemma A,
\[
\bar g^{(r)}_c = p^{(r)}_c - n_c/N.
\]
Thus \(n_c = N\,(p^{(r)}_c - \bar g^{(r)}_c)\) for every \((r,c)\). Averaging over rounds with the same \(c\) yields a consistent estimator when probes vary. Uniqueness follows because true \(n_c\) is the only value satisfying all exact equations simultaneously. ∎

---

## Theorem 1 (Aggregate identifiability — exact tier)

**Assumptions (matches `theorem_scope.exact`):**

1. Linear softmax head on fixed hidden features.
2. Zero initial head weights \(W_0 = 0\) (or known fixed \(W_0\) absorbed into public reparameterization).
3. Full-batch local training; terminal deltas equal first-order aggregate gradients scaled by known \(\eta, T, N\).
4. Server chooses known biases \(\{b^{(r)}\}\) such that the stacked probe design is full rank (see below).
5. Consistency constraint \(S = \sum_{c=1}^C S_c\).

**Claim:** The aggregate moments \(S\), \(\{S_c\}_{c=1}^C\), \(\{n_c\}\), prototypes \(\{\mu_c\}\), and dataset mean \(S/N\) are uniquely identifiable from terminal observations \(\{\Delta W^{(r)}, \Delta b^{(r)}, b^{(r)}\}\).

**Proof.**

*Counts.* Lemma B.

*Class sums (coordinate-wise).* Fix feature coordinate \(k\). Stack unknowns \(u = (S_k, S_{1,k}, \ldots, S_{C,k})^\top\). For each \((r,c)\), Lemma A gives linear equation
\[
p^{(r)}_c S_k - S_{c,k} = N\,\bar h^{(r)}_{c,k}.
\]
Rewrite as row \(a_{r,c}^\top u = N\bar h^{(r)}_{c,k}\) with \(a_{r,c} = (p^{(r)}_c, 0, \ldots, -1, \ldots, 0)\) (minus one in position \(1+c\)).

Add consistency row \(S_k - \sum_c S_{c,k} = 0\), i.e. \((1, -1, \ldots, -1)^\top u = 0\).

Let \(A_k\) be the matrix of all such rows over \(r,c\) plus the consistency row. **Full-rank probe design** means \(\mathrm{rank}(A_k) = C+1\) for every \(k\). Then \(u\) is the unique least-squares / exact solution of \(A_k u = b_k\) (TANGO's `lstsq` per coordinate in `benchmark_round09.py`).

Repeat for \(k = 1,\ldots,d\) to recover full vectors \(S, S_1, \ldots, S_C\).

*Prototypes and mean.* Lemma B gives \(n_c\). For \(n_c > 0\), \(\mu_c = S_c / n_c\) is unique. Dataset mean \(S/N\) is unique. ∎

**Remark (probe design).** Full rank requires sufficiently many distinct \(p^{(r)}\) (active probing). A single neutral round is rank-deficient; Round 09 passive baseline illustrates this empirically.

---

## Theorem 2 (Individual non-identifiability)

**Assumptions:** Same observation model as Theorem 1 (terminal aggregate updates only); hidden features fixed per sample; no side information beyond aggregates.

**Claim:** Individual samples \(\{x_i\}\) are **not** identifiable from terminal aggregate observations. Infinitely many distinct datasets induce identical terminal deltas.

**Proof.** Consider any class \(c\) with \(n_c \ge 2\). Pick any nonzero perturbations \(\delta_i \in \mathbb{R}^d\) for \(i : y_i = c\) such that
\[
\sum_{i: y_i = c} \delta_i = 0.
\]
Form perturbed features \(\tilde x_i = x_i + \delta_i\) for \(i\) in class \(c\), and \(\tilde x_i = x_i\) otherwise.

1. **Class sums unchanged:** \(\tilde S_c = \sum_{i:y_i=c} \tilde x_i = S_c + \sum \delta_i = S_c\).
2. **Counts unchanged:** labels unchanged.
3. **Total sum unchanged:** \(\tilde S = S\).
4. **Gradients unchanged:** For \(W_0 = 0\), \(\pi_{i,c}\) depends only on \(b^{(r)}\), not on \(x_i\). Bias derivatives depend only on \(n_c\). Weight derivatives:
   \[
   \frac{\partial \mathcal{L}}{\partial W_{\cdot,c}} = \sum_{i:y_i=c} (\pi_{i,c} - 1) x_i
   \]
   with identical scalar \((\pi_{i,c}-1) = p^{(r)}_c - 1\) for all \(i\) in class \(c\) at initialization. Hence
   \[
   \sum_{i:y_i=c} (\pi_{i,c}-1)\tilde x_i
   = (p^{(r)}_c-1)\sum_{i:y_i=c} \tilde x_i
   = (p^{(r)}_c-1) S_c,
   \]
   same as before perturbation.

Therefore all terminal \(\Delta W^{(r)}, \Delta b^{(r)}\) match. Two distinct datasets (e.g. swap two within-class feature vectors) can yield identical observations unless additional constraints (e.g. intermediate gradients, per-sample snapshots) are introduced.

**Corollary:** Any estimator of individual \(x_i\) from terminal aggregates alone has unavoidable error at least of order the within-class variance floor (see `individual_mse_from_prototypes` in benchmarks). ∎

---

## Lemma MB-A (Minibatch effective steps at \(W_0 = 0\))

**Assumptions:** Same as Lemma A at fixed \(W_0 = 0\) **within one local step** (weights frozen while summing minibatch contributions); client performs one shuffled pass per local step with equal minibatch size \(B\); \(N\) divisible by \(B\); PyTorch-style per-minibatch normalization \(1/B\); public \(\eta, T, B, N\).

**Claim:** Let \(M = N/B\). Over one local step at \(W_0\), the summed minibatch bias gradient equals \((N/B)\) times the full-batch bias gradient at \(W_0\). Over \(T\) local steps (reshuffled each step),
\[
\Delta b \approx -\eta\, T_{\mathrm{eff}}\, \nabla_b \mathcal{L}\Big|_{W_0}, \quad
\Delta W \approx -\eta\, T_{\mathrm{eff}}\, \nabla_W \mathcal{L}\Big|_{W_0}, \quad
T_{\mathrm{eff}} = T \cdot \frac{N}{B}.
\]
Hence Lemma A holds after replacing \(T\) with \(T_{\mathrm{eff}}\) in `avg_grad = -\Delta/(\eta T_{\mathrm{eff}})` (`effective_gradient_steps` in `code/benchmark_round12.py`).

**Proof.** With \(W_0 = 0\), \(\pi_{i,c} = p_c\) for all \(i\) during that step. For minibatch \(b\) with index set \(\mathcal{I}_b\), \(|\mathcal{I}_b| = B\),
\[
\frac{\partial \mathcal{L}_b}{\partial b_c}
= \sum_{i \in \mathcal{I}_b} \frac{\pi_{i,c} - \mathbf{1}[y_i=c]}{B}
= \frac{1}{B}\sum_{i \in \mathcal{I}_b} (p_c - \mathbf{1}[y_i=c]).
\]
Summing over \(M\) disjoint minibatches covering all samples (shuffle irrelevant to the sum at fixed \(W_0\)):
\[
\sum_{b=1}^{M} \frac{\partial \mathcal{L}_b}{\partial b_c}
= \frac{1}{B}\sum_{i=1}^{N} (p_c - \mathbf{1}[y_i=c])
= \frac{1}{B}\bigl(N p_c - n_c\bigr).
\]
Compare full-batch one step (`benchmark_round09.py`): \(\partial \mathcal{L}/\partial b_c = (1/N)(N p_c - n_c)\) after the \(1/N\) normalization, so the terminal bias increment from one full-batch step scales as \((N p_c - n_c)/N\), while the minibatch sum scales as \((N p_c - n_c)/B\). Ratio \(= N/B\). The weight block is identical with \(x_i\) factored out. Accumulating \(T\) local steps gives \(T_{\mathrm{eff}} = T(N/B)\). ∎

**Remark (Round 11 algebra error).** The incorrect identity \((1/B)\sum_b (B p - n_c^{(b)}) = (M/B)(N p - n_c)\) double-counts \(M\); the correct sum **already partitions** all \(N\) samples once, yielding \((N/B)(N p - n_c)\) only when expressed against a **mis-scaled** full-batch denominator — the lemma above states the comparison to full-batch normalization explicitly.

---

## Lemma MB-B (Within-step drift bound — Round 12)

**Setup:** One local step with \(M\) minibatches; weights updated as \(W^{(m+1)} = W^{(m)} - \eta G^{(m)}\) where \(G^{(m)}\) is the minibatch weight gradient at \(W^{(m)}\). Let \(W_0\) be the step start and \(W_M\) the step end.

**Claim (first-order).** Under \(\|G^{(m)}\| \le G_{\max}\) for all \(m\),
\[
\|W_M - W_0\|_F \le \eta M G_{\max} = \eta \frac{N}{B} G_{\max}.
\]
For softmax linear head, the bias-gradient error from replacing frozen \(W_0\) with terminal \(W_M\) within the step is bounded by
\[
\Bigl\|\nabla_b \mathcal{L}(W_M) - \nabla_b \mathcal{L}(W_0)\Bigr\|_2
\le L_b\, \|W_M - W_0\|_F,
\]
with \(L_b \le 2\sqrt{C}\,\|X\|_2\) in the simulator (Lipschitz constant of \(\pi(W^\top x)\) w.r.t. \(W\) at fixed \(x\)).

**Empirical bound (code).** `within_step_weight_drift` in `code/benchmark_round12.py` measures \(\|W_M - W_0\|_F\) per local step on `minibatch_sgd`; mean drift \(\approx 0.112\) (std \(\approx 0.004\)) with \(T_{\mathrm{eff}} = 18\) — see `lemma_mb_b_empirical` in `artifacts/round12_metrics.json`. The \(\sim 0.01\) prototype MSE floor vs exact tier \(1.5\times 10^{-4}\) is consistent with first-order drift perturbation but **not** tightly predicted by \(\|W_M-W_0\|_F\) alone.

**Drift-corrected second pass (`tango_mb_drift2`, Round 12 only).** Ad hoc inflation of \(T_{\mathrm{eff}}\); **does not** beat TANGO-MB on primary (`tango_mb_drift2` mean **0.025** vs **0.011**). Demoted to secondary ablation; not used in Round 13 primary estimators.

---

## Lemma MB-Iter (One Jacobian step for \(W_0 \neq 0\) — Round 12)

**Assumptions:** Known \(W_0\) (public init or attacker estimate); minibatch training as above; first-order expansion of \(\pi_{i,c}(W)\) around \(W_0\):
\[
\pi_{i,c}(W) \approx p^{(r)}_c + (x_i^\top \otimes e_c^\top)\,\mathrm{vec}(W - W_0),
\]
with round-\(r\) bias \(b^{(r)}\) absorbed into \(p^{(r)} = \mathrm{softmax}(b^{(r)})\) at \(W_0 = 0\) head init.

**Procedure (`tango_mb_iter` in `code/benchmark_round12.py`):**
1. Apply Lemma MB-A scaling; solve for \((S, S_c, n_c)\) at \(W_0\).
2. Form \(\hat W = W_0 - \eta T_{\mathrm{eff}}\,\widehat{\nabla_W \mathcal{L}}\) from recovered weight moments.
3. Recompute class-conditional mean probabilities \(\tilde p^{(r)}_c\) at \(\hat W\) and bias-correct \(\bar g^{(r)}\) by \((\tilde p^{(r)}_c - p^{(r)}_c)\); re-solve counts and prototypes.

This is one Newton/Jacobian correction, not full iterative FL inversion.

**Empirical (Round 11→12):** `minibatch_nonzero_init` prototype MSE improves vs single-pass TANGO-MB when `init_weight_scale > 0`.

---

## Lemma MB-JOINT (Round 13–14 — joint bias moments; negative result)

**TANGO-JOINT (`tango_joint` in `code/benchmark_round14.py`):** After \(T_{\mathrm{eff}}\) scaling, joint ridge LS on stacked bias and weight moments with **uniform** round weights (\(w_r = 1/R\)).

**TANGO-DOPT (`tango_dopt`):** Same stacked system with **D-opt** round weights \(w_r \propto 1/(\|p^{(r)}-\mathbf{1}/C\|+\epsilon)\), renormalized. Round 13 incorrectly used D-opt weights for both; Round 14 separates them.

**Empirical (primary `minibatch_sgd`, Round 14):** Both JOINT and DOPT prototype MSE \(\gg\) TANGO-MB (**~0.01**); neither beats sequential `estimate_counts_mb`. D-opt weighting does **not** rescue joint inversion. Aggressive probe rounds violate a shared-\(n_c\) linear model; stacking bias equations without hard exclusion harms counts.

**Coupled / trajectory variants:** `tango_coupled` (3 fixed-point bias Jacobian steps) mean **0.272**; `tango_trajectory_midpoint` mean **0.272** — same failure mode. Primary estimator remains **TANGO-MB**.

**Honest scaling ablation:** `passive_mb_scale_only` (R11: scale + `tango_estimate_sums`) mean **0.151** on primary — restores fair active-vs-scaling comparison (\(\approx 14\times\) active gain, not \(\approx 26\times\) from R12 coupled `passive_mb`).

---

## Scope boundaries (not covered by Theorems 1–2)

| Violation | Effect |
|---|---|
| Minibatch SGD (vanilla \(T\)) | Lemma A scaling wrong (Round 09/10: prototype MSE \(\gg 1\)). **Corrected** by Lemma MB-A / TANGO-MB (Rounds 11–12). |
| Nonzero unknown \(W_0\) | Initial \(\pi_{i,c}\) depends on \(x_i\); weight system nonlinear in aggregates. |
| Nonlinear representation drift | Hidden features change during local training; frozen-feature experiments test partial mitigation (Round 10). |
| Passive single-round probing | Rank deficiency; passive baseline in Round 09. |

These map to `theorem_scope.approximate_empirical` scenarios in metrics JSON.

---

## Proof–implementation correspondence

| Proof object | Code reference |
|---|---|
| Lemma A equations | `estimate_counts`, `tango_estimate_sums` in `code/benchmark_round09.py` |
| Lemma MB-A scaling | `effective_gradient_steps`, `tango_mb_estimate_sums` in `code/benchmark_round12.py` |
| Lemma MB-B drift | `within_step_weight_drift` in `code/benchmark_round12.py` |
| Lemma MB-Iter | `tango_mb_iter_estimate_sums` in `code/benchmark_round12.py` |
| MB count (bias LS) | `estimate_counts_mb` in `code/benchmark_round12.py` |
| TANGO-JOINT | `joint_mb_moment_invert`, `tango_joint_estimate_sums` in `code/benchmark_round13.py` |
| TANGO-COUPLED | `tango_coupled_estimate_sums` in `code/benchmark_round13.py` |
| Honest scaling ablation | `passive_mb_scale_only_estimate_sums` in `code/benchmark_round13.py` |
| Theorem 1 linear solve | per-coordinate `lstsq` in `tango_estimate_sums` |
| Theorem 2 floor | `within_class_variance`, `individual_mse_from_prototypes` |
| Probe full-rank | `design_condition_number` |
