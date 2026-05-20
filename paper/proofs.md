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

## Scope boundaries (not covered by Theorems 1–2)

| Violation | Effect |
|---|---|
| Minibatch SGD | Batch composition enters gradients; Lemma A fails (Round 09/10: prototype MSE \(\gg 1\)). |
| Nonzero unknown \(W_0\) | Initial \(\pi_{i,c}\) depends on \(x_i\); weight system nonlinear in aggregates. |
| Nonlinear representation drift | Hidden features change during local training; frozen-feature experiments test partial mitigation (Round 10). |
| Passive single-round probing | Rank deficiency; passive baseline in Round 09. |

These map to `theorem_scope.approximate_empirical` scenarios in metrics JSON.

---

## Proof–implementation correspondence

| Proof object | Code reference |
|---|---|
| Lemma A equations | `estimate_counts`, `tango_estimate_sums` in `code/benchmark_round09.py` |
| Theorem 1 linear solve | per-coordinate `lstsq` in `tango_estimate_sums` |
| Theorem 2 floor | `within_class_variance`, `individual_mse_from_prototypes` |
| Probe full-rank | `design_condition_number` |
