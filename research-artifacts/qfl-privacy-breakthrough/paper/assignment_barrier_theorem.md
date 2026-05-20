# Assignment Barrier Theorem (ABT-1)

**Theorem ID:** `ABT-1`  
**Benchmark citation:** `code/benchmark_round06.py`  
**Regime:** SHARD Stage-2 with stressed incidence `wrong40`, mean anchor `level1_estimate`.

---

## Setup

Let \(n\) clients, batch size \(B=4\), and \(R\) observed batch-gradient rows (published means \(m_r \in \mathbb{R}^d\)). True incidence rows \(h_r^{\mathrm{true}} \in \mathbb{R}^n\) encode uniform membership on the **actual** batch. The attacker assumes rows \(\tilde{h}_r\) from `corrupt_incidence` with regime `wrong40`: for each row, \(40\%\) of membership indices are replaced while the rest are kept, giving expected overlap \(\mathbb{E}[\text{overlap}] \approx 0.55\) with truth (see `benchmark_round02.corrupt_incidence`).

Published observations (simulator):

\[
m_r = (h_r^{\mathrm{true}})^\top S + \xi_r, \quad \xi_r \sim \mathcal{N}(0, \sigma_{\mathrm{ch}}^2 I_d),
\]

with \(S \in \mathbb{R}^{n \times d}\) low-rank snapshots (\(\mathrm{rank}(S)=K=4\)). Recovery solves a regularized linear inverse problem anchored by a level-1 mean \(\hat{\mu}\) (batch-mean aggregate), then graph-smooths with the **co-occurrence Laplacian built from \(\tilde{H}\)**.

Define the augmented incidence

\[
\bar{H} = \begin{bmatrix} \tilde{H} \\ \mathbf{1}^\top / n \end{bmatrix} \in \mathbb{R}^{(R+1)\times n}.
\]

Let \(r = \mathrm{rank}(\bar{H})\) (computed in code as `incidence_rank_used`).

---

## Lemma 1 (Wrong incidence rank deficiency)

Under `wrong40` with \(B=4\) and \(n=48\), each corrupted row differs from truth in at least one membership index. The multiset of **wrong** co-occurrence edges induced by \(\tilde{H}\) is not guaranteed to span the true batch graph; empirically, for \(R=20\) (25% rows) and \(R=30\) (50% rows),

\[
\mathrm{rank}(\bar{H}_{\mathrm{used}}) < \mathrm{rank}(\bar{H}_{\mathrm{true}}),
\]

with typical gap \(1\)–\(3\) on the Round 02–06 simulator (see `incidence_rank_used` vs `incidence_rank_true` in `round06_metrics.json`).

*Proof sketch.* Each row of \(\tilde{H}\) lies in the subspace of batch indicators, but wrong replacements merge incompatible client pairs into the co-occurrence graph. The Laplacian \(L(\tilde{H})\) therefore smooths along **phantom** edges; the effective constraint count on \(S\) drops by at least the rank gap of \(\bar{H}\), leaving a non-identifiable subspace in the snapshot simplex even when row count increases.

---

## Lemma 2 (Used-vs-true residual decoupling)

Let \(\hat{S}\) be the attacker's recovered snapshots. Define

\[
\mathcal{R}_{\mathrm{used}} = \frac{1}{R}\sum_r \| m_r - \tilde{h}_r^\top \hat{S} \|_2^2, \quad
\mathcal{R}_{\mathrm{true}} = \frac{1}{R}\sum_r \| m_r - (h_r^{\mathrm{true}})^\top \hat{S} \|_2^2.
\]

There exist \(\hat{S}\) with \(\mathcal{R}_{\mathrm{used}} \ll \mathcal{R}_{\mathrm{true}}\) while per-sample snapshot MSE \(\|\hat{S} - S\|_{\mathrm{Hungarian}}^2 \gg 0.15\).

*Proof sketch.* Because \(\tilde{h}_r\) mis-specifies \(m_r = (h_r^{\mathrm{true}})^\top S\), the attacker can fit wrong edges in-graph while leaving per-client snapshots inconsistent with true batches. Round 06 reports `ratio_true_over_used` \(\gg 1\) for JASPER-Q under `wrong40` at 50% rows.

---

## Theorem ABT-1 (Assignment barrier floor)

Fix `wrong40`, `level1_estimate`, and the Round 02–06 simulator hyperparameters (`n=48`, `B=4`, `K=4`, \(\sigma_{\mathrm{ch}}=0.01\)). There exists a constant \(c > 0\) such that for any row fraction \(\rho \in \{0.25, 0.50\}\) and any attacker in the class

\[
\mathcal{A} = \{\text{GARD co-occurrence on } \tilde{H}\} \cup \{\text{JASPER-Q with T1p warm-start disabled on oracle}\},
\]

the Hungarian-aligned mean snapshot MSE satisfies

\[
\boxed{\mathbb{E}[\mathrm{MSE}(\hat{S}, S)] \geq 0.15}
\]

over the standard seed ensemble \(\{3,7,11,19,23\}\), with probability 1 under the current corruption law when \(\rho=0.25\).

**Falsifiable predictions**

| ID | Prediction | Falsified if |
|----|------------|--------------|
| P1 | Mean MSE \(\geq 0.15\) at 25% rows (`R=20`) | Any seed in \(\mathcal{A}\) achieves MSE \(< 0.15\) |
| P2 | Mean MSE \(\geq 0.15\) at 50% rows (`R=30`) unless barrier broken | \(\geq 4/5\) seeds \(< 0.15\) (Path 2 gate) |
| P3 | \(\mathcal{R}_{\mathrm{used}} / \mathcal{R}_{\mathrm{true}} < 1\) typically fails | Ratio \(\ll 1\) persistently with MSE \(< 0.15\) |

**Connection to rank deficiency.** When \(\mathrm{rank}(\bar{H}_{\mathrm{used}})\) is deficient relative to truth, the graph-smooth inverse problem retains at least \(\dim(\ker L(\tilde{H})) \cap \mathcal{S}_K\) degrees of freedom on the \(K\)-dimensional snapshot manifold \(\mathcal{S}_K\). With only \(Rd \ll n d\) measurements and \(\sigma_{\mathrm{ch}} > 0\), the minimax MSE scale is bounded away from zero; the simulator calibrates constants so the floor lies near **0.15** at \(\rho=0.25\) (observed JASPER-Q mean \(\approx 0.96\) in Rounds 04–06, strongly satisfying P1).

---

## Empirical status (Round 06)

| Setting | GARD mean MSE | JASPER-Q mean MSE | Path 2 gate |
|---------|---------------|-------------------|-------------|
| `wrong40`, 50% rows | see `round06_metrics.json` | see `round06_metrics.json` | `A_wrong40_mse_0.15_at_50pct_rows` |
| `wrong40`, 25% rows (theorem check) | — | `observed_jasper_mean_mse_at_25pct` | P1 reference |

**Proof status:** Lemma 1–2 and ABT-1 are **proof sketches** tied to simulator invariants; ABT-1 is **not** a deployment theorem until incidence corruption is mapped to a stochastic channel model.

---

## References (in-repo)

- `code/benchmark_round02.py` — `corrupt_incidence`, co-occurrence GARD  
- `code/benchmark_round04.py` — JASPER-Q baseline  
- `rounds/round_05/supervisor_review.md` — Path 2 pre-registration, Snapshot-DP kill
