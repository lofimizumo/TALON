# Proposition T1 (epoch-terminal impossibility under LASA linearity)

## Setting

- Dataset snapshots \(S = [s_1,\ldots,s_N]^\top \in \mathbb{R}^{N\times d}\), \(N\) fixed.
- Each federated round (epoch) \(e = 1,\ldots,E\) publishes **one** terminal gradient
  \[
  c^{(e)} = A^{(e)} \bar{s}, \qquad
  \bar{s} = \frac{1}{N}\sum_{i=1}^N s_i,
  \]
  where \(A^{(e)} \in \mathbb{R}^{D\times d}\) is the LASA / surrogate coefficient matrix for that probe (SurrogateQFL draw).
- **No** within-epoch minibatch rows \(\{g^{(e,k)}\}\), **no** per-client rows unless stated (that is tier T1b, not T1).
- Optional i.i.d. noise \(\eta^{(e)}\) with \(\mathbb{E}[\eta^{(e)}]=0\): \(c^{(e)} = A^{(e)}\bar{s} + \eta^{(e)}\).

This is exactly the simulator contract in `benchmark_round03.py` when `terminal_batch_grads = [[c]]` per epoch.

## Observation map

Stacking terminals defines \(\mathcal{A}: \mathbb{R}^{N\times d} \to \mathbb{R}^{Ed}\):

\[
\mathcal{A}(S) = \big[ (A^{(1)}\bar{s})^\top, \ldots, (A^{(E)}\bar{s})^\top \big]^\top,
\qquad \bar{s} = \tfrac{1}{N} S^\top \mathbf{1}.
\]

Only \(\bar{s}\) appears inside every block. Write \(J = \tfrac{1}{N}\mathbf{1}\mathbf{1}^\top\) (mean operator on rows). Then \(\bar{s} = J S\) and

\[
\mathcal{A}(S) = \mathcal{B}(\bar{s}),
\qquad
\mathcal{B}(u) = \big[ (A^{(1)} u)^\top, \ldots, (A^{(E)} u)^\top \big]^\top.
\]

Thus \(\mathcal{A}(S)\) depends on \(S\) **only through** \(\bar{s}\).

## Proposition (identifiability of individuals)

**Claim.** Fix probe matrices \(\{A^{(e)}\}_{e=1}^E\) and noise law. For any \(S, S'\) with the same dataset mean \(\bar{s}\),

\[
\mathcal{A}(S) \stackrel{d}{=} \mathcal{A}(S')
\quad\Longrightarrow\quad
\{s_i - \bar{s}\}_{i=1}^N \text{ is not identifiable from terminals alone.}
\]

Equivalently: every individual deviation \(\delta_i = s_i - \bar{s}\) lies in the **joint nullspace** of the terminal observation map modulo mean.

### Proof

1. **Reduction to mean.** If \(\bar{s}(S) = \bar{s}(S')\), then \(\mathcal{A}(S) = \mathcal{A}(S')\) identically (noise-free). With noise, distributions match because each \(c^{(e)}\) depends only on \(\bar{s}\).

2. **Explicit non-uniqueness.** Choose any non-zero \(\Delta \in \mathbb{R}^{N\times d}\) with \(\Delta^\top \mathbf{1} = 0\) (row deviations sum to zero). Set \(S' = S + \Delta\). Then \(\bar{s}(S') = \bar{s}(S)\) and \(\mathcal{A}(S') = \mathcal{A}(S)\), but \(S' \neq S\) unless \(\Delta = 0\).

3. **Linear-algebra form (noiseless).** Let \(\mathrm{vec}(S)\) be column-stacked snapshots. Each block satisfies
   \[
   c^{(e)} = (J \otimes A^{(e)}) \,\mathrm{vec}(S).
   \]
   The nullspace of \(J \otimes A^{(e)}\) contains all \(\mathrm{vec}(\Delta)\) with \((\mathbf{1}^\top \otimes I_d)\mathrm{vec}(\Delta)=0\), i.e. \(\Delta^\top \mathbf{1} = 0\). Stacking \(e\) does not couple \(\delta_i\) across blocks because every block uses the **same** \(J\).

4. **Graph / smoothness priors without extra measurements.** A fixed Laplacian prior \(p(S) \propto \exp(-\tfrac{\lambda}{2}\mathrm{tr}(S^\top L S))\) does **not** add new linear constraints on \(S\) beyond \(c^{(e)}\); it only reshapes a Bayesian posterior. Under a **pure** prior with no additional rows, MAP estimates can spread rows (Round 02 Fiedler template) but cannot identify \(\delta_i\) from data likelihood because the likelihood depends only on \(\bar{s}\). Hence Hungarian snapshot MSE remains at passive scale (\(\approx 1\) on normalized synthetic data).

5. **What would break impossibility (outside T1).**
   - **T1b:** per-client terminals \(g^{(e,i)} = A^{(e)} s_i\) (one row per sample per epoch).
   - **T1p:** minibatch rows \(g^{(e,k)} = A^{(e)} \bar{s}^{(e,k)}\) with \(K>1\) distinct batch means per epoch (partial or full).
   - **T2:** full SHARD intermediate trajectory (80 rows in our benchmark).

\(\square\)

## Corollary (budget-normalized headline)

Comparing **T1** (10 terminal rows) to **SHARD** (80 intermediate rows) at ratio \(\mathrm{MSE}_{\mathrm{T1}} / \mathrm{MSE}_{\mathrm{SHARD}} \approx 2.04\) is the expected order: T1 is not a under-powered SHARD at the same budget; it is a **different observation class** that identifies at most \(\bar{s}\).

**Acceptance path for this artifact:** Treat Proposition T1 as the scientific outcome for tier T1; report **T1b** / **T1p-honest** / **budget-80** matches separately with explicit row counts (see `artifacts/round03_metrics.json` → `headline_by_tier`).

## Assumptions (explicit)

| Assumption | Role |
|------------|------|
| LASA linearity \(g = A s\) per batch/client | Used in reduction \(\bar{s}^{(e,k)} = \text{batch mean of } s_i\) |
| One scalar terminal per epoch = mean of batch gradients | Defines T1 |
| Full-rank \(A^{(e)}\) on \(\bar{s}\) for Level-1 | Identifies \(\bar{s}\), not \(s_i\) |
| Known sample index graph for priors | Oracle side information; not new gradients |

## Simulator reference

Epoch terminals are implemented as `np.mean(grads_e, axis=0)` per epoch in `code/benchmark_round03.py`; SHARD oracle uses all `batch_gradients` rows (tier T2).
