# Round 01 — Researcher Proposal: Terminal-Only QFL Snapshot Recovery

## Problem & goal

In **quantum federated learning** with **LASA-VQC** linearity (SurrogateQFL), each local epoch produces batch gradients \(g^{(e,k)} = A^{(e)} \bar{s}^{(e,k)}\) where \(\bar{s}^{(e,k)}\) is the batch-mean snapshot. **SHARD Stage 2** recovers the individual snapshot matrix \(S \in \mathbb{R}^{N \times \dim\mathfrak{g}}\) but requires **all intermediate minibatch gradient rows** plus alternating assignment.

**Goal (this run):** Discover whether **weaker observations**—terminal epoch summaries \(c^{(e)} = \frac{1}{K}\sum_k g^{(e,k)} = A^{(e)} \bar{s}\), repeated server-chosen probe rounds, and graph priors—suffice to recover **individual** snapshots without intermediate rows, approaching SHARD Stage-2 MSE.

## Hypothesis / claim

| ID | Hypothesis | Test |
|----|------------|------|
| H1 | Stacked terminal probes \(c^{(e)}\) identify **only** the dataset mean \(\bar{s}\) in the LASA linear model; individual \(s_i\) are not linearly identifiable without batch-mean rows. | Level-1 recovery vs Stage-2 oracle on same \(A^{(e)}\). |
| H2 | A **graph smoothness prior** on snapshot geometry (chain / acquisition order) can spread \(\bar{s}\) into per-sample estimates and **beat** the passive mean broadcast baseline, but not SHARD Stage-2 with full intermediates. | `graph_term_terminal`, `graph_rank_terminal` vs oracle. |
| H3 | Parent **TALON** terminal mechanisms (class-moment probes) do not transfer to individual snapshots without new structure (assignment, batch means, or audit rows). | Cite TALON Theorem-2 non-identifiability; do not re-run TANGO here. |

**Round-01 claim (testable):** Terminal-only recovery **fails** the acceptance ratio vs SHARD Stage-2; graph priors are a **conditional** nullspace filler, not a replacement for intermediate observations.

## Related work

- **SHARD** (vendored `shard_sim`): Stage 1 mean LS; Stage 2 batch-mean solve + alternating assignment (`attacker.py`).
- **Parent TALON** (`literature/parent_talon_bridge.md`): TANGO recovers **class prototypes** under terminal probes; GARD/JASPER need assignment or observed batch structure.
- **Heredge et al.** arXiv:2502.06593: algebraic inversion for separable encodings—not applicable to dense RFF \(\cos(Wx+b)\).
- **Round 05–06** (`/workspace/rounds/`): GARD helps only with oracle assignment + graph; JASPER does not fix unknown within-epoch assignment.

## QFL threat model (Round 01)

**Honest-but-curious server** in SurrogateQFL simulation:

- Observes per-epoch coefficient matrix \(A^{(e)}\) (server-chosen or public).
- Observes **terminal epoch gradient** \(c^{(e)} \in \mathbb{R}^D\) after all \(K\) minibatches (one vector per epoch).
- Does **not** observe intermediate \(g^{(e,k)}\) for \(k < K\) (terminal methods: count = 0).
- May hold **side information**: sample index order → chain graph Laplacian (same favorable prior as parent GARD oracle graph).
- Does **not** observe true batch membership, private snapshots, or inputs.

**Oracle upper bound:** SHARD `level2_disaggregate` with full \(E \times K\) batch rows (same synthetic data, same seeds).

## Method candidates (Round 01)

1. **Passive mean broadcast** — Level-1 \(\bar{s}\) replicated to all \(N\) rows (TALON-style aggregate collapse).
2. **GRAPH-TERM** — MAP: \(\min_S \|(1/N)\mathbf{1}^\top S - \bar{s}\|^2 + \lambda \mathrm{Tr}(S^\top L S)\) with chain \(L\).
3. **GRAPH-RANK-TERM** — Same anchor, but \(S = UC\) in constant + low Laplacian modes (rank 4).

**Not in Round 01 (next rounds):** partial terminal trajectories, B=1 per-client terminals, joint assignment (JASPER-TERMINAL), coupled head+snapshot inversion.

## Experiment plan

| Item | Choice |
|------|--------|
| Simulator | `SurrogateQFL` + synthetic \(N=32\), \(B=4\), \(E=10\), \(\dim\mathfrak{g}=32\), \(D=32\), noise \(\sigma=0.01\) |
| Metrics | Hungarian-aligned per-sample snapshot MSE; `observed_intermediate_batch_gradients` |
| Baselines | Passive mean; SHARD Stage-2 oracle |
| Seeds | 3, 7, 11, 19, 23 |
| Failure modes | Graph wrong/permuted; unknown assignment; rank misspecification |

**Script:** `code/benchmark_round01.py` → `artifacts/round01_metrics.json`, `logs/experiment_round01.log`.

## Changes this round

Initial round: problem framing, three terminal candidates, SHARD oracle baseline, literature bridge from parent TALON.

## Open risks

- Oracle chain graph overstates attacker capability vs permuted FL batches.
- Synthetic random inputs may not match MNIST/CIFAR snapshot geometry (Round 02+).
- Zero intermediate rows is strict; one leaked last-minibatch row per epoch would change identifiability.

## Expected outcome (pre-registration)

Terminal MSE \(\gg\) SHARD Stage-2 MSE; best terminal may edge passive baseline via graph spread but **not** within 2× of oracle.

## Round-01 results (executed)

From `artifacts/round01_metrics.json` (smooth rank-4 snapshots, \(N{=}32\), \(B{=}4\), \(E{=}10\), \(\dim\mathfrak{g}{=}32\), 5 seeds):

| Method | Mean snapshot MSE | Intermediate rows |
|--------|------------------:|--------------------:|
| SHARD Stage-2 oracle | **0.444** | 80 |
| GRAPH-TERM / GRAPH-RANK / passive | **1.000** | 0 |

- Level-1 terminal mean recovery relative error \(\approx 3.5\times 10^{-4}\) (mean is identified).
- SHARD did not converge within 50 iterations on several seeds; matching accuracy \(\approx 0\%\) (strict threshold) — oracle MSE is an **upper bound**, not saturated.
- **Headline ratio:** best terminal / SHARD \(\approx 2.25\times\) (terminal **worse**).
- GRAPH-TERM does not materially beat passive broadcast at this budget (spread along oracle chain does not align with reshuffled batch geometry).

**Honest verdict:** Terminal-only observations identify \(\bar{s}\) but not individuals; acceptance target (within 2× of SHARD Stage-2) **not met**. Next rounds: B=1 client terminals, partial trajectory rows, assignment-aware probes.
