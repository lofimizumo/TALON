# QFL terminal snapshot recovery — literature & observation taxonomy

## Observation tiers (this run)

| Tier | Server sees | Individual \(s_i\) identifiable? |
|------|-------------|----------------------------------|
| T0 | Passive final model only | No (TALON class moments only under probes) |
| T1 | Terminal epoch summary \(c^{(e)} = A^{(e)}\bar{s}\) per probe round | **Mean \(\bar{s}\) only** (H1) |
| T2 | All intermediate \(g^{(e,k)}\) (SHARD Stage 2) | Yes under rank + assignment (vendored attacker) |
| T3 | T2 + snapshot inversion to \(x\) | Input recovery (SHARD L3) |

Round 01 implements **T1** attacks vs **T2** oracle.

## LASA linear structure (SurrogateQFL)

\[
g^{(e,k)} = A^{(e)} \bar{s}^{(e,k)}, \quad
c^{(e)} = \frac{1}{K}\sum_k g^{(e,k)} = A^{(e)} \bar{s}.
\]

Stacking \(c^{(e)}\) across epochs constrains the **same** \(\bar{s}\) under noiseless linearity; it does not constrain individual deviations \(s_i - \bar{s}\) without batch-mean rows \(\bar{s}^{(e,k)}\) with \(K>1\) batches per epoch.

## Parent TALON bridge (negative result)

- Terminal active probes identify **class aggregate** geometry in a linear head, not per-client feature vectors.
- **Implication:** Mechanisms that only stack terminal summaries need **extra structure** (graph, assignment, low-rank manifold, or per-client terminals) to lift to \(S\).

## Method families (candidates)

| Family | Needs beyond T1? | Round 01 |
|--------|------------------|----------|
| Mean broadcast | No | Baseline |
| Graph MAP (GARD-style) | Oracle graph on indices | `graph_term_terminal` |
| Graph low-rank subspace | Rank + graph | `graph_rank_terminal` |
| SHARD Stage 2 | All batch rows + assignment | Oracle |
| JASPER / soft assignment | Partial rows or soft incidence | Round 03+ |
| B=1 client terminals | Per-sample \(g\) each round | Round 02+ |

## Citations

- SHARD pipeline: `vendor/shard_sim/attacker.py`, `surrogate_model.py`
- Parent rounds: `/workspace/rounds/round_05`, `round_06` (GARD, JASPER assignment barriers)
- Snapshot inversion survey: `/workspace/literature/qfl_snapshot_inversion.md`
- Bridge doc: `literature/parent_talon_bridge.md`

## Open gaps

- Identifiability proof for T1 + graph prior (when does \(\lambda\) break symmetry enough?).
- Coupled **terminal head updates + LASA snapshots** in real QFL stacks.
- Honest graph from metadata vs co-occurrence (Round 05 negative).
