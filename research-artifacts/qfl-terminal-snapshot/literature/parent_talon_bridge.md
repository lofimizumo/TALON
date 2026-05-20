# Bridge from parent TALON run (`/workspace`)

## User goal (this run)

SHARD in QFL exploits **LASA linearity**: batch gradients are \(g^{(e,k)} = A^{(e)} \bar{s}^{(e,k)}\) where \(\bar{s}^{(e,k)}\) is the batch-mean snapshot. Stage 2 recovers **individual** snapshots \(s_i\) but needs **intermediate batch-gradient rows** and (typically) assignment/incidence.

**Target:** Individual snapshot recovery **without** intermediate minibatch gradients—terminal-only or weaker observations compatible with QFL deployment.

## What parent TALON established (keep, do not re-claim as this run's result)

- **TANGO / TANGO-MB:** Terminal active probes recover **class aggregates / prototypes**, not individuals (Theorem 2 non-identifiability).
- **GARD:** Can reduce observed rows **only with oracle assignment + aligned graph**.
- **JASPER:** Unknown assignment → soft assignment fits residual but not true membership.
- **SHARD L3:** Individual **input** recovery needs Tier-3 intermediate observations.

## Implication for QFL terminal snapshot run

Terminal aggregate moments alone are **insufficient** for individual snapshots unless new side information appears:

- Extra probes / rounds with identifiable structure beyond class sums
- Partial terminal trajectory across local steps
- Graph/Laplacian prior on snapshot manifold (LASA smoothness in \(s\)-space)
- Coupled inversion: terminal head updates + encoding linearity to lift aggregates to individuals

This run must **attempt** individual \(S\) recovery and **benchmark against** `ShardAttacker` Stage 2 with full intermediate access as oracle upper bound.
