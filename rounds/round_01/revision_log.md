# Round 01 — Revision log

## Status

Initial proposal (no prior supervisor review).

## Changes this round

1. **Framed Stage 3 replacement** as snapshot-space nonlinear least squares, not gradient inversion.
2. **Proposed LAPIN** — multi-start projected Gauss–Newton with analytic \(\cos\) Jacobian + single L-BFGS polish (no Adam).
3. **Explored and discarded** (same round, implementation iterations):
   - Latent ADMM with decoupled 1D \(z\)-updates (slow, weak fit).
   - Per-coordinate alternating projection on \(\mathrm{acos}\) branches (inconsistent latent \(z\), divergent residuals).
4. **Implemented** `lapin_invert.py` and `benchmark_round01.py` under run folder; imports `shard_sim` read-only via `sys.path`.
5. **Ran** MNIST \(N{=}12\), \(14{\times}14\), `dim_g=160` benchmark; saved JSON + plots.

## Honest outcome

LAPIN does **not** beat SHARD L3 on Round 01 metrics. Documented for supervisor audit; Round 2 will pivot optimization budget toward GN+L-BFGS parity and joint inversion.

## Next round (planned)

- Address supervisor feedback if critical.
- Hybrid: LAPIN best-start → full-iter L-BFGS only (remove Adam entirely from pipeline).
- Add `dim_g < d` compressive regime from paper figures.
- Log seed sensitivity and snapshot residual vs input MSE correlation.
