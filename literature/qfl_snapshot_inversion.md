# QFL / VQC snapshot inversion — literature notes

## SHARD (this repo)

- **Stages 1–2:** Linear structure of LASA-VQC gradients → recover per-sample snapshots \(\mathbf{s}_i \in \mathbb{R}^{\dim\mathfrak{g}}\) from batched, reshuffled updates.
- **Stage 3 (baseline):** Per snapshot, solve \(\min_{x\in[0,1]^d}\|\cos(Wx+b)-s\|_2^2\) via hundreds of parallel **Adam** restarts + **L-BFGS** polish (`level3_invert` in `shard_sim/attacker.py`).

## Analytic / algebraic inversion

| Work | Reference | Idea | Gap vs RFF encoding |
|------|-----------|------|---------------------|
| Heredge et al. | arXiv:2502.06593 (privacy analysis line) | Polynomial / Gröbner inversion for **separable** encodings, bounded DLA | Random Fourier \( \cos(Wx+b)\) is **not** separable; DLA large |
| Mitarai et al. | arXiv:1806.06846 | Quantum circuit learning | No batch snapshot pipeline |

Paper appendix (`theoretical_analysis.tex`) notes Heredge-style algebra is infeasible for dense random \(W_{\text{enc}}\); SHARD uses generic numerical optimization.

## Proposed line (LAPIN)

Exploit **explicit Jacobian** of \(\cos(Wx+b)\):

\[
J(x) = \mathrm{diag}(-\sin(Wx+b))\, W .
\]

Use **projected Gauss–Newton** with encoding-aware \(\mathrm{acos}\) least-squares seeds—not parallel first-order Adam clouds. This is closer to **nonlinear least-squares** solvers used in scientific inversion than to DLG-style gradient matching.

## Open gaps (Round 2+)

- Multi-snapshot joint term \(\sum_i \|\cos(Wx_i+b)-s_i\|^2\) with shared \(W\) structure.
- Tikhonov prior on \(x\) or in latent \(z=Wx+b\) when \(d > \dim\mathfrak{g}\) (underdetermined).
- Warm-start Stage 3 from Stage 2 iterate geometry (cluster structure in snapshot space).
