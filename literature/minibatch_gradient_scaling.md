# Minibatch terminal-gradient scaling (Round 11 note)

## Problem

Federated learning clients commonly train with **minibatch SGD** (batch size \(B\), \(T\) local steps, shuffled passes). Round 09–10 showed vanilla TANGO fails (`prototype_mse ≈ 6.59` on `minibatch_sgd`) because it assumes terminal deltas equal \(\eta T\) times a **single full-batch** gradient.

## Mechanism (w0 linearization)

With \(W_0=0\), softmax probabilities depend only on server bias \(b\), not on \(x_i\) within a step. PyTorch-style updates use per-minibatch normalization \(1/B\). One full client pass per local step applies \(M = N/B\) minibatch updates; at fixed \(W_0\) their sum reproduces the full-batch gradient direction scaled by \(N/B\).

**Effective steps:** \(T_{\mathrm{eff}} = T \cdot (N/B)\).

## Method

- **TANGO-MB:** scale \(\Delta \leftarrow \Delta \cdot (T / T_{\mathrm{eff}})\) then apply Round-09 inversion.
- **STORM:** same \(T_{\mathrm{eff}}\) in stacked moment equations.

Requires **public** \(N, B, T, \eta\) (standard FL hyperparameter broadcast). Does **not** require batch order or intermediate gradients.

## Empirical (synthetic simulator)

| Estimator | `minibatch_sgd` prototype MSE |
|---|---:|
| TANGO vanilla | 6.5877 |
| Passive | 0.7532 |
| **TANGO-MB** | **0.0110** |

## Related work (conceptual, not citation-complete)

- Stochastic gradient expectation in FL analysis (e.g. FedAvg convergence proofs) uses similar \(B,N\) scaling in expectation, not per-batch leakage.
- Gradient inversion (Zhu et al., Deep Leakage) typically assumes **per-step** gradients; TALON remains terminal-only with corrected **aggregate** scaling.

## Open gaps

- Formal bound on error from weight drift within a local step.
- Nonzero \(W_0\) without linearization.
- Secure aggregation / DP noise on terminal deltas under MB tier.
