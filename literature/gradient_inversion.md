# Gradient inversion (classical FL) — literature notes

## Problem

Given model parameters \(\theta\) and observed gradient \(g = \nabla_\theta \mathcal{L}(\theta; \mathcal{B})\) for a mini-batch \(\mathcal{B}\), recover inputs \(\{x_i\}\) (and sometimes labels). Non-convex, underdetermined when \(B>1\) due to permutation invariance.

## Key methods (differentiation from our setting)

| Work | ID | Mechanism | Batch \(B>1\) | Our snapshot setting |
|------|-----|-----------|---------------|----------------------|
| DLG | Zhu et al., arXiv:1906.08935 | Minimize \(\|g_{\text{dummy}}-g_{\text{true}}\|_2\) w.r.t. dummy \(x\) | Poor | Optimizes **gradients**, not \(\cos(Wx+b)\) snapshots |
| iDLG | Zhao et al., arXiv:2001.02610 | DLG + label recovery | Poor | Same |
| Inverting Gradients | Geiping et al., arXiv:2003.14053 | Cosine similarity + TV prior on images | Limited | Image priors; not encoding-specific |
| GradInversion | Yin et al., arXiv:2004.13139 | Batch-wise shared dummy + group regularization | Scaled batch | Still classical CNN gradients |
| Learning to Invert | Wu et al., arXiv:2307.09076 | Learned attack network | Adaptive | Requires training; not structure of \(\cos\) encoding |

## Takeaway for SHARD Stage 3

Classical inversion targets **end-to-end gradients** through deep networks. SHARD Stage 3 instead inverts a **fixed, known** nonlinear encoder \(e_{\text{snap}}(x)=\cos(W_{\text{enc}}x+b_{\text{enc}})\) after Stages 1–2 already recovered snapshot vectors. Methods that rely on CNN smoothness or label tricks do not directly apply; the bottleneck is **non-convex cosine least squares** with box constraints, not gradient matching in parameter space.
