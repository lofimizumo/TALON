# Round 10 — Manuscript Closure: Proofs, Decoder Bridge, Frozen Backbone

## Problem & Goal

Round 09 closed the evaluation checklist at **ACCEPT_WITH_MINOR** (paper readiness 4/5). Round 10 targets full **ACCEPT** by closing the remaining **Major** manuscript gaps from `rounds/round_09/supervisor_review.md`:

1. Formal proof appendix matching `theorem_scope.exact`
2. Synthetic decoder / semantic probe linking hidden prototypes to decodable structure
3. Frozen nonlinear backbone experiment (representation fixed during local training)
4. Draft manuscript sections with limits leading on minibatch failure and separated count vs prototype wins

## Hypothesis / Claim

**Supported (unchanged core):** Active terminal probing identifies class-level aggregate moments and hidden prototypes under the exact linear full-batch model, with formal proofs; individuals remain non-identifiable.

**New Round-10 bridges:**

1. When TANGO recovers hidden prototypes accurately, a fixed linear decoder maps them to **synthetic pixel class means** with matching error (hidden MSE = pixel MSE in linear decoder case).
2. When features are $\phi(x)$ from a **frozen** random MLP, Tier-1 TANGO remains strong on prototypes (MSE $\ll 1$) though slightly worse than the linear hidden baseline—supporting “fixed features + linear head” without claiming representation drift during training.
3. Minibatch failure propagates to semantic metrics: pixel class-mean MSE degrades with hidden prototype failure.

**Not claimed:** Raw-image inversion, individual recovery, or SHARD-equivalent attacks under terminal-only observation.

## Related Work

- SHARD (vendored `vendor/shard_sim`): intermediate-gradient individual reconstruction.
- Gradient inversion literature (`literature/gradient_inversion.md`): motivates weaker observation tiers.
- Round 09 baselines: passive multi-round, public prior, oracle aggregate ceiling.

## Method

### Formal proofs (`paper/proofs.md`)

- **Lemma A:** First-order full-batch terminal gradient identities.
- **Lemma B:** Count identifiability from bias deltas.
- **Theorem 1:** Aggregate identifiability (coordinate-wise linear systems + full-rank probes).
- **Theorem 2:** Individual non-identifiability via within-class zero-sum perturbations.
- Assumption list matches `theorem_scope.exact` in metrics JSON.

### Decoder semantic probe (`code/benchmark_round10.py`)

- Synthetic pixels $y_i = x_i D$ with fixed orthonormal $D \in \mathbb{R}^{d \times 28}$.
- TANGO recovers $\hat\mu_c$; evaluate hidden MSE, pixel class-mean MSE $\|D\hat\mu_c - D\mu_c\|^2$, and correlations.
- Optional oracle: least-squares $D_{\text{fit}}$ from all $(x,y)$ pairs, decode $\hat\mu_c$ through $D_{\text{fit}}$ (upper bound on decodable structure given feature access for training decoder only).

### Frozen MLP backbone

- $\phi(x)$: two-layer $\tanh$ MLP with frozen random weights; only linear head trains on $\phi(x)$.
- Tests whether Tier-1 story survives **nonlinear but fixed** representations.

### Manuscript (`paper/draft.md`, `paper/method.tex`)

- Limits section **leads** with minibatch table (TANGO MSE 6.59 vs passive 0.75).
- Prototype vs count wins reported separately throughout.

## Experiment Plan

| Experiment | Metrics | Failure mode |
|---|---|---|
| decoder_balanced | hidden/pixel MSE, corr, count MAE | Should match Round 09 balanced |
| decoder_minibatch | same | Pixel MSE should explode with hidden failure |
| frozen_mlp_balanced / imbalanced | TANGO vs passive prototype MSE | Moderate MSE increase vs linear features |

Baselines: passive multi-round within frozen MLP scenario. No injected metrics.

## Changes This Round (vs Round 09)

| Deliverable | Purpose |
|---|---|
| `paper/proofs.md` | Close Major #5 (formal proofs) |
| `paper/draft.md`, `paper/method.tex` | Manuscript skeleton with limits-first |
| `code/benchmark_round10.py` | Decoder + frozen MLP experiments |
| `artifacts/round10_*`, `logs/experiment_round10.log` | Reproducible metrics |
| Updated `paper/outline.md`, `tutorial/tutorial.md` | Paper + tutorial sync |

## Round 10 Results (8 seeds)

### Decoder probe

| Scenario | Hidden proto MSE | Pixel class-mean MSE | Hidden corr | Count MAE |
|---|---:|---:|---:|---:|
| decoder_balanced | `0.000151` | `0.000151` | `1.0000` | `0.0419` |
| decoder_minibatch | `6.5877` | `6.5877` | `0.9992`* | `2.5743` |

\*High correlation under minibatch reflects **directional** alignment of wrong estimates, not usable semantic recovery; MSE is the honest metric.

Decoder balanced: LSTSQ pixel MSE from estimated prototypes `0.000151` (matches hidden).

### Frozen MLP (head-only training)

| Scenario | TANGO proto MSE | Passive proto MSE | Gain |
|---|---:|---:|---:|
| frozen_mlp_balanced | `0.00302` | `0.367` | **218x** |
| frozen_mlp_imbalanced | `0.00438` | `0.613` | **185x** |

Tier-1 survives frozen nonlinear features with small prototype error; active probing still essential.

### Manuscript limits (carried from Round 09)

| Scenario | TANGO proto MSE | Passive proto MSE |
|---|---:|---:|
| balanced_clean | `0.000151` | `0.143` |
| minibatch_sgd | `6.5877` | `0.753` |

Passive count MAE can beat TANGO on balanced data (`0.0152` vs `0.0419`); not repeated here.

## Threat Model

Unchanged: **active terminal probing** (server-chosen biases), not passive logging. Intermediate gradients not observed.

## Artifacts

- `paper/proofs.md`, `paper/method.tex`, `paper/draft.md`
- `code/benchmark_round10.py`
- `artifacts/round10_metrics.json`
- `artifacts/round10_decoder_probe.svg`, `artifacts/round10_frozen_mlp.svg`
- `logs/experiment_round10.log`
- `paper/outline.md`, `tutorial/tutorial.md`

## Open Risks

1. **Synthetic only** — no deployed FL CNN; claims must stay tiered.
2. **Decoder is linear and known for evaluation** — demonstrates structure correlation, not end-to-end image attack.
3. **Frozen $\phi$ does not model representation drift** during local training (still approximate tier).
4. **Minibatch** remains blocking for default TANGO narrative in production FL.

## Viability

Round 10 raises paper readiness toward **5/5** for the stated narrow model by supplying proofs, semantic bridge evidence, frozen-backbone support, and limits-first manuscript text. Remaining deployment gaps (secure aggregation, label noise, true nonlinear drift) are explicit future work, not hidden failures.
