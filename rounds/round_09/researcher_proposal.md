# Round 09 -- Paper-Readiness Proposal: TALON Evaluation Closure

## Thesis (unchanged, sharpened)

**TALON** (Terminal Aggregate Leakage with Observation tiers and Non-identifiability limits) remains the final paper direction. **TANGO** (Terminal Aggregate Neural Gradient Observation) is the main positive result under **active terminal probing**: a protocol-compliant server chooses classifier-bias probes, observes only terminal client deltas, and recovers class counts, class sums, dataset means, and class prototypes without intermediate minibatch gradients.

The supported claim is unchanged:

> Active terminal probes identify class-level hidden prototypes and aggregate moments, but terminal aggregate moments do not identify individual samples. SHARD-equivalent individual recovery requires stronger observations or side information.

Round 09 closes the Round-08 supervisor gaps: extended stress tests, baseline separation, count-vs-prototype reporting, theorem-scope clarification, and threat-model wording fixes.

## Threat Model (corrected wording)

Use **active terminal probing**, not passive honest-but-curious observation.

| Safe | Unsafe |
|---|---|
| Active server chooses classifier biases across repeated terminal rounds | Passive logging of ordinary training suffices |
| Weaker than intermediate-gradient observation | Standard honest-but-curious with no probe control |
| Protocol-compliant if server may set round initialization | TANGO recovers individuals from terminal-only data |

The server still needs public knowledge of architecture, learning rate, local step count, and client sample count (or a consistent estimate). This is **stronger than passive terminal logging** but **weaker than SHARD's intermediate-gradient stream**.

## Theorem Scope: Exact vs Approximate

Round 09 explicitly separates theorem tiers in the paper.

### Exact tier (Theorem 1, provable)

Assumptions:

1. Linear softmax head on fixed hidden features.
2. **Zero initial head weights** (or known fixed initialization treated as public).
3. **Full-batch** local training (one gradient over all client samples per step).
4. Terminal deltas equal first-order aggregate gradients scaled by known \(\eta, T, N\).
5. Known initial biases \(b^{(r)}\) with full-rank probe design and consistency constraint \(S=\sum_c S_c\).

Conclusion: class sums \(S_c\), counts \(n_c\), prototypes \(\mu_c=S_c/n_c\), and dataset mean \(S/N\) are identifiable coordinate-wise.

### Approximate tier (empirical, Round 09 stress)

These violate exact assumptions and are evaluated separately:

| Stress | Effect on TANGO |
|---|---|
| Minibatch SGD | **Fails** under first-order estimator (prototype MSE `6.59`, count MAE `2.57`) |
| Nonzero initial head weights | Degraded but usable (prototype MSE `0.0097`, count MAE `0.74`) |
| More local steps | Moderate degradation (prototype MSE `0.0019`) |
| 10 classes, 30-dim features | Still strong (prototype MSE `1.06e-5`, count MAE `0.029`) |
| High within-class variance | Prototype OK; individual near floor (`0.983` vs floor `0.837`) |

The paper must state: exact identifiability is a **linear-head / full-batch / first-order** statement; minibatch and multi-step training require corrected estimators or approximation bounds (future work).

### Non-identifiability (Theorem 2, unchanged)

Within-class zero-sum perturbations preserve all aggregate moments and first-order terminal observations. Individual samples remain non-identifiable from terminal aggregates alone.

## Round 09 Experiments

Implemented in `code/benchmark_round09.py`. All scenarios use **zero intermediate batch gradients**.

### New stress tests

1. **minibatch_sgd** -- 6 local steps, batch size 8, minibatch SGD.
2. **nonzero_init_head** -- initial weight scale `0.18`.
3. **large_10class_30dim** -- 10 classes, 30-dimensional hidden features, 60 samples.
4. Retained: imbalanced, more local steps, high within-class variance.

### New baselines (8 rounds each)

| Method | Role |
|---|---|
| **TANGO active** | Designed bias probes (main attack) |
| **Passive multi-round** | Repeated neutral (zero) bias -- no probe diversity |
| **Public prior** | Single-round crude dataset mean repeated for all classes |
| **Oracle aggregate** | Perfect class sums/counts upper bound (not achievable from probes) |

Oracle aggregate is **separated from TANGO**: it uses ground-truth moments to show the ceiling if aggregate recovery were perfect, not an attack the server can run.

### Count vs prototype reporting

Metrics are reported separately:

- **Count recovery:** `count_mae`, `count_relative_error`
- **Prototype recovery:** `prototype_mse`, `dataset_mean_mse`
- **Individual (negative):** `individual_mse_from_prototypes` vs within-class floor

## Numeric Results (8 seeds, 8 active/passive rounds)

### Balanced clean + baselines

| Method | Prototype MSE | Count MAE | Individual MSE |
|---|---:|---:|---:|
| TANGO active | `0.000151` | `0.0419` | `0.2667` |
| Passive multi-round | `0.143141` | `0.0152` | `0.4285` |
| Public prior | `0.412694` | `0.0`* | `0.6659` |
| Oracle aggregate | `0.0` | `0.0` | `0.2667` |

\*Public prior uses uniform count guess; count metric is not meaningful for that baseline.

Active TANGO vs passive gain: **950.6x** prototype improvement.

### Extended stress (TANGO active only)

| Scenario | Prototype MSE | Count MAE | Count rel. err. | Individual MSE | Passive proto MSE |
|---|---:|---:|---:|---:|---:|
| balanced_clean | `0.000151` | `0.0419` | `0.0052` | `0.2667` | `0.143141` |
| imbalanced_clean | `0.000382` | `0.1107` | `0.0175` | `0.2672` | `0.279253` |
| minibatch_sgd | `6.587693` | `2.5743` | `0.3218` | `6.8291` | `0.753222` |
| nonzero_init_head | `0.009720` | `0.7428` | `0.0929` | `0.2771` | `0.193810` |
| large_10class_30dim | `0.000011` | `0.0294` | `0.0049` | `0.2886` | `0.018752` |
| more_local_steps | `0.001908` | `0.1495` | `0.0187` | `0.2771` | `0.144444` |
| high_within_class_variance | `0.000358` | `0.0519` | `0.0065` | `0.9828` | `0.164738` |

### Honest negative results

1. **Minibatch SGD breaks the first-order TANGO estimator.** TANGO is worse than passive multi-round on prototype MSE (`6.59` vs `0.75`). The paper must not claim exact theorem coverage for minibatch FL without a revised observation model.
2. **Nonzero head weights degrade count recovery** (MAE `0.74`) more than prototype shape (MSE `0.0097`). Count and prototype recovery must be reported separately.
3. **Individual recovery never improves** with better prototypes; high-variance scenario individual MSE `0.983` stays near floor `0.837`.

### Positive confirmations

1. **Active probing dominates passive and public priors** on balanced data (`950x` vs passive, `2743x` vs public prior).
2. **Scale-up to 10 classes / 30 dimensions** remains strong (prototype MSE `1.06e-5`).
3. **Oracle aggregate** confirms TANGO approaches the perfect-moment ceiling on prototypes in exact-regime scenarios (balanced, large-scale).

## Comparison to SHARD (tier framing)

SHARD Stage 2 requires intermediate batch-average rows and incidence. Under terminal-only observations, SHARD has **no input**. TANGO is not a SHARD competitor on individual reconstruction; it is a **different tier** (Tier 1) with a **weaker aggregate target**.

| Tier | Observation | Individual recovery | Aggregate recovery |
|---|---|---|---|
| Tier 0 | One passive terminal round | No | Rank-limited |
| Tier 1 | Active terminal probes (TANGO) | No | Yes (exact regime) / approximate (stress) |
| Tier 2 | Partial intermediate + GARD prior | Conditional | Partial snapshots |
| Tier 3 | SHARD intermediate gradients | Yes (strong assumptions) | Yes |

## Paper Structure (draft-ready)

See `paper/outline.md` for section skeleton. Recommended flow:

1. Introduction and observation ladder.
2. Threat model: active terminal probing.
3. Theory: exact aggregate identifiability + individual non-identifiability + scope table.
4. TANGO algorithm and baselines.
5. Experiments (Round 09 table + Round 05/06/08 context).
6. Limits: minibatch failure, no individual recovery, hidden-representation endpoint.

## Artifacts

- `code/benchmark_round09.py`
- `artifacts/round09_metrics.json`
- `artifacts/round09_baselines.svg`
- `artifacts/round09_count_vs_proto.svg`
- `logs/experiment_round09.log`
- `tutorial/tutorial.md` (Round 09 section)
- `paper/outline.md`

## Viability

Round 09 raises paper readiness from **3.5/5** toward **4/5** by:

- closing evaluation gaps requested in Round 08;
- adding baselines that separate TANGO from oracle/passive/prior methods;
- reporting count and prototype metrics separately;
- documenting an honest minibatch failure mode.

Remaining gaps for the manuscript: real FL backbone (nonlinear), secure aggregation, label noise, and decoder evidence linking hidden prototypes to raw-input semantics.
