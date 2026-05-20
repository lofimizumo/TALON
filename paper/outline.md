# TALON Paper Outline (draft skeleton)

## Title (working)

Beyond SHARD: Tiered Gradient Leakage and Identifiability Limits Under Active Terminal Probing

## Abstract (bullets)

- SHARD requires intermediate minibatch gradients; many FL deployments only leak terminal updates.
- We introduce TALON, an observation-tier framework, and TANGO, an active terminal probing attack on aggregate moments.
- Theory: exact aggregate identifiability under linear-head/full-batch assumptions; individual non-identifiability from class moments alone.
- Experiments: synthetic hidden-feature FL with baselines (passive, public prior, oracle aggregate) and stress tests including minibatch failure.
- SHARD remains the strong-observation tier for individual recovery.

## 1. Introduction

- Motivation: intermediate-gradient assumption too strong.
- Contribution list: tier ladder, TANGO, theorems, negative limits, evaluation.

## 2. Background and SHARD Tier

- SHARD Stage 2/3 recap.
- Why terminal-only breaks SHARD's incidence machinery.

## 3. Threat Model: Active Terminal Probing

- Server capabilities and public knowledge.
- Explicitly not passive honest-but-curious.
- Comparison table vs intermediate-gradient observation.

## 4. TALON Observation Tiers

- Tier 0 passive terminal.
- Tier 1 TANGO.
- Tier 2 GARD (conditional).
- Tier 3 SHARD/JOLI.
- Unknown assignment limit (JASPER).

## 5. Theory

### 5.1 Exact aggregate identifiability

- Assumptions list (linear head, zero w0, full-batch, first-order).
- **Full proofs:** `paper/proofs.md` (Theorem 1, Lemmas A–B).

### 5.2 Individual non-identifiability

- Within-class zero-sum perturbation family.
- **Full proof:** `paper/proofs.md` (Theorem 2).

### 5.3 Scope table (exact vs approximate)

- Map scenarios to theorem tier (Round 09 stress table + Round 10 frozen MLP / decoder minibatch).

## 6. TANGO Algorithm

- Probe design.
- Weight system for sums; bias system for counts.
- Prototype assembly.

## 7. Baselines

- Passive multi-round neutral probes.
- Public/statistical prior.
- Oracle aggregate upper bound (not an attack).

## 8. Experiments

- Round 09: synthetic hidden-feature setup, baselines, stress tests (`benchmark_round09.py`).
- Round 10: decoder probe + frozen MLP (`benchmark_round10.py`).
- Main table: count MAE vs prototype MSE by method (**report separately**).
- Stress: minibatch (negative, lead in limits), nonzero head, scale-up, variance.

## 9. Limits and Discussion (lead with failures)

1. **Minibatch SGD:** TANGO prototype MSE `6.59` vs passive `0.75` — see `paper/draft.md` §1.
2. **Count vs prototype:** passive can win on counts; TANGO wins on prototypes in exact regime.
3. **No individual recovery** (Theorem 2).
4. **Decoder bridge:** pixel class means track hidden prototypes when Tier-1 succeeds; linear synthetic map only.
5. **Frozen MLP:** Tier-1 survives fixed $\phi(x)$; does not cover representation drift.

## 10. Discussion and Ethics

- Hidden-representation privacy endpoint.
- Future: corrected minibatch estimators, secure aggregation.

## 11. Conclusion

- Supported thesis one paragraph.

## Appendix

- `paper/proofs.md` — Theorems 1–2, lemmas, proof–code map.
- Full probe conditioning analysis.
- GARD/JASPER/JOLI summary tables from prior rounds.
- Round 10 decoder + frozen MLP tables (`artifacts/round10_metrics.json`).
