# Round 08 Revision Log

## Final Integration

Round 08 follows the Round-07 supervisor instruction to stop forcing individual recovery from terminal aggregate updates. I integrated the prior rounds into a single framework:

- **TALON:** Terminal Aggregate Leakage with Observation tiers and Non-identifiability limits.
- **TANGO:** the terminal-only positive result for aggregate/prototype leakage.
- **GARD:** conditional Stage-2 nullspace prior when incidence and graph side information are good.
- **JASPER:** negative assignment-first result and evidence for an unknown-assignment barrier.
- **JOLI:** Stage-3 polish background, not a threat-model reduction.

## Code and Experiments Added

Added `code/benchmark_round08.py`, extending the Round-07 TANGO benchmark with stress scenarios:

1. Balanced clean setting.
2. Class imbalance.
3. Terminal update noise with std `1e-3`.
4. Weak active bias probes causing poorer conditioning.
5. More local training steps, increasing first-order terminal approximation error.
6. Higher within-class variance, increasing the individual non-identifiability floor.

The script writes:

- `artifacts/round08_metrics.json`
- `artifacts/round08_tango_stress.svg`
- `logs/experiment_round08.log`

Command used:

```bash
"source/experiments/vqc_pennylane/.venv/bin/python" "research-artifacts/novel-inversion-vs-shard/code/benchmark_round08.py"
```

Runtime was under one second.

## Numeric Headline

Round-08 benchmark headline:

- Balanced clean active terminal probes: prototype MSE `0.0001506`.
- Balanced clean one-round neutral terminal baseline: prototype MSE `0.1431415`.
- Active terminal probe gain: `950.6x`.
- Balanced clean individual MSE from repeated prototypes: `0.266669`.
- Balanced clean within-class variance floor: `0.253177`.

Stress results:

- Class imbalance: best prototype MSE `0.0003817`.
- Terminal noise `1e-3`: best prototype MSE `0.0002822`.
- Weak bias / poor condition: best prototype MSE `0.0003295`, condition number about `465.5`.
- More local steps: best prototype MSE `0.0019079`.
- High within-class variance: best prototype MSE `0.0003575`, but individual MSE `0.982833` versus floor `0.836947`.

## Interpretation Changes

The final proposal now explicitly avoids overclaiming:

- It does not say TANGO recovers individual samples.
- It defines TANGO's privacy endpoint as hidden-representation class prototypes, class counts, and dataset means.
- It states terminal-only individual non-identifiability through within-class zero-sum perturbations.
- It treats SHARD/GARD/JOLI as stronger-observation tiers rather than direct terminal-only competitors.

## Tutorial Update

Updated `tutorial/tutorial.md` from the old JOLI-only tutorial into a final tutorial for the expanded project:

- threat-model ladder;
- TANGO derivation;
- terminal aggregate identifiability theorem;
- individual non-identifiability theorem;
- GARD and JASPER roles;
- Round-08 reproduction instructions;
- paper structure and claim discipline.

## Final Candidate Paper Thesis

Terminal updates can leak accurate class-level prototypes without exposing intermediate minibatch gradients, but terminal aggregate moments do not identify individual samples. The right paper is a tiered leakage and limits paper: it maps what can be recovered as observation strength increases, and it proves where SHARD-style individual inversion needs stronger side information.
