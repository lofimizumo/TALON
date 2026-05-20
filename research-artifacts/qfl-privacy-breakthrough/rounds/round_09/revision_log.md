# Round 09 revision log

## Executed

- Implemented `benchmark_round09.py` — PROBE-RAND lane only.
- Aligned gradient batch order with `make_incidence` rows.
- Ran 5 seeds; wrote metrics and experiment log.

## Outcome

**Honest failure** on breakthrough criterion: JASPER wrong40 @ 50% stays ~1.0 MSE with stale decode vs ~0.89 baseline; does not meet defense goal (≥50% attacker MSE increase at ≤10% utility loss).

## Next rounds

- SUBSPACE-MIX or server-side probe rotation **with** hidden \(A^{(e)}\) from gradient logs needs a stricter external attacker.
- Assignment / JASPER lanes remain primary per Rounds 06–07.
