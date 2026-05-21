# Round 04 — Revision log

## Status

Acceptance packaging for Round 03 science. No `supervisor_review.md` (per mission).

## New / updated artifacts

| Path | Change |
|------|--------|
| `code/qterm_attack.py` | **New** — `QtermAttack`, `QtermTier`, LASA-QTERM production API |
| `code/benchmark_round04.py` | **New** — consolidated smooth+MNIST, `acceptance_table` |
| `paper/method.md` | **New** — method definition |
| `paper/scope.md` | **New** — tier table T1 / T1p / T1b / T2 |
| `tutorial/tutorial.md` | **Rewritten** — QFL tutorial + TALON link |
| `config.json` | `current_round: 4` |
| `README.md` | Round 04 status row |
| `literature/qfl_terminal_snapshot.md` | Round 04 LASA-QTERM pointer |

## Unchanged (carried forward)

- `paper/impossibility_t1.md` — T1 proof
- `code/terminal_attacks.py` — estimator primitives
- `code/benchmark_round03.py` — historical reproducibility

## Pivot vs persist

**Persist** impossibility-first T1 story. **Package** T1p/T1b as explicit non-primary tiers via LASA-QTERM. No new imputation or headline conflation.

## Runnable verification

```bash
python code/benchmark_round04.py
```

Expected: `artifacts/round04_metrics.json` with `acceptance_verdict.primary_t1_path == "impossibility"`.
