# Round 05 — Revision log

## Context

Round 04 closed assignment-focused lanes (JASPER-Q diagnostic win, ASSIGN-LOCK v2 fail, LEAK-CERT-T1p fail). Round 05 executes the deferred **criterion B** path: **Snapshot-DP** on batch means (not SHIELD subspace masking).

## What we did

- Added `code/benchmark_round05.py` — Gaussian/Laplace batch-mean noise, ε/σ grid, three attacker channels.
- Ran full grid → `artifacts/round05_metrics.json`, `logs/experiment_round05.log` (~271s).
- Wrote `rounds/round_05/researcher_proposal.md`.
- Set `config.json` `current_round` → 5.
- **No** `supervisor_review.md` (per mission).

## Criterion snapshot

| Criterion | Result |
|-----------|--------|
| B (≥50% attack MSE ↑ @ ≤10% utility) | **Fail** — 0 feasible utility cells |
| Secondary (wrong40 @ 0.15, ≥50% rows) | **Fail** — blocked on B |

## Mechanism outcomes

- **T1p:** High σ can **more than double** attack MSE, but utility always ≫10%.
- **SHARD / JASPER-Q Stage-2:** Attack MSE largely **insensitive** to batch-mean noise (<12% increase).
- **Snapshot-DP vs SHIELD:** Same utility wall; different attack channel sensitivity.

## Next round

- Formal **kill** Snapshot-DP in proposal unless utility metric is redesigned.
- Return to assignment / cert lanes per Round 04 preview, or new defense class (e.g. secure aggregation) with pre-registered utility.
