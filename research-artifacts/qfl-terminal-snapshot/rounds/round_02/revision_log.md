# Round 02 — Revision log

## Status

Response to `rounds/round_01/supervisor_review.md` (**REVISE_MAJOR**). No `supervisor_review.md` written this round (per mission).

## Changes from Round 01

### Code

- `code/terminal_attacks.py` — removed `sqrt(100)` anchor; Fiedler graph spread; `permuted_chain_laplacian`, `pad_partial_for_shard`, `fixed_order_snapshot_mse`, updated `snapshot_row_std` diagnostic.
- `code/benchmark_round02.py` — new benchmark: graph λ ablation, T1b B=1, partial trajectory, SHARD 200-iter diagnostics, optional `--mnist`.
- `config.json` — `current_round: 2`.

### Artifacts

- `artifacts/round02_metrics.json` — smooth + mnist tracks, per-seed ablations.
- `logs/experiment_round02.log` — full run log.

### Literature

- `literature/qfl_terminal_snapshot.md` — T1b / partial tiers, graph degeneracy note.

## Supervisor issue mapping

| Issue | Round 02 response |
|-------|-------------------|
| Graph degenerate | Fixed; graph row-std > 0; beats passive on MSE slightly |
| No individual recovery at T1 | Confirmed; B=1 + partial tiers recover |
| SHARD under-converged | 200 iterations; matching still 0% (strict metric) |
| Misleading ratio | JSON uses `terminal_mse_over_shard_mse` with interpretation string |
| MNIST | Optional track run |

## Pivot vs persist

**Persist** individual-snapshot goal. **Pivot** claim language: B=1 / partial terminals are distinct threat tiers with higher row budget—not “free” SHARD parity from epoch means alone.

## Next round

- JASPER-TERMINAL / soft assignment without full intermediates.
- Formal T1 impossibility proposition in proposal.
- SHARD matching threshold audit (why 0% at good MSE).
