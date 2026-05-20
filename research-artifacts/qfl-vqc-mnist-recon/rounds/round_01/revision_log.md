# Round 01 — revision log

## Summary

First implementation of the VQC-stack MNIST reconstruction benchmark for `qfl-vqc-mnist-recon`.

## Changes

| Area | Action |
|------|--------|
| `code/_paths.py` | Wire `vendor/shard_sim`, parent `code/` (JOLI), sibling `qfl-terminal-snapshot/code` (LASA-QTERM). |
| `code/benchmark_round01.py` | End-to-end: FederatedDataLoader MNIST 8×8 → SurrogateQFL gradients → SHARD / T1p / T1b snapshots → shard + JOLI L3 → metrics + grid. |
| `rounds/round_01/` | Researcher proposal + this log (no `supervisor_review.md` per mission). |
| `literature/vqc_stack_bridge.md` | Pre-existing stack definition (unchanged). |

## Rationale

- **8×8 MNIST** chosen for Round 1 to make multi-path × multi-inverter × multi-seed L3 feasible; 28×28 remains the config-preferred target for Round 2+.
- Import **QtermAttack** from sibling rather than reimplementing partial/B=1 disaggregation.
- Store reconstruction grid from seed **7** during the main run to avoid a second full pipeline pass.

## Pivot vs persist

**Persist** the parent-repo SurrogateQFL + SHARD + JOLI stack; no pivot away from real MNIST pixels or Level-3 image metrics.

## Post-run notes (Round 01 experiments)

- Re-ran with `SHARD_MAX_ITER=200`; seeds 7 and 11 still show SHARD L2 non-convergence (`||ΔS||_F` ≫ ε).
- **LASA-QTERM T1b** beats SHARD oracle on mean input MSE (0.0026 vs 0.147) — not because T1b is weaker-tier, but because B=1 terminal disaggregation succeeds where full-batch SHARD stalls on these shuffles.
- Aggregate targets (MSE ≤ 0.05, PSNR ≥ 18 dB) are met only via T1b; oracle and T1p miss.

## Next round (anticipated)

- Fix SHARD oracle baseline (more `max_iter`, initialization) so upper bound is valid.
- Scale to 14×14 or 28×28; sweep `dim_g`, L3 `adam_steps` / `n_batch`.
- Report per-seed target pass rate, not mean-only.
