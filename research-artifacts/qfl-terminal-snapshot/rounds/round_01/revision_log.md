# Round 01 — Revision log

## Status

First scientist round (no prior supervisor review).

## What was delivered

- `rounds/round_01/researcher_proposal.md` — hypotheses H1–H3, QFL threat model, three terminal methods + SHARD oracle plan.
- `code/terminal_attacks.py` — passive mean, GRAPH-TERM, GRAPH-RANK-TERM.
- `code/benchmark_round01.py` — SurrogateQFL synthetic run, metrics, JSON export.
- `literature/qfl_terminal_snapshot.md` — terminal vs intermediate observation taxonomy.
- Runnable benchmark path logged under `logs/experiment_round01.log`.

## Design choices

- **Strict terminal count:** attacks use only epoch-averaged gradients (one row per epoch in Level-1 stack); `observed_intermediate_batch_gradients = 0`.
- **Oracle graph:** chain on sample index mirrors parent Round-05 favorable GARD setting; documented as side information, not learned from gradients.
- **Did not overwrite** parent `/workspace` TALON artifacts; cited as negative limit for aggregates-only.

## Pivot vs persist

**Persist** individual-snapshot goal; **narrow** Round-01 claim to identifiability diagnosis rather than promising SHARD parity from terminals alone.

## Next round (if supervisor REVISE)

- Add permuted/wrong graph and unknown-assignment stress.
- MNIST-backed SurrogateQFL snapshots.
- Probe B=1 federated terminal-only regime as separate threat tier.
