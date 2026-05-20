# Round 14 — Revision log

## Supervisor carry-over (Round 13 REVISE_MINOR)

| Item | Action | Status |
|---|---|---|
| SHARD `level3_invert` cross | `code/shard_cross_round14.py` + JSON `shard_baseline_cross` | Done |
| JOINT vs DOPT duplicate | Uniform JOINT in `benchmark_round14.py` | Done |
| Scoped Phase-2 package | `paper/phase2_scope.md` | Done |
| `benchmark_round14.py` | Frozen MLP, label noise, median/IQR, secure-agg note | Done |
| Tutorial minibatch fix | §11 Round 14 TANGO-MB section | Done |

## Code changes

- `code/benchmark_round14.py` — new primary benchmark
- `code/shard_cross_round14.py` — synthetic SHARD L1–L3 + TANGO-MB tier table
- `paper/proofs.md` — Lemma MB-JOINT split (uniform vs D-opt)
- `paper/draft.md` — Round 14 headlines + SHARD tier table pointer
- `tutorial/tutorial.md` — §11 minibatch Lemma MB-A / TANGO-MB
- `config.json` — `current_round`: 14, status `PHASE2_ROUND14_SCOPED_CLOSURE`

## Not in scope (Round 14)

- Deployed CNN FL slice (permanent scope fence in `phase2_scope.md`)
- `supervisor_review.md` (parent agent)
- Holistic ACCEPT without stakeholder sign-off on §2 exclusions

## Verification

```bash
python3 code/benchmark_round14.py
```

Check `artifacts/round14_metrics.json` (run completed):
- `joint_vs_dopt_fix.identical_on_primary` = **0** (JOINT mean **0.272**, DOPT **0.283**)
- `shard_baseline_cross.shard_intermediate.level3_reconstruction_mse` = **0.0332**
- `phase2_scoped_accept.numeric_gate_minibatch_primary_win` = **true**
- `frozen_mlp_minibatch`: TANGO-MB median **0.091** vs passive **1.104**
