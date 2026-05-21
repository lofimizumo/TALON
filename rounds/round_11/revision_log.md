# Round 11 Revision Log

## Trigger

Phase 2 mandate: Round 10 ACCEPT suspended for core attack until **minibatch SGD** prototype recovery beats passive (`minibatch_sgd` MSE). User/supervisor: first-order full-batch assumption is the fatal flaw.

## What changed

| Item | Action |
|---|---|
| Root cause | Identified terminal-delta **scale** error: \(T_{\mathrm{eff}} = T(N/B)\) not \(T\) |
| New methods | **TANGO-MB**, **STORM** in `code/benchmark_round11.py` |
| Primary eval | `minibatch_sgd` listed first; Phase-2 headline in JSON |
| Vanilla TANGO | Retained as negative control (still fails at `6.59`) |
| Full-batch tier | TANGO-MB reduces to vanilla when `use_minibatch=False` |
| Docs | `paper/proofs.md` Remark MB-A; `literature/minibatch_gradient_scaling.md` |
| Config | `current_round: 11`, phase2 status note |

## What we did NOT do (per mandate)

- Re-label minibatch failure as final “limits” without a fix
- Add full-batch-only stress tests as the main result
- Claim Phase-2 win from decoder/frozen full-batch rows
- Hide behind synthetic theorems without minibatch implementation

## Key numbers (`artifacts/round11_metrics.json`)

```
minibatch_sgd:
  tango_vanilla  prototype_mse = 6.5877
  tango_mb       prototype_mse = 0.0110  ← Phase-2 target
  passive        prototype_mse = 0.7532
  gain           ≈ 68.3× vs passive

balanced_clean:
  tango_mb = tango_vanilla = 0.000151  (regression OK)
```

## Commands

```bash
python3 code/benchmark_round11.py
```

Runtime ~0.7s; logs: `logs/experiment_round11.log`.

## Honest assessment

**Phase-2 scientist target met:** TANGO-MB beats passive on minibatch prototype MSE. Vanilla TANGO still fails as expected. Residual minibatch error (`0.011`) is above exact tier (`0.00015`) due to within-step weight drift—documented as approximate tier, not hidden.

**Supervisor decision pending:** Whether scale correction + public-\(B\) assumption satisfies “fundamental fix” vs requires deployed-FL validation.
