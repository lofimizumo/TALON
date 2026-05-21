# Round 12 — Revision log (response to Round 11 REVISE_MAJOR)

## Verdict addressed

Round 11 **REVISE_MAJOR** (3 Critical, 4 Major). This round targets every prioritized supervisor action item with code, proofs, and re-run metrics.

## Changes vs Round 11

| Item | Round 11 | Round 12 |
|---|---|---|
| Lemma MB-A algebra | Suspect \((M/B)(Np-n_c)\) identity | **Corrected** partition-sum proof; promoted to Lemma MB-A |
| Within-step drift | Mentioned only | **Lemma MB-B** + `within_step_weight_drift()` empirical mean **0.112** |
| W₀≠0 minibatch | Single linearized scale | **`tango_mb_iter`**: one Jacobian bias step (\(\alpha=0.15\)); nonzero-init mean proto **0.013** vs MB **0.017** |
| Count MAE (primary) | **0.362** (worse than passive **0.277**) | **0.100** (beats passive) via least-perturbed probe round |
| STORM | Identical to TANGO-MB | **Removed**; replaced by **`tango_mb_drift2`** and **`stack_mb_ridge`** |
| Active vs scaling | Not quantified | passive+MB **0.284** vs TANGO-MB **0.011** → **~26×** active over scaling-only |
| Reporting | Mean only | **Median + IQR** per scenario/method |
| Scenarios | 4 | **7** (+ label noise, terminal noise, frozen MLP minibatch) |
| Primary proto MSE | **0.0110** | **0.0110** (unchanged; still beats passive **0.753**) |

## Files touched

- `paper/proofs.md` — Lemma MB-A fix, MB-B drift bound, MB-Iter, code map
- `paper/draft.md` — Round 12 headlines, dual metrics, active vs scaling
- `code/benchmark_round12.py` — new benchmark (replaces STORM)
- `artifacts/round12_metrics.json`, `artifacts/round12_minibatch_methods.svg`
- `logs/experiment_round12.log`
- `config.json` — `current_round: 12`

## Persist / pivot

**Persist** TANGO-MB \(T_{\mathrm{eff}}\) scaling as the core fix; **extend** with drift2, principled counts, and Jacobian iter for \(W_0 \neq 0\). **Pivot away** from STORM naming (no added tier at ridge=0).

## Open risks (unchanged)

- SHARD `level3_invert` comparison not in this round (`require_baseline_comparison_vs_shard` still open).
- Drift2 mean proto **0.025** on primary — helpful but not default; MB remains primary.
- `stack_mb_ridge` evaluated on passive probes by design (ill-conditioning); not primary headline.

## Reproduce

```bash
python3 code/benchmark_round12.py
```
