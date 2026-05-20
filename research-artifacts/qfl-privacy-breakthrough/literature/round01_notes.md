# Round 01 literature & evidence notes

## Sources (primary / in-repo)

| ID | Source | Use in R01 |
|----|--------|------------|
| S1 | `vendor/shard_sim` — SHARD attacker pipeline | Stage-2 oracle baseline; QFL track |
| S2 | `qfl-terminal-snapshot` — LASA-QTERM, `impossibility_t1.md` | T1p attack; tier semantics |
| S3 | Parent `/workspace/rounds/round_05` — GARD stress test | GARD-SPARSE protocol template |
| S4 | `literature/prior_findings_bridge.md` | Lane inventory |

## Evidence table (Round 01 experiments)

| Lane | Claim | Result | vs config target |
|------|-------|--------|------------------|
| GARD-SPARSE | Fewer rows @ equal MSE | 9 rows @ MSE≤0.10 vs 60 for LS | **A: 85% reduction (pass)** |
| QFL-SHIELD | Cut T1p MSE 50%, ≤10% utility | −27% attack reduction; utility ≫10% | **B: fail** |
| LEAK-CERT | Cert 2× tighter than naive | tight/naive = 1.0 | **C: fail** |
| ASSIGN-LOCK | Hide assignment | Not implemented | — |

## Interpretation (supported)

- **Sparse graph prior** remains the strongest *attacker-side* observation reducer when assignment is known and snapshots are smooth low-rank (consistent with parent Round 05).
- **Client subspace mask** without adversarial calibration can **amplify** structured attacks that already exploit low-rank geometry (Qterm partial disaggregate).
- **T1p with 70 rows** over-constrains the linear system for a simple dof certificate; tighter bounds need tier-specific rank or sensitivity analysis.

## Speculation (labeled)

- Combining GARD-SPARSE *attacker model* with ASSIGN-LOCK *defense* may trade off: fewer rows for attacker vs hidden incidence.
- DP-style noise on **published batch means** (server-side) may succeed where client snapshot masking failed.

## Citations to add (external, not fetched R01)

- Federated learning gradient leakage surveys (e.g., Zhu et al., deep leakage from gradients).
- Graph Laplacian regularization in inverse problems / GARD-style priors.

## Runnable path

```bash
cd research-artifacts/qfl-privacy-breakthrough
python3 code/benchmark_round01.py
```

Outputs: `artifacts/round01_metrics.json`, `logs/experiment_round01.log`.
