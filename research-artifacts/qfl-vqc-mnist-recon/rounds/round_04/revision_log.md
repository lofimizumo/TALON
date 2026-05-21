# Round 04 — revision log

## Summary

Combine **best L3 from parent Round 03** (JOLI `tv_lbfgs` Pareto point) with **GARD-SPARSE oracle partial rows** and **T1p + graph prior** snapshot polish; optional **14×14** if 28×28 misses ≥2/3 seed acceptance.

## Changes

| Area | Action |
|------|--------|
| `code/benchmark_round04.py` | New E2E: `gard_sparse_oracle`, `lasa_qterm_T1p_graph`, `shard_oracle` + `joli_r3`; 28×28 `dim_g∈{160,100}`; conditional 14×14 |
| `rounds/round_04/` | Researcher proposal + this log (**no** supervisor review) |

## Rationale

- Round 02: best compressive mean MSE ~0.08 (T1b) still below PSNR/MSE targets; snapshot stage is the bottleneck.
- Privacy breakthrough: GARD-SPARSE @ oracle + partial rows achieves snapshot MSE ≤ 0.10 — port MAP + λ grid to MNIST incidence.
- Terminal snapshot R03: graph priors on mean-only observations — apply as **pre-L3 polish** on T1p recoveries.

## Pivot vs persist

**Persist** full VQC stack and 28×28 primary track; add snapshot-path lanes before further L3 budget increases.
