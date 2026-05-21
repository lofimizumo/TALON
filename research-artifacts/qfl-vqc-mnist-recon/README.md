# Research run: qfl-vqc-mnist-recon

**Goal:** Real VQC-stack validation: MNIST image reconstruction via SurrogateQFL LASA pipeline—recover snapshots without intermediate gradients where possible, then Level-3 inversion—with target metrics PSNR≥18 dB and input MSE≤0.05.

**Planned rounds:** 8

## Fast iteration (default)

Multi-hour stalls on 28×28 L3 were traced in `paper/root_cause_l3_stall.md`. Quick mode uses **14×14 MNIST**, fewer clients/seeds, capped JOLI steps, no LAPIN, and sequential seeds.

```bash
cd /workspace/research-artifacts/qfl-vqc-mnist-recon

# Unified quick benchmark (~5–30 s)
QFL_QUICK=1 PYTHONPATH=/workspace/vendor:/workspace/code python3 code/benchmark_quick.py
# → artifacts/quick_metrics.json

# Round scripts respect QFL_QUICK=1 via quick_config.patch_module_globals
QFL_QUICK=1 PYTHONPATH=/workspace/vendor:/workspace/code python3 code/benchmark_round03.py
QFL_QUICK=1 PYTHONPATH=/workspace/vendor:/workspace/code python3 code/benchmark_round04.py
```

**Environment**

| Variable | Default | Effect |
|----------|---------|--------|
| `QFL_QUICK=1` | on | 14×14, N=12, 2 seeds, adam=120, no LAPIN |
| `QFL_QUICK=0` | — | Use full knobs in each benchmark script |
| `QFL_FULL=1` | — | Force 28×28 acceptance profile (hours) |
| `QFL_FULL_DIM_SWEEP=1` | — | dim_g sweep on full runs (round 5+) |

## Full acceptance (hours)

```bash
QFL_QUICK=0 QFL_FULL=1 PYTHONPATH=/workspace/vendor:/workspace/code python3 code/benchmark_round03.py
```

## Status

| Round | Scientist | Supervisor | Verdict |
|-------|-----------|------------|---------|
| 1 | done | done | 8×8 baseline |
| 2 | done | done | 28×28 dim sweep — no pass |
| 3 | partial | done | R3 metrics (full L3 stall fixed) |
| 4 | — | — | quick path wired |
| quick | done | — | `artifacts/quick_metrics.json` (~5 s) |

## Paths

- Config: `config.json`
- Root cause: `paper/root_cause_l3_stall.md`
- L3 budgets: `code/l3_budget.py`, `code/quick_config.py`
- Tutorial: `tutorial/tutorial.md`

## Workflow

Use the **autonomous-research** skill: scientist → supervisor per round. Prefer **`benchmark_quick.py`** between rounds; run full 28×28 only when quick trends justify the cost.
