# Research run: qfl-vqc-mnist-recon

**Goal:** Real VQC-stack validation: MNIST image reconstruction via SurrogateQFL LASA pipeline—recover snapshots without intermediate gradients where possible, then Level-3 inversion—with target metrics PSNR≥18 dB and input MSE≤0.05.

**Planned rounds:** 8 | **Min rounds before accept:** 5 | **Stop:** metrics or round 8

## Workflow (autonomous-research skill)

Each round uses **two separate subagents** (strict order):

1. **Research scientist** → `rounds/round_NN/researcher_proposal.md`, `revision_log.md`, `code/`, `artifacts/`
2. **Research supervisor** → `rounds/round_NN/supervisor_review.md`

Parent agent coordinates paths and README only — does not draft proposals or verdicts.

## Fast iteration (default)

Multi-hour stalls on 28×28 L3: `paper/root_cause_l3_stall.md`. Quick mode: 14×14, fewer clients, capped JOLI, `StallWatchdog` in `code/run_monitor.py`.

```bash
cd /workspace/research-artifacts/qfl-vqc-mnist-recon

QFL_QUICK=1 PYTHONPATH=/workspace/vendor:/workspace/code python3 code/benchmark_quick.py
QFL_QUICK=1 PYTHONPATH=/workspace/vendor:/workspace/code python3 code/benchmark_round05.py
```

Full 28×28 (hours, supervisor-approved only):

```bash
QFL_QUICK=0 QFL_FULL=1 PYTHONPATH=/workspace/vendor:/workspace/code python3 code/benchmark_round05.py
```

| Variable | Effect |
|----------|--------|
| `QFL_QUICK=1` | 14×14 quick (default dev) |
| `QFL_FULL=1` | Force 28×28 acceptance profile |
| `QFL_ABORT_ON_STALL=1` | Hard-abort on watchdog stall |

## Status

| Round | Scientist | Supervisor | Verdict |
|-------|-----------|------------|---------|
| 1 | done | done | REVISE_MAJOR |
| 2 | done | done | REVISE_MAJOR |
| 3 | done | done | REVISE_MAJOR |
| 4 | done | done | REVISE_MAJOR |
| 5 | done | done | REVISE_MAJOR |
| 6 | — | — | — |

**Latest (R5):** Quick 14×14 adam sweep complete (~96 s); best PSNR **14.09 dB** (still below 18 dB gate); **0/2** joint pass. Full 28×28 run required next — see `rounds/round_05/supervisor_review.md`.

## Paths

- Config: `config.json`
- Root cause: `paper/root_cause_l3_stall.md`
- Budgets: `code/l3_budget.py`, `code/quick_config.py`, `code/run_monitor.py`
- Tutorial: `tutorial/tutorial.md`
