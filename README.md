# TALON research run (`06.TALON`)

**Workspace:** `/Users/yetao/Documents/06.My Papers/CQU/2026/`  
**This folder:** `/Users/yetao/Documents/06.My Papers/CQU/2026/06.TALON/`  
**SHARD paper (optional, read-only):** `/Users/yetao/Documents/06.My Papers/CQU/2026/01.SHARD/`  
**Vendored code:** `vendor/shard_sim/` (copied from SHARD supplementary materials; no external import at runtime)

**Goal:** Find a stronger direction than SHARD by prioritizing Stage 2 / full-method redesign and reducing the requirement for intermediate minibatch gradients. Stage-3-only improvements are background.

**Status:** **COMPLETE — final Round 8 audit accepted TALON/TANGO as a paper direction with important limits** (2026-05-20)

| Round | Scientist | Supervisor | Verdict |
|-------|-----------|------------|---------|
| 1 | done | done | REVISE_MAJOR (LAPIN loses to SHARD) |
| 2 | done | done | REVISE_MINOR (JOLI wins MSE in compressive regime) |
| 3 | done | done | **ACCEPT_WITH_MINOR** |
| 4 | done | done | REVISE_MAJOR (GARD promising but oracle-driven) |
| 5 | done | done | REVISE_MAJOR / PIVOT_REQUIRED (assignment dominates) |
| 6 | done | done | REVISE_MAJOR / PIVOT_REQUIRED (JASPER negative result) |
| 7 | done | done | ACCEPT_WITH_MAJOR_LIMITATIONS / REFINE_AND_INTEGRATE |
| 8 | done | done | **ACCEPT AS FINAL PAPER DIRECTION, NOT SHARD-EQUIVALENT** |

## Final Outcome: TALON / TANGO

**TALON**: Terminal Aggregate Leakage with Observation tiers and Non-identifiability limits.

**TANGO**: Terminal Aggregate Neural Gradient Observation, the main positive terminal-only method.

Supported thesis:

> Active terminal probes can identify class-level hidden prototypes and aggregate moments without intermediate minibatch gradients, but terminal aggregate moments do not identify individual samples. SHARD-equivalent individual recovery requires stronger observations or side information.

- **Final code:** `code/benchmark_round08.py`
- **Final artifacts:** `artifacts/round08_metrics.json`, `artifacts/round08_tango_stress.svg`
- **Tutorial:** `tutorial/tutorial.md`

## Reproduce

```bash
cd "/Users/yetao/Documents/06.My Papers/CQU/2026/06.TALON"
.venv/bin/python code/benchmark_round08.py
```

Benchmarks that import `shard_sim` (rounds 01–03) use `vendor/shard_sim/` via `code/_paths.py`. Rounds 04–08 need only numpy/matplotlib.

## Key Result

Balanced synthetic hidden-feature benchmark, 8 active terminal probes, **zero intermediate batch gradients**:

- Prototype MSE: `0.0001506`
- One passive terminal round prototype MSE: `0.1431415`
- Active gain: about `950.6x`
- Individual reconstruction from prototypes: `0.266669`
- Within-class variance floor: `0.253177`

## Cursor Cloud

Skills and subagents for Cloud Agents live under:

- `.cursor/skills/autonomous-research/`
- `.cursor/agents/research-scientist.md`, `research-supervisor.md`

See `AGENTS.md` for how agents should use them.

## Layout

| Path | Role |
|------|------|
| `.cursor/skills/` | Agent skills (synced via git for Cloud) |
| `config.json` | Run metadata and acceptance criteria |
| `rounds/round_0N/` | Proposals, reviews, revision logs |
| `code/` | Benchmarks and methods |
| `artifacts/` | Metrics and plots |
| `tutorial/` | Idea tutorial |
