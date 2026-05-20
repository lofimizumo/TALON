# Agent instructions (TALON)

## Multi-round research

- Use the **`autonomous-research`** skill (`.cursor/skills/autonomous-research/SKILL.md`) for scientist/supervisor research loops.
- Delegate with subagents defined in `.cursor/agents/`:
  - `research-scientist` — proposals, code, experiments
  - `research-supervisor` — audit, verdict, honesty checks
- Run benchmarks from repo root: `.venv/bin/python code/benchmark_round08.py` (see `README.md`).

## Reproducibility

- Vendored SHARD simulator: `vendor/shard_sim/`
- Experiment outputs: `artifacts/`, `logs/`, `rounds/round_NN/`
