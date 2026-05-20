---
name: autonomous-research
description: >-
  Runs multi-round autonomous research using two independent subagents (research
  scientist and research supervisor), not a single dual-role orchestrator.
  Scientist proposes and revises ideas toward a major goal; supervisor audits
  honesty, feasibility, proofs, and experiment code/results. Use for literature
  reviews, hypothesis exploration, research roadmaps, experiment design, or when
  the user asks for dual-agent research loops, supervisor audit, or research
  artifact folders with tutorial PDF.
---

# Autonomous research (dual subagent)

Conduct **multi-round** research with **two separate subagent invocations per round**. The primary agent **coordinates files and scheduling only** — it must **not** draft proposals or audit judgments in the parent thread (except logistics: paths, round counters, PDF build commands).

## Before starting (required if missing)

If the user did not specify, **ask** (use AskQuestion when available):

1. **Round count** — e.g. 2, 3, 5, or “until acceptance.”
2. **Acceptance standard** — what must be true to stop early, e.g.:
   - Supervisor **verdict** `ACCEPT` or `ACCEPT_WITH_MINOR`
   - No open **Critical** audit items
   - Reproducible experiment checklist passed
   - Tutorial draft complete at agreed depth

Record answers in `config.json` before round 1.

Also confirm: **major goal**, **domain constraints** (methods, datasets, ethics), **workspace root** for the run folder, and whether **code/experiments** are in scope this session.

## Subagents (two roles, never merged)

| Role | Agent file | Task `subagent_type` |
|------|------------|----------------------|
| Research scientist | `research-scientist.md` in `.cursor/agents/` or `~/.cursor/agents/` | `generalPurpose` (or `explore` for read-only literature passes) |
| Research supervisor | `research-supervisor.md` | `generalPurpose` |

Install agents from this skill if missing: copy [agents/research-scientist.md](agents/research-scientist.md) and [agents/research-supervisor.md](agents/research-supervisor.md) into `~/.cursor/agents/` (or project `.cursor/agents/`).

**Per round — strict order:**

1. **Scientist only** — read prior `supervisor_review.md` (if any), produce/revise proposal, literature notes, code, experiments; write outputs under `rounds/round_NN/`.
2. **Wait for scientist to finish.**
3. **Supervisor only** — read scientist artifacts; **do not** receive hidden parent synthesis; write `supervisor_review.md` with verdict and scored rubric.
4. **Wait for supervisor to finish.**
5. Parent updates `README.md` round status, checks acceptance, starts next round or closes.

**Forbidden:** One Task prompt that says “first propose then critique”; parent generating `proposal.md` and `review.md` in one turn; supervisor invoked before scientist artifacts exist for that round.

## Initialize run folder

From workspace root:

```bash
python ~/.cursor/skills/autonomous-research/scripts/init_research_run.py \
  --goal "YOUR MAJOR GOAL" \
  --slug short-topic-name \
  --rounds 3
```

Or create the same tree manually — see [reference.md](reference.md).

Default location: `research-artifacts/<slug>/` at workspace root (override with `--root`).

## Round workflow

### Round 1 — scientist

Delegate with a prompt that includes:

- `config.json` path and major goal
- Output paths: `rounds/round_01/researcher_proposal.md`, `revision_log.md`, optional `literature/`, `code/`, `artifacts/`
- Instruction to follow the scientist agent body and **write all deliverables to disk**

### Round 1 — supervisor

Delegate **in a new Task** with:

- Paths to scientist files only (+ `code/` if present)
- Output: `rounds/round_01/supervisor_review.md` (verdict + rubric + audit of code/results)
- Instruction to follow supervisor agent body; be **skeptical** of idealized assumptions and proof gaps

### Rounds 2+

Scientist prompt must include **full text or paths** of the latest `supervisor_review.md`. Scientist may:

- Revise the current direction, or
- Pivot to a novel direction if justified in `revision_log.md`

Repeat scientist → supervisor until `config.rounds` exhausted or acceptance met.

### After final round

1. Merge stable narrative into `tutorial/tutorial.md` (scientist draft, supervisor fact-check pass optional as round N+1 read-only).
2. Build PDF — see [reference.md](reference.md) (`tutorial/tutorial.pdf`).
3. Update root `README.md` with index, verdict history, and links.

## Acceptance (stop early)

Stop when **any** configured condition holds (document which in `README.md`):

| Signal | Typical rule |
|--------|----------------|
| Verdict | `ACCEPT` or `ACCEPT_WITH_MINOR` in latest `supervisor_review.md` |
| Critical issues | Zero open items tagged `Critical` |
| Rounds | `current_round >= config.rounds` |

If rounds exhausted without acceptance, still deliver artifacts and label status `INCONCLUSIVE` in README.

## Parent agent allowed / forbidden

| Allowed | Forbidden |
|---------|-----------|
| `init_research_run.py`, README/config updates | Writing core research claims without scientist Task |
| Spawning two Tasks per round in order | Combined propose+audit in one Task or parent message |
| Pandoc/latex build for tutorial PDF | Overriding supervisor verdict without new supervisor round |
| Copying subagent output into canonical paths if tool cannot write | Cherry-picking only positive results for tutorial |

## Output checklist

```
research-artifacts/<slug>/
  README.md
  config.json
  rounds/round_NN/
    researcher_proposal.md
    supervisor_review.md
    revision_log.md
  literature/
  artifacts/          # figures, tables, exports
  code/               # experiment scripts
  logs/
  tutorial/
    tutorial.md
    tutorial.pdf
```

## Additional resources

- Folder layout, prompts, PDF build: [reference.md](reference.md)
- Agent prompts to install: [agents/](agents/)

## Legacy name

The old skill `autonomous-qfl-researcher` is superseded by this skill for all domains (not only federated or quantum learning).
