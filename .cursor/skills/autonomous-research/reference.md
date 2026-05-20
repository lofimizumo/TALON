# Autonomous research — reference

## `config.json` template

```json
{
  "goal": "One-sentence major research goal",
  "slug": "topic-slug",
  "rounds_planned": 3,
  "acceptance": {
    "supervisor_verdicts": ["ACCEPT", "ACCEPT_WITH_MINOR"],
    "max_critical_issues": 0,
    "require_tutorial_pdf": true
  },
  "domain_notes": "Optional: methods, datasets, ethics, hardware limits",
  "started_at": "ISO-8601"
}
```

## Round file contracts

### `researcher_proposal.md`

- **Problem & goal** (1 paragraph)
- **Hypothesis / claim** (testable)
- **Related work** (bullets + citations or arXiv IDs)
- **Method** (assumptions explicit)
- **Experiment plan** (metrics, baselines, failure modes)
- **Changes this round** (if round > 1)
- **Open risks**

### `supervisor_review.md`

- **Verdict**: `REJECT` | `REVISE_MAJOR` | `REVISE_MINOR` | `ACCEPT_WITH_MINOR` | `ACCEPT`
- **Honesty / reproducibility audit** (code paths, frozen seeds, hardcoded results, train/test leakage)
- **Feasibility** (compute, data, timeline)
- **Theory** (unstated assumptions, proof gaps, over-claims)
- **Rubric** (1–5): novelty, soundness, feasibility, impact, evaluability
- **Issues** tagged `Critical` | `Major` | `Minor`
- **Concrete suggestions** (creative + practical)

### `revision_log.md`

Short changelog: what changed, why, pivot vs persist.

## Scientist Task prompt skeleton

```text
You are the research-scientist subagent.

Goal: {from config.json}
Round: {N} of {planned}
Prior supervisor review: {path or "none"}

Read your agent definition at ~/.cursor/agents/research-scientist.md

Deliverables (write files, do not only chat):
- rounds/round_{NN}/researcher_proposal.md
- rounds/round_{NN}/revision_log.md
- Update literature/, code/, artifacts/ as needed

If experiments are in scope: implement or revise under code/, log commands in logs/.
Cite primary sources; label speculation.
```

## Supervisor Task prompt skeleton

```text
You are the research-supervisor subagent.

Round: {N}
Read ONLY these inputs:
- rounds/round_{NN}/researcher_proposal.md
- rounds/round_{NN}/revision_log.md
- code/ and artifacts/ if present

Read your agent definition at ~/.cursor/agents/research-supervisor.md

Write rounds/round_{NN}/supervisor_review.md with verdict and full audit.
Do not propose a replacement research program; audit and advise.
```

## Tutorial PDF

Preferred path after final accepted or last round:

```bash
cd research-artifacts/<slug>/tutorial
pandoc tutorial.md -o tutorial.pdf \
  --pdf-engine=xelatex \
  -V geometry:margin=1in \
  --toc
```

Fallback: `pandoc tutorial.md -o tutorial.pdf` (pdflatex). If pandoc missing, leave `tutorial.md` and note build command in README.

Tutorial should read as a **self-contained idea guide**: motivation, method sketch, limitations, pointers to `code/` and key figures in `artifacts/`.

## Parallelism

- **Across rounds**: strictly sequential (scientist → supervisor).
- **Inside scientist round**: may use additional `explore` Tasks for literature **only** if the scientist subagent delegates; parent should not spawn explore + supervisor in parallel on the same round’s proposal before supervisor review.

## Rubric (supervisor)

Score 1–5 each:

| Criterion | Question |
|-----------|----------|
| Novelty | New framing or mechanism vs obvious baselines? |
| Soundness | Claims match assumptions and evidence? |
| Feasibility | Can this be built and evaluated with stated resources? |
| Impact | Matters for the stated major goal? |
| Evaluability | Clear metrics, baselines, ablations? |
