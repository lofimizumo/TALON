#!/usr/bin/env python3
"""Scaffold a dual-agent research run folder."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--goal", required=True, help="Major research goal (one sentence).")
    p.add_argument("--slug", required=True, help="Short folder name, e.g. shard-defense.")
    p.add_argument("--rounds", type=int, default=3, help="Planned scientist/supervisor rounds.")
    p.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Workspace root (default: cwd). Run dir: <root>/research-artifacts/<slug>/",
    )
    args = p.parse_args()

    root = (args.root or Path.cwd()).resolve()
    run_dir = root / "research-artifacts" / args.slug
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty: {run_dir}")

    dirs = [
        run_dir,
        run_dir / "rounds" / "round_01",
        run_dir / "literature",
        run_dir / "artifacts",
        run_dir / "code",
        run_dir / "logs",
        run_dir / "tutorial",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    config = {
        "goal": args.goal,
        "slug": args.slug,
        "rounds_planned": args.rounds,
        "acceptance": {
            "supervisor_verdicts": ["ACCEPT", "ACCEPT_WITH_MINOR"],
            "max_critical_issues": 0,
            "require_tutorial_pdf": True,
        },
        "domain_notes": "",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    readme = f"""# Research run: {args.slug}

**Goal:** {args.goal}

**Planned rounds:** {args.rounds}

## Status

| Round | Scientist | Supervisor | Verdict |
|-------|-----------|------------|---------|
| 1 | [ ] | [ ] | |

## Paths

- Config: `config.json`
- Tutorial: `tutorial/tutorial.md` → `tutorial/tutorial.pdf`

## Workflow

Use the **autonomous-research** skill: one Task for `research-scientist`, then one for `research-supervisor` per round.
"""
    (run_dir / "README.md").write_text(readme, encoding="utf-8")

    (run_dir / "tutorial" / "tutorial.md").write_text(
        f"# Idea tutorial: {args.slug}\n\n"
        f"**Goal:** {args.goal}\n\n"
        "_Draft after final round; compile to PDF with pandoc (see skill reference)._\n",
        encoding="utf-8",
    )

    print(f"Created {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
