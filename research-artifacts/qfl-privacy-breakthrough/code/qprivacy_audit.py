#!/usr/bin/env python3
"""QFL-PRIVACY-MAP CLI — audit defender/auditor tiers from metrics bundles.

Usage:
  python3 code/qprivacy_audit.py --bundle artifacts/round08_metrics.json
  python3 code/qprivacy_audit.py --bundle artifacts/round08_metrics.json --markdown
  python3 code/qprivacy_audit.py --tiers T1 T1p S2-oracle
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "code"))

from qprivacy_map_core import FRAMEWORK_ID, tier_table_markdown  # noqa: E402


def load_bundle(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "privacy_map" in data:
        return data
    # Legacy: synthesize minimal view from round07 leak cert + round01 gard
    raise ValueError(
        f"{path} is not a Round-08 privacy-map bundle; run benchmark_round08.py first."
    )


def format_audit_report(bundle: dict, *, markdown: bool) -> str:
    pm = bundle["privacy_map"]
    gates = pm["config_breakthrough"]
    lines: list[str] = []

    if markdown:
        lines.append(f"# {FRAMEWORK_ID} audit report\n")
        lines.append(pm["tier_table_markdown"])
        lines.append("\n## Config gates\n")
        for k, v in gates.items():
            lines.append(f"- **{k}**: `{v}`")
        lines.append("\n## Tier verdicts\n")
        for row in pm["tier_table"]:
            a = row["audit"]
            lines.append(f"### {row['tier_id']} — {row['status']}\n")
            lines.append(f"{a.get('verdict', '')}\n")
        if bundle.get("leak_cert_fix"):
            fix = bundle["leak_cert_fix"]
            lines.append("## LEAK-CERT fix (Round 08)\n")
            lines.append(f"- Round 07: {fix['round07_issue']}")
            lines.append(f"- Round 08: {fix['round08_fix']}")
        return "\n".join(lines)

    lines.append(f"{FRAMEWORK_ID} audit — {bundle.get('benchmark', 'unknown')}")
    lines.append("")
    for row in pm["tier_table"]:
        lines.append(
            f"  [{row['tier_id']}] {row['status']:20s}  {row['key_metric']}"
        )
    lines.append("")
    lines.append("Config gates:")
    for k, v in gates.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    for row in pm["tier_table"]:
        lines.append(f"--- {row['tier_id']} ---")
        lines.append(f"  {row['audit'].get('verdict', '')}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="QFL-PRIVACY-MAP tier auditor CLI")
    parser.add_argument(
        "--bundle",
        type=Path,
        default=RUN_ROOT / "artifacts" / "round08_metrics.json",
        help="Path to round08_metrics.json (or compatible bundle)",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Emit markdown tier table + verdicts",
    )
    parser.add_argument(
        "--tiers",
        nargs="*",
        default=None,
        help="Filter tier IDs (T1, T1p, S2-oracle, S2-wrong40)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print privacy_map JSON only",
    )
    args = parser.parse_args()

    bundle = load_bundle(args.bundle)
    if args.tiers:
        pm = bundle["privacy_map"]
        pm = {
            **pm,
            "tier_table": [r for r in pm["tier_table"] if r["tier_id"] in args.tiers],
        }
        pm["tier_table_markdown"] = tier_table_markdown(pm["tier_table"])
        bundle = {**bundle, "privacy_map": pm}

    if args.json:
        print(json.dumps(bundle["privacy_map"], indent=2))
        return

    print(format_audit_report(bundle, markdown=args.markdown))


if __name__ == "__main__":
    main()
