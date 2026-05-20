"""QFL-PRIVACY-MAP — shared tier definitions and audit helpers (Round 08+)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

FRAMEWORK_ID = "QFL-PRIVACY-MAP"
THEOREM_ABT1 = "ABT-1"


@dataclass(frozen=True)
class PrivacyTier:
    tier_id: str
    observation_class: str
    defender_posture: str
    auditor_channel: str
    status: str
    evidence_ref: str


PRIVACY_TIERS: tuple[PrivacyTier, ...] = (
    PrivacyTier(
        tier_id="T1",
        observation_class="Epoch-terminal-only (LASA-QTERM T1)",
        defender_posture="Publish only per-epoch terminal gradients; no batch rows.",
        auditor_channel="LASA-QTERM T1 + passive mean broadcast",
        status="IMPOSSIBLE",
        evidence_ref="qfl-terminal-snapshot/paper/impossibility_t1.md",
    ),
    PrivacyTier(
        tier_id="T1p",
        observation_class="Partial terminal rows (p<K per epoch)",
        defender_posture="Limit published terminal rows; certify leak vs honest naive.",
        auditor_channel="LEAK-CERT-T1p v2 (Ky-Fan tail + trace floor)",
        status="CERTIFIED_BOUND",
        evidence_ref="code/benchmark_round08.py::run_leak_cert_t1p_honest",
    ),
    PrivacyTier(
        tier_id="S2-oracle",
        observation_class="Stage-2 batch means + oracle incidence + chain graph",
        defender_posture="Conditional: hide rows only when assignment/graph are trusted.",
        auditor_channel="GARD-SPARSE co-occurrence MAP",
        status="CONDITIONAL_THREAT",
        evidence_ref="round01 GARD-SPARSE @ 15/60 rows (75% reduction)",
    ),
    PrivacyTier(
        tier_id="S2-wrong40",
        observation_class="Stage-2 batch means + wrong40 incidence",
        defender_posture="Assignment hiding / secure batch indexing required.",
        auditor_channel="GARD + JASPER-Q (level1 anchor)",
        status="BARRIER",
        evidence_ref="paper/assignment_barrier_theorem.md (ABT-1)",
    ),
)


def tier_table_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Tier | Observation | Defender | Auditor | Status | Key metric |",
        "|------|-------------|----------|---------|--------|------------|",
    ]
    for r in rows:
        lines.append(
            f"| **{r['tier_id']}** | {r['observation_class']} | "
            f"{r['defender_posture_short']} | {r['auditor_channel_short']} | "
            f"{r['status']} | {r['key_metric']} |"
        )
    return "\n".join(lines)


def audit_t1_impossibility(t1_mse_mean: float, *, tol: float = 0.15) -> dict[str, Any]:
    """T1: individual recovery impossible when MSE ≈ population variance (broadcast)."""
    impossible = t1_mse_mean >= 1.0 - tol
    return {
        "tier_id": "T1",
        "status": "IMPOSSIBLE" if impossible else "REVIEW",
        "t1_mse_mean": t1_mse_mean,
        "pass": impossible,
        "verdict": (
            "Terminal-only map depends only on dataset mean; per-sample snapshots not identifiable."
            if impossible
            else f"T1 MSE {t1_mse_mean:.4g} below impossibility floor — check probe setup."
        ),
    }


def audit_t1p_leak_cert(
    per_run: list[dict[str, Any]],
    *,
    use_honest_naive: bool = True,
) -> dict[str, Any]:
    """T1p LEAK-CERT v2 — criterion C uses broadcast mean only (honest naive)."""
    naive_key = "cert_naive_broadcast" if use_honest_naive else "cert_naive_trace_inflated"
    ratio_key = (
        "naive_broadcast_over_cert"
        if use_honest_naive
        else "naive_trace_over_cert"
    )
    ratios = [r[ratio_key] for r in per_run]
    covers = [r["cert_covers_empirical"] for r in per_run]
    emp_mean = float(np.mean([r["empirical_t1p_mse"] for r in per_run]))
    cert_mean = float(np.mean([r["cert_tight_upper_bound"] for r in per_run]))
    naive_mean = float(np.mean([r[naive_key] for r in per_run]))
    ratio_mean = float(np.mean(ratios))
    pass_2x = all(r >= 2.0 for r in ratios)
    pass_cover = all(covers)
    gaming_flag = False
    if use_honest_naive and per_run and "naive_trace_over_cert" in per_run[0]:
        trace_ratios = [r["naive_trace_over_cert"] for r in per_run]
        gaming_flag = float(np.mean(trace_ratios)) >= 2.0 and not pass_2x

    return {
        "tier_id": "T1p",
        "status": "CERTIFIED_BOUND" if pass_2x and pass_cover else "CERT_WEAK",
        "empirical_t1p_mse_mean": emp_mean,
        "cert_tight_mean": cert_mean,
        "naive_baseline_mean": naive_mean,
        "naive_over_cert_ratio_mean": ratio_mean,
        "criterion_c_2x_honest_naive": pass_2x,
        "cert_covers_empirical_all": pass_cover,
        "round07_trace_inflated_gaming_detected": gaming_flag,
        "pass": pass_2x and pass_cover,
        "verdict": (
            "Honest naive (broadcast mean) satisfies ≥2× cert tightening with coverage."
            if pass_2x and pass_cover
            else (
                "Criterion C fails under honest naive; Round 07 trace-inflated baseline was gaming."
                if gaming_flag
                else "Certificate does not beat honest naive by 2× on all seeds."
            )
        ),
        "per_run": per_run,
    }


def audit_gard_sparse_oracle(
    per_run: list[dict[str, Any]],
    *,
    target_mse: float = 0.10,
    fraction: float = 0.25,
) -> dict[str, Any]:
    """Oracle GARD-SPARSE conditional threat at 75% row reduction (15/60 rows)."""
    subset = [r for r in per_run if abs(r["fraction"] - fraction) < 1e-9]
    mses = [r["gard_sparse_mse"] for r in subset]
    rows = [r["observed_rows"] for r in subset]
    full_rows = subset[0]["full_rows"] if subset else 60
    mean_mse = float(np.mean(mses)) if mses else float("nan")
    reduction = 1.0 - (int(np.mean(rows)) / full_rows) if rows else None
    reaches = mean_mse <= target_mse if mses else False
    return {
        "tier_id": "S2-oracle",
        "status": "CONDITIONAL_THREAT" if reaches else "THREAT_WEAK",
        "fraction": fraction,
        "observed_rows_mean": int(round(np.mean(rows))) if rows else None,
        "full_rows": full_rows,
        "row_reduction_vs_full": reduction,
        "gard_sparse_mse_mean": mean_mse,
        "target_mse": target_mse,
        "parent_mean_reaches_target": reaches,
        "pass": reaches and reduction is not None and reduction >= 0.75,
        "verdict": (
            f"Oracle GARD-SPARSE @ {fraction:.0%} rows: mean MSE {mean_mse:.4g} "
            f"({'≤' if reaches else '>'}) {target_mse}; "
            f"{reduction * 100:.0f}% row reduction vs full."
            if reduction is not None
            else "No oracle GARD runs at target fraction."
        ),
        "per_run": subset,
    }


def audit_wrong40_barrier(
    gard_runs: list[dict[str, Any]],
    jasper_runs: list[dict[str, Any]],
    *,
    fractions: tuple[float, ...] = (0.25, 0.50),
    floor_mse: float = 0.15,
) -> dict[str, Any]:
    """ABT-1 assignment barrier under wrong40."""
    checks = []
    for frac in fractions:
        g_sub = [r for r in gard_runs if abs(r["fraction"] - frac) < 1e-9]
        j_sub = [r for r in jasper_runs if abs(r["fraction"] - frac) < 1e-9]
        g_mean = float(np.mean([r["snapshot_mse"] for r in g_sub])) if g_sub else float("nan")
        j_mean = float(np.mean([r["snapshot_mse"] for r in j_sub])) if j_sub else float("nan")
        barrier_holds = g_mean >= floor_mse and j_mean >= floor_mse
        checks.append(
            {
                "fraction": frac,
                "gard_mean_mse": g_mean,
                "jasper_mean_mse": j_mean,
                "abt1_floor": floor_mse,
                "barrier_holds": barrier_holds,
            }
        )
    all_hold = all(c["barrier_holds"] for c in checks)
    return {
        "tier_id": "S2-wrong40",
        "theorem_id": THEOREM_ABT1,
        "status": "BARRIER" if all_hold else "BARRIER_BROKEN",
        "assignment_regime": "wrong40",
        "fraction_checks": checks,
        "pass": all_hold,
        "verdict": (
            "ABT-1 floor holds: wrong incidence blocks snapshot MSE < 0.15 at tested fractions."
            if all_hold
            else "Barrier violated — review incidence corruption or attacker class."
        ),
    }


def build_privacy_map_payload(audits: dict[str, Any]) -> dict[str, Any]:
    """Assemble tier table rows for JSON + markdown export."""
    t1 = audits["T1"]
    t1p = audits["T1p"]
    gard = audits["S2-oracle"]
    w40 = audits["S2-wrong40"]

    tier_rows = [
        {
            "tier_id": "T1",
            "observation_class": PRIVACY_TIERS[0].observation_class,
            "defender_posture_short": "Terminal-only publish",
            "auditor_channel_short": "LASA-QTERM T1",
            "status": t1["status"],
            "key_metric": f"T1 MSE mean={t1['t1_mse_mean']:.3f}",
            "audit": t1,
        },
        {
            "tier_id": "T1p",
            "observation_class": PRIVACY_TIERS[1].observation_class,
            "defender_posture_short": "Partial terminal rows",
            "auditor_channel_short": "LEAK-CERT v2 (honest naive)",
            "status": t1p["status"],
            "key_metric": (
                f"naive/cert={t1p['naive_over_cert_ratio_mean']:.2f}× "
                f"(emp={t1p['empirical_t1p_mse_mean']:.3f})"
            ),
            "audit": t1p,
        },
        {
            "tier_id": "S2-oracle",
            "observation_class": PRIVACY_TIERS[2].observation_class,
            "defender_posture_short": "Oracle incidence + graph",
            "auditor_channel_short": "GARD-SPARSE",
            "status": gard["status"],
            "key_metric": (
                f"GARD MSE={gard['gard_sparse_mse_mean']:.3f} @ "
                f"{gard['row_reduction_vs_full'] * 100:.0f}% reduction"
                if gard.get("row_reduction_vs_full") is not None
                else "n/a"
            ),
            "audit": gard,
        },
        {
            "tier_id": "S2-wrong40",
            "observation_class": PRIVACY_TIERS[3].observation_class,
            "defender_posture_short": "Hide batch assignment",
            "auditor_channel_short": "GARD + JASPER-Q",
            "status": w40["status"],
            "key_metric": (
                f"JASPER mean @50%={next(c['jasper_mean_mse'] for c in w40['fraction_checks'] if c['fraction'] == 0.5):.3f}"
                if any(c["fraction"] == 0.5 for c in w40["fraction_checks"])
                else "n/a"
            ),
            "audit": w40,
        },
    ]

    config_c = t1p["criterion_c_2x_honest_naive"] and t1p["cert_covers_empirical_all"]
    config_a = gard["pass"]

    return {
        "framework_id": FRAMEWORK_ID,
        "tier_table": tier_rows,
        "tier_table_markdown": tier_table_markdown(tier_rows),
        "config_breakthrough": {
            "A_gard_oracle_75pct_reduction_at_mse_0.10": config_a,
            "C_leak_cert_2x_honest_naive": config_c,
            "T1_impossibility_confirmed": t1["pass"],
            "ABT1_wrong40_barrier": w40["pass"],
        },
        "audits": audits,
    }
