#!/usr/bin/env python3
"""Round-14 benchmark: Phase-2 scoped closure (REVISE_MINOR from Round 13).

- Fix TANGO-JOINT (uniform round weights) vs TANGO-DOPT (D-opt weights)
- Primary: minibatch_sgd + frozen_mlp_minibatch + label noise; median/IQR
- SHARD level3_invert tier cross via code/shard_cross_round14.py
- Secure-aggregation scope note (not implemented; claim boundary)
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np

RUN_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = RUN_ROOT / "artifacts"
LOGS = RUN_ROOT / "logs"

_spec = importlib.util.spec_from_file_location(
    "benchmark_round13", RUN_ROOT / "code" / "benchmark_round13.py"
)
_r13 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _r13
assert _spec.loader is not None
_spec.loader.exec_module(_r13)

_spec_cross = importlib.util.spec_from_file_location(
    "shard_cross_round14", RUN_ROOT / "code" / "shard_cross_round14.py"
)
_cross = importlib.util.module_from_spec(_spec_cross)
sys.modules[_spec_cross.name] = _cross
assert _spec_cross.loader is not None
_spec_cross.loader.exec_module(_cross)

SimConfig = _r13.SimConfig
make_dataset = _r13.make_dataset
make_bias_probes = _r13.make_bias_probes
observe_terminal_updates = _r13.observe_terminal_updates
observe_terminal_updates_features = _r13.observe_terminal_updates_features
frozen_mlp_features = _r13.frozen_mlp_features
evaluate_method = _r13.evaluate_method
design_condition_number = _r13.design_condition_number
effective_gradient_steps = _r13.effective_gradient_steps
scale_terminal_deltas = _r13.scale_terminal_deltas
tango_mb_estimate_sums = _r13.tango_mb_estimate_sums
passive_mb_scale_only_estimate_sums = _r13.passive_mb_scale_only_estimate_sums
tango_estimate_sums = _r13.tango_estimate_sums
tango_dopt_estimate_sums = _r13.tango_dopt_estimate_sums
tango_coupled_estimate_sums = _r13.tango_coupled_estimate_sums
joint_mb_moment_invert = _r13.joint_mb_moment_invert
apply_label_noise = _r13.apply_label_noise
aggregate_robust = _r13.aggregate_robust
within_step_weight_drift = _r13.within_step_weight_drift
public_prior_prototypes = _r13.public_prior_prototypes
oracle_aggregate_prototypes = _r13.oracle_aggregate_prototypes
tango_mb_from_scaled = _r13.tango_mb_from_scaled
LABEL_FLIP_RATE = _r13.LABEL_FLIP_RATE


def tango_joint_estimate_sums(
    delta_w: np.ndarray,
    delta_b: np.ndarray,
    probs: np.ndarray,
    cfg: SimConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TANGO-JOINT: uniform round weights (distinct from D-opt TANGO-DOPT)."""
    dw, db = scale_terminal_deltas(delta_w, delta_b, cfg)
    return joint_mb_moment_invert(dw, db, probs, cfg, round_weights=None)


SCENARIOS: dict[str, SimConfig] = {
    "minibatch_sgd": SimConfig(use_minibatch=True, minibatch_size=8, local_steps=6),
    "frozen_mlp_minibatch": SimConfig(
        use_minibatch=True, minibatch_size=8, local_steps=6
    ),
    "minibatch_label_noise": SimConfig(
        use_minibatch=True, minibatch_size=8, local_steps=6
    ),
    "minibatch_terminal_noise": SimConfig(
        use_minibatch=True,
        minibatch_size=8,
        local_steps=6,
        terminal_noise_std=0.002,
    ),
    "balanced_clean": SimConfig(),
}

METHODS = (
    "tango_mb",
    "tango_joint",
    "tango_dopt",
    "tango_coupled",
    "passive_multi_round",
    "passive_mb_scale_only",
    "oracle_aggregate",
)


def setup_logging() -> logging.Logger:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("round14")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(LOGS / "experiment_round14.log", mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def estimate_for_method(
    method: str,
    delta_w: np.ndarray,
    delta_b: np.ndarray,
    probs: np.ndarray,
    cfg: SimConfig,
    true_sums: np.ndarray,
    true_counts: np.ndarray,
    x: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if method == "tango_mb":
        proto, _, counts = tango_mb_estimate_sums(delta_w, delta_b, probs, cfg)
        return proto, counts
    if method == "tango_joint":
        proto, _, counts = tango_joint_estimate_sums(delta_w, delta_b, probs, cfg)
        return proto, counts
    if method == "tango_dopt":
        proto, _, counts = tango_dopt_estimate_sums(delta_w, delta_b, probs, cfg)
        return proto, counts
    if method == "tango_coupled":
        proto, _, counts = tango_coupled_estimate_sums(
            delta_w, delta_b, probs, cfg, x, labels, seed
        )
        return proto, counts
    if method == "passive_multi_round":
        proto, _, counts = tango_estimate_sums(delta_w, delta_b, probs, cfg)
        return proto, counts
    if method == "passive_mb_scale_only":
        proto, _, counts = passive_mb_scale_only_estimate_sums(
            delta_w, delta_b, probs, cfg
        )
        return proto, counts
    if method == "oracle_aggregate":
        return oracle_aggregate_prototypes(true_sums, true_counts), true_counts.copy()
    raise ValueError(f"unknown method: {method}")


def write_minibatch_svg(rows: list[dict[str, float | str]]) -> Path:
    path = ARTIFACTS / "round14_minibatch_methods.svg"
    width, height = 960, 460
    margin = 62
    method_keys = {
        "tango_mb": "tango_mb_prototype_mse_median",
        "tango_joint": "tango_joint_prototype_mse_median",
        "tango_dopt": "tango_dopt_prototype_mse_median",
        "passive_multi_round": "passive_multi_round_prototype_mse_median",
        "passive_mb_scale_only": "passive_mb_scale_only_prototype_mse_median",
    }
    colors = {
        "tango_mb": "#2ca02c",
        "tango_joint": "#1f77b4",
        "tango_dopt": "#ff7f0e",
        "passive_multi_round": "#d62728",
        "passive_mb_scale_only": "#8c564b",
    }
    labels = {
        "tango_mb": "TANGO-MB",
        "tango_joint": "JOINT (uniform)",
        "tango_dopt": "DOPT",
        "passive_multi_round": "passive",
        "passive_mb_scale_only": "scale-only",
    }
    scenarios = [r["scenario"] for r in rows]
    max_y = max(float(r[k]) for r in rows for k in method_keys.values()) * 1.15

    def sx(i: int) -> float:
        return margin + i * (width - 2 * margin) / max(1, len(rows) - 1)

    def sy(y: float) -> float:
        return height - margin - y / max(max_y, 1e-9) * (height - 2 * margin)

    polys = []
    for method, key in method_keys.items():
        vals = [float(r[key]) for r in rows]
        pts = " ".join(f"{sx(i):.1f},{sy(y):.1f}" for i, y in enumerate(vals))
        circles = "\n".join(
            f'<circle cx="{sx(i):.1f}" cy="{sy(y):.1f}" r="3.5" fill="{colors[method]}"/>'
            for i, y in enumerate(vals)
        )
        polys.append(
            f'<polyline fill="none" stroke="{colors[method]}" stroke-width="2.2" '
            f'points="{pts}"/>\n{circles}'
        )
    x_labels = "\n".join(
        f'<text x="{sx(i):.1f}" y="{height-30}" transform="rotate(22 {sx(i):.1f},{height-30})" '
        f'text-anchor="start" font-size="10">{name}</text>'
        for i, name in enumerate(scenarios)
    )
    legend = "\n".join(
        f'<circle cx="{width-250}" cy="{54+i*18}" r="4" fill="{colors[m]}"/>'
        f'<text x="{width-238}" y="{58+i*18}" font-size="10">{labels[m]}</text>'
        for i, m in enumerate(method_keys)
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2:.1f}" y="22" text-anchor="middle" font-size="14" font-family="sans-serif">Round 14: prototype MSE median (JOINT uniform vs DOPT)</text>
<line x1="{margin}" x2="{width-margin}" y1="{height-margin}" y2="{height-margin}" stroke="#333"/>
<line x1="{margin}" x2="{margin}" y1="{margin}" y2="{height-margin}" stroke="#333"/>
{x_labels}
{''.join(polys)}
{legend}
<text transform="translate(14,{height/2:.1f}) rotate(-90)" text-anchor="middle" font-size="12">prototype MSE (median)</text>
</svg>"""
    path.write_text(svg)
    return path


def run() -> dict[str, object]:
    logger = setup_logging()
    started = time.time()
    logger.info("Round 14 Phase-2 scoped closure benchmark started")

    per_seed: list[dict[str, float | str]] = []
    drift_samples: list[float] = []

    for scenario, cfg in SCENARIOS.items():
        eff = effective_gradient_steps(cfg)
        logger.info("scenario=%s effective_gradient_steps=%.4f", scenario, eff)
        for seed in cfg.seeds:
            x_raw, labels_raw, _ = make_dataset(cfg, seed)
            labels = labels_raw
            if scenario == "minibatch_label_noise":
                labels = apply_label_noise(labels_raw, cfg, seed, LABEL_FLIP_RATE)

            x = x_raw
            if scenario == "frozen_mlp_minibatch":
                x = frozen_mlp_features(x_raw, seed, hidden_dim=24)

            if scenario == "minibatch_sgd" and cfg.use_minibatch:
                drift_samples.append(within_step_weight_drift(x, labels, cfg, seed))

            true_counts = np.asarray(cfg.class_counts, dtype=np.float64)
            true_sums = np.vstack(
                [x[labels == c].sum(axis=0) for c in range(cfg.n_classes)]
            )
            true_proto = true_sums / true_counts[:, None]
            within_class_var = float(
                np.mean(
                    [
                        np.mean((x[labels == c] - true_proto[c]) ** 2)
                        for c in range(cfg.n_classes)
                    ]
                )
            )
            n_rounds = max(cfg.probe_rounds)

            biases_active = make_bias_probes(cfg, seed, n_rounds, passive=False)
            if scenario == "frozen_mlp_minibatch":
                dw_a, db_a, probs_a = observe_terminal_updates_features(
                    x, labels, biases_active, cfg, seed
                )
            else:
                dw_a, db_a, probs_a = observe_terminal_updates(
                    x, labels, biases_active, cfg, seed
                )

            biases_passive = make_bias_probes(cfg, seed, n_rounds, passive=True)
            if scenario == "frozen_mlp_minibatch":
                dw_p, db_p, probs_p = observe_terminal_updates_features(
                    x, labels, biases_passive, cfg, seed + 1
                )
            else:
                dw_p, db_p, probs_p = observe_terminal_updates(
                    x, labels, biases_passive, cfg, seed + 1
                )

            obs = {
                "tango_mb": (dw_a, db_a, probs_a),
                "tango_joint": (dw_a, db_a, probs_a),
                "tango_dopt": (dw_a, db_a, probs_a),
                "tango_coupled": (dw_a, db_a, probs_a),
                "passive_multi_round": (dw_p, db_p, probs_p),
                "passive_mb_scale_only": (dw_p, db_p, probs_p),
                "oracle_aggregate": (dw_a, db_a, probs_a),
            }

            for method in METHODS:
                dw, db, probs = obs[method]
                est_proto, est_counts = estimate_for_method(
                    method,
                    dw,
                    db,
                    probs,
                    cfg,
                    true_sums,
                    true_counts,
                    x,
                    labels,
                    seed,
                )
                metrics = evaluate_method(
                    method,
                    est_proto,
                    est_counts,
                    true_proto,
                    true_counts,
                    true_sums,
                    x,
                    labels,
                    cfg,
                )
                row = {
                    "scenario": scenario,
                    "seed": float(seed),
                    "probe_rounds": float(n_rounds),
                    "effective_gradient_steps": eff,
                    "use_minibatch": float(cfg.use_minibatch),
                    "observed_intermediate_batch_gradients": 0.0,
                    "within_class_variance": within_class_var,
                    "design_condition_number": design_condition_number(probs_a, cfg),
                    **metrics,
                }
                per_seed.append(row)
                logger.info(
                    "scenario=%s seed=%s method=%s count_mae=%.4f proto_mse=%.6f",
                    scenario,
                    seed,
                    method,
                    row["count_mae"],
                    row["prototype_mse"],
                )

    metric_keys = [
        "count_mae",
        "count_relative_error",
        "prototype_mse",
        "dataset_mean_mse",
        "individual_mse_from_prototypes",
        "within_class_variance",
        "design_condition_number",
        "effective_gradient_steps",
    ]
    by_scenario_method: list[dict[str, float | str]] = []
    for scenario in SCENARIOS:
        for method in METHODS:
            rows = [
                r
                for r in per_seed
                if r["scenario"] == scenario and r["method"] == method
            ]
            agg = aggregate_robust(rows, metric_keys)
            agg["scenario"] = scenario
            agg["method"] = method
            by_scenario_method.append(agg)

    scenario_method_headlines: list[dict[str, float | str]] = []
    for scenario in SCENARIOS:
        rows = [r for r in by_scenario_method if r["scenario"] == scenario]
        entry: dict[str, float | str] = {"scenario": scenario}
        for method in METHODS:
            mrow = next(r for r in rows if r["method"] == method)
            entry[f"{method}_prototype_mse_mean"] = mrow["prototype_mse_mean"]
            entry[f"{method}_prototype_mse_median"] = mrow["prototype_mse_median"]
            entry[f"{method}_prototype_mse_iqr"] = mrow["prototype_mse_iqr"]
            entry[f"{method}_count_mae_mean"] = mrow["count_mae_mean"]
            entry[f"{method}_count_mae_median"] = mrow["count_mae_median"]
        passive_mse = float(entry["passive_multi_round_prototype_mse_mean"])
        tango_mb_mse = float(entry["tango_mb_prototype_mse_mean"])
        scale_only_mse = float(
            entry["passive_mb_scale_only_prototype_mse_mean"]
        )
        joint_mse = float(entry["tango_joint_prototype_mse_mean"])
        dopt_mse = float(entry["tango_dopt_prototype_mse_mean"])
        entry["tango_mb_vs_passive_gain_x"] = passive_mse / max(tango_mb_mse, 1e-12)
        entry["phase2_primary_win"] = float(tango_mb_mse < passive_mse)
        entry["active_over_r11_scaling_only_x"] = scale_only_mse / max(
            tango_mb_mse, 1e-12
        )
        entry["joint_vs_dopt_identical"] = float(
            abs(joint_mse - dopt_mse) < 1e-9
        )
        entry["dopt_beats_joint"] = float(dopt_mse < joint_mse)
        entry["joint_beats_dopt"] = float(joint_mse < dopt_mse)
        scenario_method_headlines.append(entry)

    mb_rows = [
        r
        for r in scenario_method_headlines
        if "minibatch" in str(r["scenario"]) or r["scenario"] == "frozen_mlp_minibatch"
    ]
    plot_path = write_minibatch_svg(mb_rows)

    primary = next(
        r for r in scenario_method_headlines if r["scenario"] == "minibatch_sgd"
    )

    logger.info("Running SHARD tier cross (synthetic level3_invert)...")
    shard_cross = _cross.run_cross(logger=logger)

    drift_mean = float(np.mean(drift_samples)) if drift_samples else 0.0
    drift_std = float(np.std(drift_samples)) if drift_samples else 0.0

    secure_agg_note = (
        "Secure aggregation (masked sum of client updates) is out of scope for "
        "this simulator: TANGO-MB assumes the server observes per-client terminal "
        "deltas from active probes, not only a global aggregate. A masked global "
        "sum would hide cross-client structure needed for multi-round design "
        "inversion; we do not claim robustness to secure agg without additional "
        "side information or multiple non-colluding observers."
    )

    metrics: dict[str, object] = {
        "benchmark": "round14_phase2_scoped_closure",
        "phase2_goal": (
            "Close REVISE_MINOR: SHARD tier cross, JOINT vs DOPT fix, "
            "scoped acceptance package, frozen_mlp+label noise, median/IQR"
        ),
        "joint_vs_dopt_fix": {
            "tango_joint": "uniform round weights (round_weights=None)",
            "tango_dopt": "D-opt weights w_r propto 1/||p-uniform||",
            "primary_joint_mse_mean": primary["tango_joint_prototype_mse_mean"],
            "primary_dopt_mse_mean": primary["tango_dopt_prototype_mse_mean"],
            "primary_joint_mse_median": primary["tango_joint_prototype_mse_median"],
            "primary_dopt_mse_median": primary["tango_dopt_prototype_mse_median"],
            "identical_on_primary": primary["joint_vs_dopt_identical"],
            "dopt_beats_joint": primary["dopt_beats_joint"],
        },
        "secure_aggregation_note": secure_agg_note,
        "shard_baseline_cross": shard_cross,
        "lemma_mb_b_empirical": {
            "within_step_weight_drift_mean": drift_mean,
            "within_step_weight_drift_std": drift_std,
        },
        "scenario_configs": {name: asdict(cfg) for name, cfg in SCENARIOS.items()},
        "label_flip_rate": LABEL_FLIP_RATE,
        "per_seed": per_seed,
        "aggregate_by_scenario_method": by_scenario_method,
        "scenario_method_headlines": scenario_method_headlines,
        "phase2_primary": {
            "scenario": "minibatch_sgd",
            "passive_prototype_mse_mean": primary[
                "passive_multi_round_prototype_mse_mean"
            ],
            "passive_prototype_mse_median": primary[
                "passive_multi_round_prototype_mse_median"
            ],
            "passive_mb_scale_only_prototype_mse_mean": primary[
                "passive_mb_scale_only_prototype_mse_mean"
            ],
            "passive_mb_scale_only_prototype_mse_median": primary[
                "passive_mb_scale_only_prototype_mse_median"
            ],
            "passive_mb_scale_only_prototype_mse_iqr": primary[
                "passive_mb_scale_only_prototype_mse_iqr"
            ],
            "tango_mb_prototype_mse_mean": primary["tango_mb_prototype_mse_mean"],
            "tango_mb_prototype_mse_median": primary["tango_mb_prototype_mse_median"],
            "tango_mb_prototype_mse_iqr": primary["tango_mb_prototype_mse_iqr"],
            "tango_joint_prototype_mse_mean": primary[
                "tango_joint_prototype_mse_mean"
            ],
            "tango_joint_prototype_mse_median": primary[
                "tango_joint_prototype_mse_median"
            ],
            "tango_joint_prototype_mse_iqr": primary[
                "tango_joint_prototype_mse_iqr"
            ],
            "tango_dopt_prototype_mse_mean": primary["tango_dopt_prototype_mse_mean"],
            "tango_dopt_prototype_mse_median": primary[
                "tango_dopt_prototype_mse_median"
            ],
            "tango_dopt_prototype_mse_iqr": primary["tango_dopt_prototype_mse_iqr"],
            "tango_coupled_prototype_mse_mean": primary[
                "tango_coupled_prototype_mse_mean"
            ],
            "tango_mb_count_mae_mean": primary["tango_mb_count_mae_mean"],
            "tango_joint_count_mae_mean": primary["tango_joint_count_mae_mean"],
            "tango_dopt_count_mae_mean": primary["tango_dopt_count_mae_mean"],
            "tango_mb_vs_passive_gain_x": primary["tango_mb_vs_passive_gain_x"],
            "active_over_r11_scaling_only_x": primary[
                "active_over_r11_scaling_only_x"
            ],
            "phase2_primary_win": primary["phase2_primary_win"],
            "joint_vs_dopt_identical": primary["joint_vs_dopt_identical"],
            "dopt_beats_joint": primary["dopt_beats_joint"],
        },
        "phase2_scoped_accept": {
            "numeric_gate_minibatch_primary_win": bool(
                float(primary["phase2_primary_win"]) > 0.5
            ),
            "shard_baseline_addressed": True,
            "joint_dopt_distinct": bool(
                float(primary["joint_vs_dopt_identical"]) < 0.5
            ),
            "claim_scope_doc": "paper/phase2_scope.md",
        },
        "headline": {
            "minibatch_tango_mb_mse_median": primary["tango_mb_prototype_mse_median"],
            "minibatch_tango_joint_mse_median": primary[
                "tango_joint_prototype_mse_median"
            ],
            "minibatch_tango_dopt_mse_median": primary[
                "tango_dopt_prototype_mse_median"
            ],
            "minibatch_phase2_win": primary["phase2_primary_win"],
            "shard_level3_reconstruction_mse": shard_cross["shard_intermediate"][
                "level3_reconstruction_mse"
            ],
        },
        "plots": [str(plot_path.relative_to(RUN_ROOT))],
        "runtime_seconds": time.time() - started,
    }

    out_path = ARTIFACTS / "round14_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    logger.info("wrote %s", out_path)
    logger.info("phase2_primary=%s", metrics["phase2_primary"])
    logger.info("joint_vs_dopt_fix=%s", metrics["joint_vs_dopt_fix"])
    logger.info("shard_baseline_cross keys=%s", list(shard_cross.keys()))
    return metrics


if __name__ == "__main__":
    run()
