#!/usr/bin/env python3
"""Round-11 benchmark: minibatch-realistic TANGO (Phase 2).

Primary evaluation uses minibatch SGD (shuffled batches, multiple local steps).
Implements:
- TANGO vanilla (Round-09 first-order full-batch assumption) — expected to fail
- TANGO-MB: minibatch gradient-moment scaling correction
- STORM: stacked terminal moment system with effective-step scaling
- Passive multi-round baseline (with and without MB correction)

Success target (config.json phase2): TANGO-MB prototype MSE < passive on minibatch_sgd.
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
    "benchmark_round09", RUN_ROOT / "code" / "benchmark_round09.py"
)
_r09 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _r09
assert _spec.loader is not None
_spec.loader.exec_module(_r09)

SimConfig = _r09.SimConfig
make_dataset = _r09.make_dataset
make_bias_probes = _r09.make_bias_probes
observe_terminal_updates = _r09.observe_terminal_updates
tango_estimate_sums = _r09.tango_estimate_sums
public_prior_prototypes = _r09.public_prior_prototypes
oracle_aggregate_prototypes = _r09.oracle_aggregate_prototypes
evaluate_method = _r09.evaluate_method
aggregate = _r09.aggregate
mean_squared = _r09.mean_squared
design_condition_number = _r09.design_condition_number


# Primary Phase-2 scenario first; balanced retained for regression.
SCENARIOS: dict[str, SimConfig] = {
    "minibatch_sgd": SimConfig(use_minibatch=True, minibatch_size=8, local_steps=6),
    "minibatch_nonzero_init": SimConfig(
        use_minibatch=True, minibatch_size=8, local_steps=6, init_weight_scale=0.18
    ),
    "balanced_clean": SimConfig(),
    "imbalanced_clean": SimConfig(class_counts=(4, 8, 12)),
}


def setup_logging() -> logging.Logger:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("round11")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(LOGS / "experiment_round11.log", mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def effective_gradient_steps(cfg: SimConfig) -> float:
    """Expected terminal-gradient scale under w0≈0 and one shuffled pass per local step.

    Each local step applies n/B minibatch updates; at w=0 their sum matches (n/B)×
    a single full-batch step. Over T local steps the terminal delta scales by
    T·(N/B) rather than T assumed by vanilla TANGO.
    """
    if not cfg.use_minibatch:
        return float(cfg.local_steps)
    batches_per_step = cfg.n_samples / cfg.minibatch_size
    return float(cfg.local_steps * batches_per_step)


def scale_terminal_deltas(
    delta_w: np.ndarray, delta_b: np.ndarray, cfg: SimConfig
) -> tuple[np.ndarray, np.ndarray]:
    eff = effective_gradient_steps(cfg)
    if eff <= 0:
        return delta_w, delta_b
    scale = cfg.local_steps / eff
    return delta_w * scale, delta_b * scale


def tango_mb_estimate_sums(
    delta_w: np.ndarray,
    delta_b: np.ndarray,
    probs: np.ndarray,
    cfg: SimConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dw, db = scale_terminal_deltas(delta_w, delta_b, cfg)
    return tango_estimate_sums(dw, db, probs, cfg)


def storm_estimate_sums(
    delta_w: np.ndarray,
    delta_b: np.ndarray,
    probs: np.ndarray,
    cfg: SimConfig,
    ridge: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """STORM: Stochastic Terminal Observation Recovered Moments.

    Same linear moment system as TANGO but divides terminal deltas by
    η·T_eff with T_eff = T·(N/B) under minibatch SGD. Optional ridge on the
    stacked probe design for numerical stability (default 0).
    """
    eff = effective_gradient_steps(cfg)
    avg_grad_w = -delta_w / (cfg.lr * eff)
    avg_grad_b = -delta_b / (cfg.lr * eff)
    counts_per_round = cfg.n_samples * (probs - avg_grad_b)
    counts = np.clip(counts_per_round.mean(axis=0), 1.0, cfg.n_samples)

    class_sums = np.zeros((cfg.n_classes, cfg.dim_x), dtype=np.float64)
    total = np.zeros(cfg.dim_x, dtype=np.float64)
    n_vars = cfg.n_classes + 1
    for k in range(cfg.dim_x):
        rows: list[np.ndarray] = []
        ys: list[float] = []
        for r in range(delta_w.shape[0]):
            for c in range(cfg.n_classes):
                row = np.zeros(n_vars, dtype=np.float64)
                row[0] = probs[r, c]
                row[1 + c] = -1.0
                rows.append(row)
                ys.append(cfg.n_samples * avg_grad_w[r, k, c])
        row = np.zeros(n_vars, dtype=np.float64)
        row[0] = 1.0
        row[1:] = -1.0
        rows.append(10.0 * row)
        ys.append(0.0)
        a_mat = np.vstack(rows)
        y_vec = np.asarray(ys, dtype=np.float64)
        if ridge > 0:
            sol = np.linalg.solve(
                a_mat.T @ a_mat + ridge * np.eye(n_vars), a_mat.T @ y_vec
            )
        else:
            sol, *_ = np.linalg.lstsq(a_mat, y_vec, rcond=None)
        total[k] = sol[0]
        class_sums[:, k] = sol[1:]
    prototypes = class_sums / counts[:, None]
    return prototypes, class_sums, counts


METHODS = (
    "tango_vanilla",
    "tango_mb",
    "storm",
    "passive_multi_round",
    "passive_mb",
    "public_prior",
    "oracle_aggregate",
)


def estimate_for_method(
    method: str,
    delta_w: np.ndarray,
    delta_b: np.ndarray,
    probs: np.ndarray,
    cfg: SimConfig,
    true_sums: np.ndarray,
    true_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if method == "tango_vanilla":
        proto, _, counts = tango_estimate_sums(delta_w, delta_b, probs, cfg)
        return proto, counts
    if method == "tango_mb":
        proto, _, counts = tango_mb_estimate_sums(delta_w, delta_b, probs, cfg)
        return proto, counts
    if method == "storm":
        proto, _, counts = storm_estimate_sums(delta_w, delta_b, probs, cfg)
        return proto, counts
    if method == "passive_multi_round":
        proto, _, counts = tango_estimate_sums(delta_w, delta_b, probs, cfg)
        return proto, counts
    if method == "passive_mb":
        dw, db = scale_terminal_deltas(delta_w, delta_b, cfg)
        proto, _, counts = tango_estimate_sums(dw, db, probs, cfg)
        return proto, counts
    if method == "public_prior":
        return public_prior_prototypes(delta_w[:1], probs[:1], cfg), np.full(
            cfg.n_classes, cfg.n_samples / cfg.n_classes
        )
    if method == "oracle_aggregate":
        return oracle_aggregate_prototypes(true_sums, true_counts), true_counts.copy()
    raise ValueError(f"unknown method: {method}")


def write_minibatch_svg(rows: list[dict[str, float | str]]) -> Path:
    path = ARTIFACTS / "round11_minibatch_methods.svg"
    width, height = 880, 420
    margin = 62
    method_keys = {
        "tango_vanilla": "tango_vanilla_prototype_mse_mean",
        "tango_mb": "tango_mb_prototype_mse_mean",
        "storm": "storm_prototype_mse_mean",
        "passive_multi_round": "passive_multi_round_prototype_mse_mean",
        "passive_mb": "passive_mb_prototype_mse_mean",
    }
    colors = {
        "tango_vanilla": "#d62728",
        "tango_mb": "#2ca02c",
        "storm": "#17becf",
        "passive_multi_round": "#ff7f0e",
        "passive_mb": "#9467bd",
    }
    labels = {
        "tango_vanilla": "TANGO vanilla",
        "tango_mb": "TANGO-MB",
        "storm": "STORM",
        "passive_multi_round": "passive",
        "passive_mb": "passive+MB",
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
        f'<circle cx="{width-220}" cy="{54+i*18}" r="4" fill="{colors[m]}"/>'
        f'<text x="{width-208}" y="{58+i*18}" font-size="10">{labels[m]}</text>'
        for i, m in enumerate(method_keys)
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2:.1f}" y="22" text-anchor="middle" font-size="14" font-family="sans-serif">Round 11: prototype MSE — minibatch-primary methods</text>
<line x1="{margin}" x2="{width-margin}" y1="{height-margin}" y2="{height-margin}" stroke="#333"/>
<line x1="{margin}" x2="{margin}" y1="{margin}" y2="{height-margin}" stroke="#333"/>
{x_labels}
{''.join(polys)}
{legend}
<text transform="translate(14,{height/2:.1f}) rotate(-90)" text-anchor="middle" font-size="12">prototype MSE</text>
</svg>"""
    path.write_text(svg)
    return path


def run() -> dict[str, object]:
    logger = setup_logging()
    started = time.time()
    logger.info("Round 11 Phase-2 minibatch-primary benchmark started")

    per_seed: list[dict[str, float | str]] = []

    for scenario, cfg in SCENARIOS.items():
        eff = effective_gradient_steps(cfg)
        logger.info(
            "scenario=%s config=%s effective_gradient_steps=%.4f",
            scenario,
            asdict(cfg),
            eff,
        )
        for seed in cfg.seeds:
            x, labels, _ = make_dataset(cfg, seed)
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
            dw_a, db_a, probs_a = observe_terminal_updates(
                x, labels, biases_active, cfg, seed
            )
            biases_passive = make_bias_probes(cfg, seed, n_rounds, passive=True)
            dw_p, db_p, probs_p = observe_terminal_updates(
                x, labels, biases_passive, cfg, seed + 1
            )

            obs = {
                "tango_vanilla": (dw_a, db_a, probs_a),
                "tango_mb": (dw_a, db_a, probs_a),
                "storm": (dw_a, db_a, probs_a),
                "passive_multi_round": (dw_p, db_p, probs_p),
                "passive_mb": (dw_p, db_p, probs_p),
                "public_prior": (dw_p, db_p, probs_p),
                "oracle_aggregate": (dw_a, db_a, probs_a),
            }

            for method in METHODS:
                dw, db, probs = obs[method]
                est_proto, est_counts = estimate_for_method(
                    method, dw, db, probs, cfg, true_sums, true_counts
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
            agg = aggregate(rows, metric_keys)
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
            entry[f"{method}_count_mae_mean"] = mrow["count_mae_mean"]
        passive_mse = float(entry["passive_multi_round_prototype_mse_mean"])
        tango_mb_mse = float(entry["tango_mb_prototype_mse_mean"])
        entry["tango_mb_vs_passive_gain_x"] = passive_mse / max(tango_mb_mse, 1e-12)
        entry["phase2_primary_win"] = float(tango_mb_mse < passive_mse)
        scenario_method_headlines.append(entry)

    mb_rows = [r for r in scenario_method_headlines if "minibatch" in str(r["scenario"])]
    plot_path = write_minibatch_svg(mb_rows)

    primary = next(
        r for r in scenario_method_headlines if r["scenario"] == "minibatch_sgd"
    )
    balanced = next(
        r for r in scenario_method_headlines if r["scenario"] == "balanced_clean"
    )

    metrics: dict[str, object] = {
        "benchmark": "round11_phase2_minibatch_primary",
        "phase2_goal": (
            "Fix TANGO under minibatch SGD; beat passive on minibatch_sgd prototype MSE"
        ),
        "methods": {
            "tango_vanilla": "Round-09 first-order full-batch terminal scaling",
            "tango_mb": (
                "Minibatch moment scaling: divide terminal deltas by T_eff=T*(N/B) "
                "before TANGO inversion (w0 linearization)"
            ),
            "storm": (
                "Stacked terminal moment system with same T_eff; optional ridge (0 default)"
            ),
            "passive_multi_round": "Neutral bias probes, vanilla scaling",
            "passive_mb": "Neutral bias probes with TANGO-MB scaling",
        },
        "effective_gradient_steps_formula": (
            "T_eff = local_steps * (n_samples / minibatch_size) when use_minibatch; "
            "else T_eff = local_steps"
        ),
        "observation_model": {
            "observed": [
                "terminal client model deltas per round",
                "server-chosen classifier biases",
                "public optimizer hyperparameters (lr, local_steps, batch_size, n)",
            ],
            "not_observed": [
                "intermediate minibatch gradients",
                "batch order or batch incidence",
            ],
        },
        "theorem_scope": {
            "exact_full_batch": "Round-09/10 linear full-batch tier unchanged",
            "minibatch_corrected": (
                "TANGO-MB/STORM: first-order terminal model with effective gradient "
                "steps T*(N/B) under w0≈0 and one full client pass per local step"
            ),
            "residual_approximation": [
                "weight drift within multi-minibatch local steps",
                "nonzero init (partially corrected empirically)",
            ],
        },
        "scenario_configs": {name: asdict(cfg) for name, cfg in SCENARIOS.items()},
        "per_seed": per_seed,
        "aggregate_by_scenario_method": by_scenario_method,
        "scenario_method_headlines": scenario_method_headlines,
        "phase2_primary": {
            "scenario": "minibatch_sgd",
            "passive_prototype_mse": primary["passive_multi_round_prototype_mse_mean"],
            "tango_vanilla_prototype_mse": primary["tango_vanilla_prototype_mse_mean"],
            "tango_mb_prototype_mse": primary["tango_mb_prototype_mse_mean"],
            "storm_prototype_mse": primary["storm_prototype_mse_mean"],
            "passive_mb_prototype_mse": primary["passive_mb_prototype_mse_mean"],
            "tango_mb_vs_passive_gain_x": primary["tango_mb_vs_passive_gain_x"],
            "phase2_primary_win": primary["phase2_primary_win"],
        },
        "headline": {
            "minibatch_tango_vanilla_mse": primary["tango_vanilla_prototype_mse_mean"],
            "minibatch_tango_mb_mse": primary["tango_mb_prototype_mse_mean"],
            "minibatch_storm_mse": primary["storm_prototype_mse_mean"],
            "minibatch_passive_mse": primary["passive_multi_round_prototype_mse_mean"],
            "minibatch_passive_mb_mse": primary["passive_mb_prototype_mse_mean"],
            "minibatch_tango_mb_vs_passive_gain_x": primary["tango_mb_vs_passive_gain_x"],
            "minibatch_phase2_win": primary["phase2_primary_win"],
            "balanced_tango_mb_mse": balanced["tango_mb_prototype_mse_mean"],
            "balanced_tango_vanilla_mse": balanced["tango_vanilla_prototype_mse_mean"],
        },
        "plots": [str(plot_path.relative_to(RUN_ROOT))],
        "runtime_seconds": time.time() - started,
    }
    # Remove mistaken threat_model placeholder
    metrics["threat_model"] = (
        "active terminal probing; hyperparameters public; no intermediate gradients"
    )

    out_path = ARTIFACTS / "round11_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    logger.info("wrote %s", out_path)
    logger.info("phase2_primary=%s", metrics["phase2_primary"])
    logger.info("headline=%s", metrics["headline"])
    return metrics


if __name__ == "__main__":
    run()
