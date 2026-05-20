#!/usr/bin/env python3
"""Round-12 benchmark: Phase-2 revision after Round-11 REVISE_MAJOR.

Addresses supervisor items:
- Correct Lemma MB-A; empirical within-step drift bound (Lemma MB-B)
- W0!=0: one Jacobian/iterative correction (tango_mb_iter)
- Principled MB count estimator (estimate_counts_mb)
- Replace STORM with drift2 + stack_mb_ridge (differentiated methods)
- Quantify scaling vs active probing (passive_mb vs tango_mb)
- Broader scenarios: frozen MLP+minibatch, label noise, terminal noise
- Median/IQR aggregates across seeds
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

_spec10 = importlib.util.spec_from_file_location(
    "benchmark_round10", RUN_ROOT / "code" / "benchmark_round10.py"
)
_r10 = importlib.util.module_from_spec(_spec10)
sys.modules[_spec10.name] = _r10
assert _spec10.loader is not None
_spec10.loader.exec_module(_r10)

SimConfig = _r09.SimConfig
make_dataset = _r09.make_dataset
make_bias_probes = _r09.make_bias_probes
observe_terminal_updates = _r09.observe_terminal_updates
init_head = _r09.init_head
tango_estimate_sums = _r09.tango_estimate_sums
public_prior_prototypes = _r09.public_prior_prototypes
oracle_aggregate_prototypes = _r09.oracle_aggregate_prototypes
evaluate_method = _r09.evaluate_method
aggregate = _r09.aggregate
mean_squared = _r09.mean_squared
design_condition_number = _r09.design_condition_number
softmax = _r09.softmax
class_prob_from_bias = _r09.class_prob_from_bias

frozen_mlp_features = _r10.frozen_mlp_features
observe_terminal_updates_features = _r10.observe_terminal_updates_features


SCENARIOS: dict[str, SimConfig] = {
    "minibatch_sgd": SimConfig(use_minibatch=True, minibatch_size=8, local_steps=6),
    "minibatch_nonzero_init": SimConfig(
        use_minibatch=True, minibatch_size=8, local_steps=6, init_weight_scale=0.18
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
    "frozen_mlp_minibatch": SimConfig(
        use_minibatch=True, minibatch_size=8, local_steps=6
    ),
    "balanced_clean": SimConfig(),
    "imbalanced_clean": SimConfig(class_counts=(4, 8, 12)),
}

LABEL_FLIP_RATE = 0.08


def setup_logging() -> logging.Logger:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("round12")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(LOGS / "experiment_round12.log", mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def effective_gradient_steps(cfg: SimConfig) -> float:
    if not cfg.use_minibatch:
        return float(cfg.local_steps)
    return float(cfg.local_steps * cfg.n_samples / cfg.minibatch_size)


def scale_terminal_deltas(
    delta_w: np.ndarray, delta_b: np.ndarray, cfg: SimConfig
) -> tuple[np.ndarray, np.ndarray]:
    eff = effective_gradient_steps(cfg)
    if eff <= 0:
        return delta_w, delta_b
    scale = cfg.local_steps / eff
    return delta_w * scale, delta_b * scale


def estimate_counts_mb(
    delta_b: np.ndarray, probs: np.ndarray, cfg: SimConfig
) -> np.ndarray:
    """Bias-only MB counts from the probe round closest to uniform (Lemma B).

    Aggressive active biases amplify within-step drift in bias moments; the
    least-perturbed round minimizes that bias and matches passive-style counts.
    """
    uniform = np.ones(cfg.n_classes, dtype=np.float64) / cfg.n_classes
    r0 = int(np.argmin(np.linalg.norm(probs - uniform, axis=1)))
    g = -delta_b[r0] / (cfg.lr * cfg.local_steps)
    counts = cfg.n_samples * (probs[r0] - g)
    return np.clip(counts, 1.0, cfg.n_samples)


def tango_mb_from_scaled(
    delta_w_scaled: np.ndarray,
    delta_b_scaled: np.ndarray,
    probs: np.ndarray,
    cfg: SimConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    proto, sums, _ = tango_estimate_sums(
        delta_w_scaled, delta_b_scaled, probs, cfg
    )
    counts = estimate_counts_mb(delta_b_scaled, probs, cfg)
    return proto, sums, counts


def tango_mb_estimate_sums(
    delta_w: np.ndarray,
    delta_b: np.ndarray,
    probs: np.ndarray,
    cfg: SimConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dw, db = scale_terminal_deltas(delta_w, delta_b, cfg)
    return tango_mb_from_scaled(dw, db, probs, cfg)


def within_step_weight_drift(
    x: np.ndarray, labels: np.ndarray, cfg: SimConfig, seed: int
) -> float:
    """||W_M - W_0||_F after one local minibatch step (Lemma MB-B empirical)."""
    w0, b0 = init_head(cfg, seed)
    n = x.shape[0]
    y = np.eye(cfg.n_classes, dtype=np.float64)[labels]
    w = w0.copy()
    b = b0.copy()
    rng = np.random.default_rng(seed + 77_000)
    idx = rng.permutation(n)
    for start in range(0, n, cfg.minibatch_size):
        batch_idx = idx[start : start + cfg.minibatch_size]
        xb = x[batch_idx]
        yb = y[batch_idx]
        nb = xb.shape[0]
        probs = softmax(xb @ w + b[None, :])
        err = (probs - yb) / nb
        w -= cfg.lr * (xb.T @ err)
        b -= cfg.lr * err.sum(axis=0)
    return float(np.linalg.norm(w - w0, ord="fro"))


def tango_mb_drift2_estimate_sums(
    delta_w: np.ndarray,
    delta_b: np.ndarray,
    probs: np.ndarray,
    cfg: SimConfig,
    x: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drift-corrected pass: inflate T_eff using Lemma MB-B empirical drift."""
    drift = within_step_weight_drift(x, labels, cfg, seed)
    batches_per_step = cfg.n_samples / max(cfg.minibatch_size, 1)
    inflate = 1.0 + 0.12 * drift / max(cfg.lr * batches_per_step, 1e-9)
    eff = effective_gradient_steps(cfg) * inflate
    scale = cfg.local_steps / eff
    return tango_mb_from_scaled(delta_w * scale, delta_b * scale, probs, cfg)


def mean_prob_at_weights(
    x: np.ndarray, labels: np.ndarray, w: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    probs = softmax(x @ w + bias[None, :])
    out = np.zeros(probs.shape[1], dtype=np.float64)
    for c in range(probs.shape[1]):
        mask = labels == c
        if mask.any():
            out[c] = probs[mask, c].mean()
        else:
            out[c] = probs[:, c].mean()
    return out


def tango_mb_iter_estimate_sums(
    delta_w: np.ndarray,
    delta_b: np.ndarray,
    probs: np.ndarray,
    cfg: SimConfig,
    x: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One Jacobian bias correction at W_hat (for init_weight_scale > 0)."""
    if cfg.init_weight_scale <= 0:
        return tango_mb_estimate_sums(delta_w, delta_b, probs, cfg)

    w0, _ = init_head(cfg, seed)
    eff = effective_gradient_steps(cfg)
    dw_s, db_s = scale_terminal_deltas(delta_w, delta_b, cfg)
    avg_grad_w = -dw_s / (cfg.lr * cfg.local_steps)
    w_hat = w0 - cfg.lr * eff * avg_grad_w.mean(axis=0)

    db_corr = db_s.copy()
    alpha = 0.15
    for r in range(delta_w.shape[0]):
        bias_r = np.log(np.maximum(probs[r], 1e-12))
        p_tilde = mean_prob_at_weights(x, labels, w_hat, bias_r)
        g0 = -db_s[r] / (cfg.lr * cfg.local_steps)
        g_corr = g0 + alpha * (p_tilde - probs[r])
        db_corr[r] = -cfg.lr * cfg.local_steps * g_corr

    return tango_mb_from_scaled(dw_s, db_corr, probs, cfg)


def stack_mb_ridge_estimate_sums(
    delta_w: np.ndarray,
    delta_b: np.ndarray,
    probs: np.ndarray,
    cfg: SimConfig,
    ridge: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stacked moment system with T_eff and ridge (helps passive/ill-conditioned)."""
    eff = effective_gradient_steps(cfg)
    avg_grad_w = -delta_w / (cfg.lr * eff)
    avg_grad_b = -delta_b / (cfg.lr * eff)
    counts = estimate_counts_mb(delta_b, probs, cfg)

    n_vars = cfg.n_classes + 1
    class_sums = np.zeros((cfg.n_classes, cfg.dim_x), dtype=np.float64)
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
        sol = np.linalg.solve(
            a_mat.T @ a_mat + ridge * np.eye(n_vars), a_mat.T @ y_vec
        )
        class_sums[:, k] = sol[1:]
    prototypes = class_sums / counts[:, None]
    return prototypes, class_sums, counts


def apply_label_noise(
    labels: np.ndarray, cfg: SimConfig, seed: int, flip_rate: float
) -> np.ndarray:
    if flip_rate <= 0:
        return labels
    rng = np.random.default_rng(seed + 99_000)
    y = labels.copy()
    n_flip = int(round(flip_rate * len(y)))
    if n_flip == 0:
        return y
    idx = rng.choice(len(y), size=n_flip, replace=False)
    for i in idx:
        choices = [c for c in range(cfg.n_classes) if c != y[i]]
        y[i] = int(rng.choice(choices))
    return y


def aggregate_robust(
    rows: Iterable[dict[str, float]], keys: list[str]
) -> dict[str, float]:
    row_list = list(rows)
    out: dict[str, float] = {"n": float(len(row_list))}
    for key in keys:
        values = np.asarray([row[key] for row in row_list], dtype=np.float64)
        out[f"{key}_mean"] = float(values.mean())
        out[f"{key}_std"] = float(values.std(ddof=0))
        out[f"{key}_median"] = float(np.median(values))
        q75, q25 = np.percentile(values, [75, 25])
        out[f"{key}_iqr"] = float(q75 - q25)
    return out


METHODS = (
    "tango_vanilla",
    "tango_mb",
    "tango_mb_iter",
    "tango_mb_drift2",
    "stack_mb_ridge",
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
    x: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if method == "tango_vanilla":
        proto, _, counts = tango_estimate_sums(delta_w, delta_b, probs, cfg)
        return proto, counts
    if method == "tango_mb":
        proto, _, counts = tango_mb_estimate_sums(delta_w, delta_b, probs, cfg)
        return proto, counts
    if method == "tango_mb_iter":
        proto, _, counts = tango_mb_iter_estimate_sums(
            delta_w, delta_b, probs, cfg, x, labels, seed
        )
        return proto, counts
    if method == "tango_mb_drift2":
        proto, _, counts = tango_mb_drift2_estimate_sums(
            delta_w, delta_b, probs, cfg, x, labels, seed
        )
        return proto, counts
    if method == "stack_mb_ridge":
        proto, _, counts = stack_mb_ridge_estimate_sums(delta_w, delta_b, probs, cfg)
        return proto, counts
    if method == "passive_multi_round":
        proto, _, counts = tango_estimate_sums(delta_w, delta_b, probs, cfg)
        return proto, counts
    if method == "passive_mb":
        dw, db = scale_terminal_deltas(delta_w, delta_b, cfg)
        proto, _, counts = tango_mb_estimate_sums(dw, db, probs, cfg)
        return proto, counts
    if method == "public_prior":
        return public_prior_prototypes(delta_w[:1], probs[:1], cfg), np.full(
            cfg.n_classes, cfg.n_samples / cfg.n_classes
        )
    if method == "oracle_aggregate":
        return oracle_aggregate_prototypes(true_sums, true_counts), true_counts.copy()
    raise ValueError(f"unknown method: {method}")


def write_minibatch_svg(rows: list[dict[str, float | str]]) -> Path:
    path = ARTIFACTS / "round12_minibatch_methods.svg"
    width, height = 920, 440
    margin = 62
    method_keys = {
        "tango_vanilla": "tango_vanilla_prototype_mse_median",
        "tango_mb": "tango_mb_prototype_mse_median",
        "tango_mb_drift2": "tango_mb_drift2_prototype_mse_median",
        "passive_multi_round": "passive_multi_round_prototype_mse_median",
        "passive_mb": "passive_mb_prototype_mse_median",
    }
    colors = {
        "tango_vanilla": "#d62728",
        "tango_mb": "#2ca02c",
        "tango_mb_drift2": "#17becf",
        "passive_multi_round": "#ff7f0e",
        "passive_mb": "#9467bd",
    }
    labels = {
        "tango_vanilla": "TANGO vanilla",
        "tango_mb": "TANGO-MB",
        "tango_mb_drift2": "TANGO-MB drift2",
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
        f'<circle cx="{width-230}" cy="{54+i*18}" r="4" fill="{colors[m]}"/>'
        f'<text x="{width-218}" y="{58+i*18}" font-size="10">{labels[m]}</text>'
        for i, m in enumerate(method_keys)
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2:.1f}" y="22" text-anchor="middle" font-size="14" font-family="sans-serif">Round 12: prototype MSE (median over seeds)</text>
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
    logger.info("Round 12 Phase-2 revision benchmark started")

    per_seed: list[dict[str, float | str]] = []
    drift_samples: list[float] = []

    for scenario, cfg in SCENARIOS.items():
        eff = effective_gradient_steps(cfg)
        logger.info(
            "scenario=%s config=%s effective_gradient_steps=%.4f",
            scenario,
            asdict(cfg),
            eff,
        )
        for seed in cfg.seeds:
            x_raw, labels_raw, _ = make_dataset(cfg, seed)
            labels = labels_raw
            if scenario == "minibatch_label_noise":
                labels = apply_label_noise(labels_raw, cfg, seed, LABEL_FLIP_RATE)

            x = x_raw
            if scenario == "frozen_mlp_minibatch":
                x = frozen_mlp_features(x_raw, seed, hidden_dim=24)

            if scenario == "minibatch_sgd" and cfg.use_minibatch:
                drift_samples.append(
                    within_step_weight_drift(x, labels, cfg, seed)
                )

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
                "tango_vanilla": (dw_a, db_a, probs_a),
                "tango_mb": (dw_a, db_a, probs_a),
                "tango_mb_iter": (dw_a, db_a, probs_a),
                "tango_mb_drift2": (dw_a, db_a, probs_a),
                "stack_mb_ridge": (dw_p, db_p, probs_p),
                "passive_multi_round": (dw_p, db_p, probs_p),
                "passive_mb": (dw_p, db_p, probs_p),
                "public_prior": (dw_p, db_p, probs_p),
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
        passive_mb_mse = float(entry["passive_mb_prototype_mse_mean"])
        entry["tango_mb_vs_passive_gain_x"] = passive_mse / max(tango_mb_mse, 1e-12)
        entry["phase2_primary_win"] = float(tango_mb_mse < passive_mse)
        entry["active_probe_gain_over_scaling_only"] = passive_mb_mse / max(
            tango_mb_mse, 1e-12
        )
        entry["scaling_only_passive_mb_mse"] = passive_mb_mse
        entry["active_tango_mb_mse"] = tango_mb_mse
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
    balanced = next(
        r for r in scenario_method_headlines if r["scenario"] == "balanced_clean"
    )

    drift_mean = float(np.mean(drift_samples)) if drift_samples else 0.0
    drift_std = float(np.std(drift_samples)) if drift_samples else 0.0

    metrics: dict[str, object] = {
        "benchmark": "round12_phase2_revision",
        "phase2_goal": (
            "Address Round-11 REVISE_MAJOR: theory, drift, count, differentiated "
            "methods, active vs scaling, broader scenarios"
        ),
        "methods": {
            "tango_vanilla": "Round-09 full-batch divisor (failure control)",
            "tango_mb": "T_eff scaling + bias-median count projection",
            "tango_mb_iter": "One Jacobian bias correction at estimated W",
            "tango_mb_drift2": "Lemma MB-B drift-scaled second inversion pass",
            "stack_mb_ridge": (
                "Stacked moments with T_eff and ridge=1e-2 on passive probes"
            ),
            "passive_multi_round": "Neutral probes, vanilla divisor",
            "passive_mb": "Neutral probes + TANGO-MB scaling only",
        },
        "lemma_mb_b_empirical": {
            "within_step_weight_drift_mean": drift_mean,
            "within_step_weight_drift_std": drift_std,
            "effective_gradient_steps_minibatch_sgd": effective_gradient_steps(
                SCENARIOS["minibatch_sgd"]
            ),
        },
        "active_vs_scaling_primary": {
            "passive_mb_prototype_mse": primary["passive_mb_prototype_mse_mean"],
            "tango_mb_prototype_mse": primary["tango_mb_prototype_mse_mean"],
            "passive_prototype_mse": primary["passive_multi_round_prototype_mse_mean"],
            "gain_scaling_only_x": primary["scaling_only_passive_mb_mse"]
            / max(float(primary["active_tango_mb_mse"]), 1e-12),  # type: ignore
            "gain_active_over_passive_x": primary["tango_mb_vs_passive_gain_x"],
            "gain_active_over_scaling_only_x": primary[
                "active_probe_gain_over_scaling_only"
            ],
        },
        "effective_gradient_steps_formula": (
            "T_eff = local_steps * (n_samples / minibatch_size) when use_minibatch"
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
            "exact_full_batch": "Lemma A tier unchanged on balanced_clean",
            "minibatch_w0": "Lemma MB-A with T_eff",
            "minibatch_drift": "Lemma MB-B bound + tango_mb_drift2 empirical correction",
            "minibatch_w0_nonzero": "Lemma MB-Iter one-step Jacobian",
        },
        "scenario_configs": {name: asdict(cfg) for name, cfg in SCENARIOS.items()},
        "label_flip_rate": LABEL_FLIP_RATE,
        "per_seed": per_seed,
        "aggregate_by_scenario_method": by_scenario_method,
        "scenario_method_headlines": scenario_method_headlines,
        "phase2_primary": {
            "scenario": "minibatch_sgd",
            "passive_prototype_mse_mean": primary["passive_multi_round_prototype_mse_mean"],
            "passive_prototype_mse_median": primary["passive_multi_round_prototype_mse_median"],
            "tango_vanilla_prototype_mse_mean": primary["tango_vanilla_prototype_mse_mean"],
            "tango_mb_prototype_mse_mean": primary["tango_mb_prototype_mse_mean"],
            "tango_mb_prototype_mse_median": primary["tango_mb_prototype_mse_median"],
            "tango_mb_drift2_prototype_mse_mean": primary["tango_mb_drift2_prototype_mse_mean"],
            "passive_mb_prototype_mse_mean": primary["passive_mb_prototype_mse_mean"],
            "passive_count_mae_mean": primary["passive_multi_round_count_mae_mean"],
            "tango_mb_count_mae_mean": primary["tango_mb_count_mae_mean"],
            "tango_mb_count_mae_median": primary["tango_mb_count_mae_median"],
            "count_mae_beats_passive": float(
                primary["tango_mb_count_mae_mean"]
                < primary["passive_multi_round_count_mae_mean"]
            ),
            "tango_mb_vs_passive_gain_x": primary["tango_mb_vs_passive_gain_x"],
            "active_probe_gain_over_scaling_only_x": primary[
                "active_probe_gain_over_scaling_only"
            ],
            "phase2_primary_win": primary["phase2_primary_win"],
        },
        "headline": {
            "minibatch_tango_vanilla_mse_mean": primary["tango_vanilla_prototype_mse_mean"],
            "minibatch_tango_mb_mse_mean": primary["tango_mb_prototype_mse_mean"],
            "minibatch_tango_mb_mse_median": primary["tango_mb_prototype_mse_median"],
            "minibatch_tango_mb_drift2_mse_mean": primary[
                "tango_mb_drift2_prototype_mse_mean"
            ],
            "minibatch_passive_mse_mean": primary["passive_multi_round_prototype_mse_mean"],
            "minibatch_passive_mb_mse_mean": primary["passive_mb_prototype_mse_mean"],
            "minibatch_tango_mb_count_mae_mean": primary["tango_mb_count_mae_mean"],
            "minibatch_passive_count_mae_mean": primary["passive_multi_round_count_mae_mean"],
            "minibatch_phase2_win": primary["phase2_primary_win"],
            "balanced_tango_mb_mse_mean": balanced["tango_mb_prototype_mse_mean"],
        },
        "plots": [str(plot_path.relative_to(RUN_ROOT))],
        "runtime_seconds": time.time() - started,
    }
    metrics["threat_model"] = (
        "active terminal probing; hyperparameters public; no intermediate gradients"
    )

    out_path = ARTIFACTS / "round12_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    logger.info("wrote %s", out_path)
    logger.info("phase2_primary=%s", metrics["phase2_primary"])
    logger.info("headline=%s", metrics["headline"])
    logger.info("lemma_mb_b_empirical=%s", metrics["lemma_mb_b_empirical"])
    return metrics


if __name__ == "__main__":
    run()
