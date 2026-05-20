#!/usr/bin/env python3
"""Round-09 benchmark: paper-readiness evaluation for TALON/TANGO.

Extends Round 08 with:
- minibatch SGD and nonzero initial head-weight stress tests
- larger class count and representation dimension
- passive multi-round terminal baseline
- public/statistical prototype prior baseline
- oracle aggregate upper bound (separate from TANGO)
- separate reporting of count recovery vs prototype recovery
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


RUN_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = RUN_ROOT / "artifacts"
LOGS = RUN_ROOT / "logs"


@dataclass(frozen=True)
class SimConfig:
    n_classes: int = 3
    class_counts: tuple[int, ...] = (8, 8, 8)
    dim_x: int = 10
    prototype_scale: float = 2.0
    within_class_std: float = 0.55
    lr: float = 0.04
    local_steps: int = 3
    probe_rounds: tuple[int, ...] = (1, 2, 4, 8)
    bias_scale: float = 1.2
    terminal_noise_std: float = 0.0
    use_minibatch: bool = False
    minibatch_size: int = 8
    init_weight_scale: float = 0.0
    seeds: tuple[int, ...] = (3, 7, 11, 19, 23, 29, 31, 37)

    @property
    def n_samples(self) -> int:
        return int(sum(self.class_counts))


SCENARIOS: dict[str, SimConfig] = {
    "balanced_clean": SimConfig(),
    "imbalanced_clean": SimConfig(class_counts=(4, 8, 12)),
    "minibatch_sgd": SimConfig(use_minibatch=True, minibatch_size=8, local_steps=6),
    "nonzero_init_head": SimConfig(init_weight_scale=0.18),
    "large_10class_30dim": SimConfig(
        n_classes=10,
        class_counts=(6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
        dim_x=30,
        bias_scale=1.5,
    ),
    "more_local_steps": SimConfig(local_steps=8),
    "high_within_class_variance": SimConfig(within_class_std=1.0),
}


def setup_logging() -> logging.Logger:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("round09")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(LOGS / "experiment_round09.log", mode="w")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def class_prob_from_bias(bias: np.ndarray) -> np.ndarray:
    shifted = bias - bias.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def make_dataset(cfg: SimConfig, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(cfg.n_classes, cfg.dim_x))
    q, _ = np.linalg.qr(raw.T)
    prototypes = cfg.prototype_scale * q[:, : cfg.n_classes].T
    labels = np.concatenate(
        [np.full(n, c, dtype=np.int64) for c, n in enumerate(cfg.class_counts)]
    )
    x = np.vstack(
        [
            prototypes[c]
            + cfg.within_class_std * rng.normal(size=(cfg.class_counts[c], cfg.dim_x))
            for c in range(cfg.n_classes)
        ]
    )
    return x.astype(np.float64), labels, prototypes.astype(np.float64)


def init_head(cfg: SimConfig, seed: int) -> tuple[np.ndarray, np.ndarray]:
    w0 = np.zeros((cfg.dim_x, cfg.n_classes), dtype=np.float64)
    if cfg.init_weight_scale > 0:
        rng = np.random.default_rng(seed + 12_000)
        w0 = cfg.init_weight_scale * rng.normal(size=w0.shape)
    return w0, np.zeros(cfg.n_classes, dtype=np.float64)


def local_terminal_update(
    x: np.ndarray,
    labels: np.ndarray,
    w0: np.ndarray,
    b0: np.ndarray,
    cfg: SimConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    y = np.eye(cfg.n_classes, dtype=np.float64)[labels]
    w = w0.copy()
    b = b0.copy()
    rng = np.random.default_rng(seed + 77_000)
    for step in range(cfg.local_steps):
        if cfg.use_minibatch:
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
        else:
            probs = softmax(x @ w + b[None, :])
            err = (probs - y) / n
            w -= cfg.lr * (x.T @ err)
            b -= cfg.lr * err.sum(axis=0)
    return w - w0, b - b0


def make_bias_probes(
    cfg: SimConfig, seed: int, n_rounds: int, passive: bool = False
) -> np.ndarray:
    if passive:
        return np.zeros((n_rounds, cfg.n_classes), dtype=np.float64)
    rng = np.random.default_rng(seed + 50_000)
    probes = [np.zeros(cfg.n_classes, dtype=np.float64)]
    while len(probes) < n_rounds:
        probes.append(cfg.bias_scale * rng.normal(size=cfg.n_classes))
    return np.vstack(probes)


def observe_terminal_updates(
    x: np.ndarray,
    labels: np.ndarray,
    biases: np.ndarray,
    cfg: SimConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    deltas_w = []
    deltas_b = []
    probs = []
    w0, _ = init_head(cfg, seed)
    for r, bias in enumerate(biases):
        delta_w, delta_b = local_terminal_update(
            x, labels, w0, bias, cfg, seed + 100 * r
        )
        deltas_w.append(delta_w)
        deltas_b.append(delta_b)
        probs.append(class_prob_from_bias(bias))
    delta_w = np.stack(deltas_w)
    delta_b = np.stack(deltas_b)
    if cfg.terminal_noise_std > 0:
        rng = np.random.default_rng(seed + 91_000 + len(biases))
        delta_w = delta_w + cfg.terminal_noise_std * rng.normal(size=delta_w.shape)
        delta_b = delta_b + cfg.terminal_noise_std * rng.normal(size=delta_b.shape)
    return delta_w, delta_b, np.stack(probs)


def design_condition_number(probs: np.ndarray, cfg: SimConfig) -> float:
    rows = []
    for p in probs:
        for c in range(cfg.n_classes):
            row = np.zeros(cfg.n_classes + 1, dtype=np.float64)
            row[0] = p[c]
            row[1 + c] = -1.0
            rows.append(row)
    row = np.zeros(cfg.n_classes + 1, dtype=np.float64)
    row[0] = 1.0
    row[1:] = -1.0
    rows.append(10.0 * row)
    return float(np.linalg.cond(np.vstack(rows)))


def estimate_counts(delta_b: np.ndarray, probs: np.ndarray, cfg: SimConfig) -> np.ndarray:
    avg_grad_b = -delta_b / (cfg.lr * cfg.local_steps)
    counts_per_round = cfg.n_samples * (probs - avg_grad_b)
    counts = counts_per_round.mean(axis=0)
    return np.clip(counts, 1.0, cfg.n_samples)


def tango_estimate_sums(
    delta_w: np.ndarray,
    delta_b: np.ndarray,
    probs: np.ndarray,
    cfg: SimConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    avg_grad_w = -delta_w / (cfg.lr * cfg.local_steps)
    counts = estimate_counts(delta_b, probs, cfg)
    class_sums = np.zeros((cfg.n_classes, cfg.dim_x), dtype=np.float64)
    total = np.zeros(cfg.dim_x, dtype=np.float64)
    for k in range(cfg.dim_x):
        a_rows = []
        y_rows = []
        for r in range(delta_w.shape[0]):
            for c in range(cfg.n_classes):
                row = np.zeros(cfg.n_classes + 1, dtype=np.float64)
                row[0] = probs[r, c]
                row[1 + c] = -1.0
                a_rows.append(row)
                y_rows.append(cfg.n_samples * avg_grad_w[r, k, c])
        row = np.zeros(cfg.n_classes + 1, dtype=np.float64)
        row[0] = 1.0
        row[1:] = -1.0
        a_rows.append(10.0 * row)
        y_rows.append(0.0)
        sol, *_ = np.linalg.lstsq(np.vstack(a_rows), np.asarray(y_rows), rcond=None)
        total[k] = sol[0]
        class_sums[:, k] = sol[1:]
    prototypes = class_sums / counts[:, None]
    return prototypes, class_sums, counts


def public_prior_prototypes(
    delta_w: np.ndarray,
    probs: np.ndarray,
    cfg: SimConfig,
) -> np.ndarray:
    """Statistical prior: all classes share one prototype = crude dataset mean."""
    avg_grad_w = -delta_w[0] / (cfg.lr * cfg.local_steps)
    p0 = probs[0]
    # Rank-deficient single-round estimate of total sum S.
    crude_s = np.zeros(cfg.dim_x, dtype=np.float64)
    for k in range(cfg.dim_x):
        denom = float(np.sum(p0 * (1.0 - p0)))
        if abs(denom) < 1e-12:
            crude_s[k] = 0.0
        else:
            crude_s[k] = cfg.n_samples * avg_grad_w[k].sum() / denom
    mean = crude_s / cfg.n_samples
    return np.repeat(mean[None, :], cfg.n_classes, axis=0)


def oracle_aggregate_prototypes(
    true_sums: np.ndarray, true_counts: np.ndarray
) -> np.ndarray:
    """Upper bound: perfect aggregate moments (not achievable from terminal probes)."""
    return true_sums / true_counts[:, None]


def nearest_neighbor_mse(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    remaining = list(range(x_true.shape[0]))
    sq = 0.0
    for pred in x_hat:
        dists = [float(np.mean((pred - x_true[j]) ** 2)) for j in remaining]
        idx = int(np.argmin(dists))
        sq += dists[idx]
        remaining.pop(idx)
    return sq / x_hat.shape[0]


def mean_squared(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def count_metrics(est_counts: np.ndarray, true_counts: np.ndarray) -> dict[str, float]:
    mae = float(np.mean(np.abs(est_counts - true_counts)))
    rel = float(np.mean(np.abs(est_counts - true_counts) / np.maximum(true_counts, 1.0)))
    return {"count_mae": mae, "count_relative_error": rel}


def aggregate(rows: Iterable[dict[str, float]], keys: list[str]) -> dict[str, float]:
    row_list = list(rows)
    out: dict[str, float] = {"n": float(len(row_list))}
    for key in keys:
        values = np.asarray([row[key] for row in row_list], dtype=np.float64)
        out[f"{key}_mean"] = float(values.mean())
        out[f"{key}_std"] = float(values.std(ddof=0))
    return out


def write_baselines_svg(rows: list[dict[str, float | str]]) -> Path:
    path = ARTIFACTS / "round09_baselines.svg"
    width, height = 920, 420
    margin = 62
    method_keys = {
        "tango_active": "tango_prototype_mse_mean",
        "passive_multi_round": "passive_multi_round_prototype_mse_mean",
        "public_prior": "public_prior_prototype_mse_mean",
        "oracle_aggregate": "oracle_aggregate_prototype_mse_mean",
    }
    colors = {"tango_active": "#1f77b4", "passive_multi_round": "#ff7f0e",
              "public_prior": "#9467bd", "oracle_aggregate": "#2ca02c"}
    labels = {
        "tango_active": "TANGO active",
        "passive_multi_round": "passive multi-round",
        "public_prior": "public prior",
        "oracle_aggregate": "oracle aggregate",
    }
    scenarios = [row["scenario"] for row in rows]
    max_y = max(
        float(row[key])
        for row in rows
        for key in method_keys.values()
    ) * 1.15

    def sx(i: int) -> float:
        return margin + i * (width - 2 * margin) / max(1, len(rows) - 1)

    def sy(y: float) -> float:
        return height - margin - y / max(max_y, 1e-9) * (height - 2 * margin)

    polys = []
    for method, key in method_keys.items():
        vals = [float(row[key]) for row in rows]
        pts = " ".join(f"{sx(i):.1f},{sy(y):.1f}" for i, y in enumerate(vals))
        circles = "\n".join(
            f'<circle cx="{sx(i):.1f}" cy="{sy(y):.1f}" r="3.5" fill="{colors[method]}"/>'
            for i, y in enumerate(vals)
        )
        polys.append(
            f'<polyline fill="none" stroke="{colors[method]}" stroke-width="2.2" points="{pts}"/>\n{circles}'
        )
    x_labels = "\n".join(
        f'<text x="{sx(i):.1f}" y="{height-30}" transform="rotate(24 {sx(i):.1f},{height-30})" '
        f'text-anchor="start" font-size="10">{name}</text>'
        for i, name in enumerate(scenarios)
    )
    legend = "\n".join(
        f'<circle cx="{width-210}" cy="{58+i*20}" r="4" fill="{colors[m]}"/>'
        f'<text x="{width-198}" y="{62+i*20}" font-size="11">{labels[m]}</text>'
        for i, m in enumerate(method_keys)
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2:.1f}" y="22" text-anchor="middle" font-size="15" font-family="sans-serif">Round 09: prototype MSE by method (8 active/passive rounds)</text>
<line x1="{margin}" x2="{width-margin}" y1="{height-margin}" y2="{height-margin}" stroke="#333"/>
<line x1="{margin}" x2="{margin}" y1="{margin}" y2="{height-margin}" stroke="#333"/>
{x_labels}
{''.join(polys)}
{legend}
<text transform="translate(14,{height/2:.1f}) rotate(-90)" text-anchor="middle" font-size="12">prototype MSE</text>
</svg>"""
    path.write_text(svg)
    return path


def write_count_proto_svg(rows: list[dict[str, float | str]]) -> Path:
    path = ARTIFACTS / "round09_count_vs_proto.svg"
    width, height = 700, 420
    margin = 58
    max_count = max(float(row["tango_count_mae_mean"]) for row in rows) * 1.15
    max_proto = max(float(row["tango_prototype_mse_mean"]) for row in rows) * 1.15

    def sx(i: int) -> float:
        return margin + i * (width - 2 * margin) / max(1, len(rows) - 1)

    def sy_count(y: float) -> float:
        return height - margin - y / max(max_count, 1e-9) * (height - 2 * margin)

    def sy_proto(y: float) -> float:
        return height - margin - y / max(max_proto, 1e-9) * (height - 2 * margin)

    count_pts = " ".join(
        f"{sx(i):.1f},{sy_count(float(row['tango_count_mae_mean'])):.1f}"
        for i, row in enumerate(rows)
    )
    proto_pts = " ".join(
        f"{sx(i):.1f},{sy_proto(float(row['tango_prototype_mse_mean'])):.1f}"
        for i, row in enumerate(rows)
    )
    x_labels = "\n".join(
        f'<text x="{sx(i):.1f}" y="{height-28}" transform="rotate(22 {sx(i):.1f},{height-28})" '
        f'text-anchor="start" font-size="10">{row["scenario"]}</text>'
        for i, row in enumerate(rows)
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2:.1f}" y="22" text-anchor="middle" font-size="15" font-family="sans-serif">Round 09: TANGO count MAE vs prototype MSE (8 rounds)</text>
<line x1="{margin}" x2="{width-margin}" y1="{height-margin}" y2="{height-margin}" stroke="#333"/>
<line x1="{margin}" x2="{margin}" y1="{margin}" y2="{height-margin}" stroke="#333"/>
{x_labels}
<polyline fill="none" stroke="#d62728" stroke-width="2.5" points="{count_pts}"/>
<polyline fill="none" stroke="#1f77b4" stroke-width="2.5" points="{proto_pts}"/>
<rect x="{width-210}" y="50" width="185" height="52" fill="white" stroke="#ddd"/>
<circle cx="{width-195}" cy="66" r="4" fill="#d62728"/><text x="{width-182}" y="70" font-size="11">count MAE</text>
<circle cx="{width-195}" cy="88" r="4" fill="#1f77b4"/><text x="{width-182}" y="92" font-size="11">prototype MSE</text>
</svg>"""
    path.write_text(svg)
    return path


def evaluate_method(
    method: str,
    est_proto: np.ndarray,
    est_counts: np.ndarray,
    true_proto: np.ndarray,
    true_counts: np.ndarray,
    true_sums: np.ndarray,
    x: np.ndarray,
    labels: np.ndarray,
    cfg: SimConfig,
) -> dict[str, float]:
    cm = count_metrics(est_counts, true_counts)
    proto_mse = mean_squared(est_proto, true_proto)
    dataset_mean_mse = mean_squared(
        est_proto * est_counts[:, None] / cfg.n_samples,
        true_sums.sum(axis=0) / cfg.n_samples,
    )
    proto_dataset = np.vstack(
        [
            np.repeat(est_proto[c][None, :], cfg.class_counts[c], axis=0)
            for c in range(cfg.n_classes)
        ]
    )
    return {
        "method": method,
        **cm,
        "prototype_mse": proto_mse,
        "dataset_mean_mse": dataset_mean_mse,
        "individual_mse_from_prototypes": nearest_neighbor_mse(proto_dataset, x),
    }


def run() -> dict[str, object]:
    logger = setup_logging()
    started = time.time()
    logger.info("Round 09 TALON paper-readiness benchmark started")
    per_seed: list[dict[str, float | str]] = []

    for scenario, cfg in SCENARIOS.items():
        logger.info("scenario=%s config=%s", scenario, asdict(cfg))
        for seed in cfg.seeds:
            x, labels, true_centers = make_dataset(cfg, seed)
            true_counts = np.asarray(cfg.class_counts, dtype=np.float64)
            true_sums = np.vstack([x[labels == c].sum(axis=0) for c in range(cfg.n_classes)])
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

            # TANGO active probes
            biases_active = make_bias_probes(cfg, seed, n_rounds, passive=False)
            dw_a, db_a, probs_a = observe_terminal_updates(x, labels, biases_active, cfg, seed)
            proto_a, sums_a, counts_a = tango_estimate_sums(dw_a, db_a, probs_a, cfg)

            # Passive multi-round (repeated neutral bias)
            biases_passive = make_bias_probes(cfg, seed, n_rounds, passive=True)
            dw_p, db_p, probs_p = observe_terminal_updates(x, labels, biases_passive, cfg, seed + 1)
            proto_p, _, counts_p = tango_estimate_sums(dw_p, db_p, probs_p, cfg)

            # Public statistical prior (single-round crude mean, no active probing)
            proto_pub = public_prior_prototypes(dw_p[:1], probs_p[:1], cfg)
            counts_pub = np.full(cfg.n_classes, cfg.n_samples / cfg.n_classes)

            # Oracle aggregate upper bound
            proto_oracle = oracle_aggregate_prototypes(true_sums, true_counts)
            counts_oracle = true_counts.copy()

            methods = {
                "tango_active": (proto_a, counts_a),
                "passive_multi_round": (proto_p, counts_p),
                "public_prior": (proto_pub, counts_pub),
                "oracle_aggregate": (proto_oracle, counts_oracle),
            }
            for method, (est_proto, est_counts) in methods.items():
                metrics = evaluate_method(
                    method, est_proto, est_counts, true_proto, true_counts,
                    true_sums, x, labels, cfg,
                )
                row = {
                    "scenario": scenario,
                    "seed": float(seed),
                    "probe_rounds": float(n_rounds),
                    "observed_intermediate_batch_gradients": 0.0,
                    "within_class_variance": within_class_var,
                    "design_condition_number": design_condition_number(probs_a, cfg),
                    **metrics,
                }
                per_seed.append(row)
                logger.info(
                    "scenario=%s seed=%s method=%s count_mae=%.4f proto_mse=%.6f indiv=%.6f",
                    scenario,
                    seed,
                    method,
                    row["count_mae"],
                    row["prototype_mse"],
                    row["individual_mse_from_prototypes"],
                )

    metric_keys = [
        "count_mae",
        "count_relative_error",
        "prototype_mse",
        "dataset_mean_mse",
        "individual_mse_from_prototypes",
        "within_class_variance",
        "design_condition_number",
    ]
    by_scenario_method: list[dict[str, float | str]] = []
    for scenario in SCENARIOS:
        for method in ["tango_active", "passive_multi_round", "public_prior", "oracle_aggregate"]:
            rows = [
                r for r in per_seed
                if r["scenario"] == scenario and r["method"] == method
            ]
            agg = aggregate(rows, metric_keys)
            agg["scenario"] = scenario
            agg["method"] = method
            by_scenario_method.append(agg)

    scenario_method_headlines: list[dict[str, float | str]] = []
    for scenario in SCENARIOS:
        rows = [r for r in by_scenario_method if r["scenario"] == scenario]
        tango = next(r for r in rows if r["method"] == "tango_active")
        passive = next(r for r in rows if r["method"] == "passive_multi_round")
        public = next(r for r in rows if r["method"] == "public_prior")
        oracle = next(r for r in rows if r["method"] == "oracle_aggregate")
        scenario_method_headlines.append(
            {
                "scenario": scenario,
                "tango_count_mae_mean": tango["count_mae_mean"],
                "tango_count_relative_error_mean": tango["count_relative_error_mean"],
                "tango_prototype_mse_mean": tango["prototype_mse_mean"],
                "tango_individual_mse_mean": tango["individual_mse_from_prototypes_mean"],
                "passive_multi_round_prototype_mse_mean": passive["prototype_mse_mean"],
                "public_prior_prototype_mse_mean": public["prototype_mse_mean"],
                "oracle_aggregate_prototype_mse_mean": oracle["prototype_mse_mean"],
                "tango_vs_passive_proto_gain_x": float(passive["prototype_mse_mean"])
                / max(float(tango["prototype_mse_mean"]), 1e-12),
                "within_class_variance_mean": tango["within_class_variance_mean"],
            }
        )

    baseline_plot = write_baselines_svg(scenario_method_headlines)
    count_plot = write_count_proto_svg(scenario_method_headlines)

    balanced = next(r for r in scenario_method_headlines if r["scenario"] == "balanced_clean")
    minibatch = next(r for r in scenario_method_headlines if r["scenario"] == "minibatch_sgd")
    nonzero = next(r for r in scenario_method_headlines if r["scenario"] == "nonzero_init_head")
    large = next(r for r in scenario_method_headlines if r["scenario"] == "large_10class_30dim")

    metrics: dict[str, object] = {
        "benchmark": "round09_talon_paper_readiness",
        "method": "TALON/TANGO with baselines and extended stress tests",
        "threat_model": "active terminal probing via server-chosen classifier biases (not passive honest-but-curious)",
        "observation_model": {
            "observed": [
                "terminal client model deltas per round",
                "server-chosen initial classifier biases",
                "public optimizer hyperparameters and architecture",
            ],
            "not_observed": [
                "intermediate minibatch gradients",
                "batch order",
                "batch incidence",
                "per-sample snapshots",
            ],
            "privacy_target": "class counts, class sums, prototypes, dataset mean; not individuals",
        },
        "theorem_scope": {
            "exact": "linear head, zero initial weights, full-batch local training, first-order terminal deltas, full-rank probe design",
            "approximate_empirical": [
                "minibatch_sgd",
                "nonzero_init_head",
                "more_local_steps",
                "large_10class_30dim",
            ],
        },
        "scenario_configs": {name: asdict(cfg) for name, cfg in SCENARIOS.items()},
        "per_seed": per_seed,
        "aggregate_by_scenario_method": by_scenario_method,
        "scenario_method_headlines": scenario_method_headlines,
        "count_recovery_reported_separately": True,
        "headline": {
            "balanced_tango_prototype_mse": balanced["tango_prototype_mse_mean"],
            "balanced_tango_count_mae": balanced["tango_count_mae_mean"],
            "balanced_passive_prototype_mse": balanced["passive_multi_round_prototype_mse_mean"],
            "balanced_public_prior_prototype_mse": balanced["public_prior_prototype_mse_mean"],
            "balanced_oracle_prototype_mse": balanced["oracle_aggregate_prototype_mse_mean"],
            "balanced_tango_vs_passive_gain_x": balanced["tango_vs_passive_proto_gain_x"],
            "minibatch_tango_prototype_mse": minibatch["tango_prototype_mse_mean"],
            "minibatch_tango_count_mae": minibatch["tango_count_mae_mean"],
            "nonzero_head_tango_prototype_mse": nonzero["tango_prototype_mse_mean"],
            "large_scale_tango_prototype_mse": large["tango_prototype_mse_mean"],
            "large_scale_tango_count_mae": large["tango_count_mae_mean"],
            "identifiability": "aggregate moments identifiable under exact assumptions; individuals remain non-identifiable",
        },
        "plots": [
            str(baseline_plot.relative_to(RUN_ROOT)),
            str(count_plot.relative_to(RUN_ROOT)),
        ],
        "runtime_seconds": time.time() - started,
    }
    out_path = ARTIFACTS / "round09_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    logger.info("wrote %s", out_path)
    logger.info("headline=%s", metrics["headline"])
    return metrics


if __name__ == "__main__":
    run()
