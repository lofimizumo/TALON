#!/usr/bin/env python3
"""Round-08 benchmark: stress tests for terminal aggregate leakage.

This script extends the Round-07 TANGO proof of concept.  It keeps the
observation model terminal-only: the attacker sees repeated terminal model
deltas from server-chosen initial classifier biases, not intermediate
minibatch gradients, batch order, incidence, or per-sample snapshots.
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
    seeds: tuple[int, ...] = (3, 7, 11, 19, 23, 29, 31, 37)

    @property
    def n_samples(self) -> int:
        return int(sum(self.class_counts))


SCENARIOS: dict[str, SimConfig] = {
    "balanced_clean": SimConfig(),
    "imbalanced_clean": SimConfig(class_counts=(4, 8, 12)),
    "noisy_terminal_1e-3": SimConfig(terminal_noise_std=1e-3),
    "weak_bias_poor_condition": SimConfig(bias_scale=0.08),
    "more_local_steps": SimConfig(local_steps=8),
    "high_within_class_variance": SimConfig(within_class_std=1.0),
}


def setup_logging() -> logging.Logger:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("round08")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(LOGS / "experiment_round08.log", mode="w")
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


def local_terminal_update(
    x: np.ndarray,
    labels: np.ndarray,
    w0: np.ndarray,
    b0: np.ndarray,
    cfg: SimConfig,
) -> tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    y = np.eye(cfg.n_classes, dtype=np.float64)[labels]
    w = w0.copy()
    b = b0.copy()
    for _ in range(cfg.local_steps):
        probs = softmax(x @ w + b[None, :])
        err = (probs - y) / n
        grad_w = x.T @ err
        grad_b = err.sum(axis=0)
        w -= cfg.lr * grad_w
        b -= cfg.lr * grad_b
    return w - w0, b - b0


def make_bias_probes(cfg: SimConfig, seed: int, n_rounds: int) -> np.ndarray:
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
    w0 = np.zeros((cfg.dim_x, cfg.n_classes), dtype=np.float64)
    for bias in biases:
        delta_w, delta_b = local_terminal_update(x, labels, w0, bias, cfg)
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
    return class_sums / counts[:, None], class_sums, counts


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


def aggregate(rows: Iterable[dict[str, float]], keys: list[str]) -> dict[str, float]:
    row_list = list(rows)
    out: dict[str, float] = {"n": float(len(row_list))}
    for key in keys:
        values = np.asarray([row[key] for row in row_list], dtype=np.float64)
        out[f"{key}_mean"] = float(values.mean())
        out[f"{key}_std"] = float(values.std(ddof=0))
    return out


def write_svg_plot(rows: list[dict[str, float]]) -> Path:
    path = ARTIFACTS / "round08_tango_stress.svg"
    width, height = 900, 430
    margin = 58
    scenario_names = [row["scenario"] for row in rows]
    proto = [row["best_active_prototype_mse_mean"] for row in rows]
    indiv = [row["best_active_individual_mse_mean"] for row in rows]
    floor = [row["within_class_variance_mean"] for row in rows]
    max_y = max(max(proto), max(indiv), max(floor)) * 1.12

    def sx(i: int) -> float:
        return margin + i * (width - 2 * margin) / max(1, len(rows) - 1)

    def sy(y: float) -> float:
        return height - margin - y / max_y * (height - 2 * margin)

    def poly(values: list[float], color: str) -> str:
        pts = " ".join(f"{sx(i):.1f},{sy(y):.1f}" for i, y in enumerate(values))
        circles = "\n".join(
            f'<circle cx="{sx(i):.1f}" cy="{sy(y):.1f}" r="4" fill="{color}" />'
            for i, y in enumerate(values)
        )
        return f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{pts}" />\n{circles}'

    x_labels = "\n".join(
        f'<text x="{sx(i):.1f}" y="{height-32}" transform="rotate(28 {sx(i):.1f},{height-32})" '
        f'text-anchor="start" font-size="10">{name}</text>'
        for i, name in enumerate(scenario_names)
    )
    y_ticks = []
    for i in range(6):
        y = max_y * i / 5.0
        y_ticks.append(
            f'<line x1="{margin-5}" x2="{width-margin}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="#eee" />'
        )
        y_ticks.append(
            f'<text x="{margin-10}" y="{sy(y)+4:.1f}" text-anchor="end" font-size="11">{y:.3f}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2:.1f}" y="24" text-anchor="middle" font-size="16" font-family="sans-serif">Round 08: TANGO stress tests, best active terminal probes</text>
{''.join(y_ticks)}
<line x1="{margin}" x2="{width-margin}" y1="{height-margin}" y2="{height-margin}" stroke="#333"/>
<line x1="{margin}" x2="{margin}" y1="{margin}" y2="{height-margin}" stroke="#333"/>
{x_labels}
<text transform="translate(16,{height/2:.1f}) rotate(-90)" text-anchor="middle" font-size="12">mean squared error</text>
{poly(proto, "#1f77b4")}
{poly(indiv, "#d62728")}
{poly(floor, "#2ca02c")}
<rect x="{width-268}" y="54" width="225" height="74" fill="white" stroke="#ddd"/>
<circle cx="{width-250}" cy="74" r="4" fill="#1f77b4"/><text x="{width-238}" y="78" font-size="12">prototype MSE</text>
<circle cx="{width-250}" cy="96" r="4" fill="#d62728"/><text x="{width-238}" y="100" font-size="12">individual via prototypes</text>
<circle cx="{width-250}" cy="118" r="4" fill="#2ca02c"/><text x="{width-238}" y="122" font-size="12">within-class floor</text>
</svg>
"""
    path.write_text(svg)
    return path


def run() -> dict[str, object]:
    logger = setup_logging()
    started = time.time()
    logger.info("Round 08 TANGO stress benchmark started")
    per_seed: list[dict[str, float | str]] = []

    for scenario, cfg in SCENARIOS.items():
        logger.info("scenario=%s config=%s", scenario, asdict(cfg))
        for seed in cfg.seeds:
            x, labels, true_centers = make_dataset(cfg, seed)
            true_counts = np.asarray(cfg.class_counts, dtype=np.float64)
            true_sums = np.vstack([x[labels == c].sum(axis=0) for c in range(cfg.n_classes)])
            true_proto = true_sums / true_counts[:, None]
            true_mean = x.mean(axis=0)
            within_class_var = float(
                np.mean(
                    [
                        np.mean((x[labels == c] - true_proto[c]) ** 2)
                        for c in range(cfg.n_classes)
                    ]
                )
            )
            for n_rounds in cfg.probe_rounds:
                biases = make_bias_probes(cfg, seed, n_rounds)
                delta_w, delta_b, probs = observe_terminal_updates(x, labels, biases, cfg, seed)
                est_proto, est_sums, est_counts = tango_estimate_sums(delta_w, delta_b, probs, cfg)
                proto_dataset = np.vstack(
                    [
                        np.repeat(est_proto[c][None, :], cfg.class_counts[c], axis=0)
                        for c in range(cfg.n_classes)
                    ]
                )
                row = {
                    "scenario": scenario,
                    "seed": float(seed),
                    "probe_rounds": float(n_rounds),
                    "observed_terminal_updates": float(n_rounds),
                    "observed_intermediate_batch_gradients": 0.0,
                    "prototype_mse": mean_squared(est_proto, true_proto),
                    "prototype_to_population_center_mse": mean_squared(est_proto, true_centers),
                    "dataset_mean_mse": mean_squared(est_sums.sum(axis=0) / cfg.n_samples, true_mean),
                    "count_mae": float(np.mean(np.abs(est_counts - true_counts))),
                    "individual_mse_from_prototypes": nearest_neighbor_mse(proto_dataset, x),
                    "within_class_variance": within_class_var,
                    "design_condition_number": design_condition_number(probs, cfg),
                }
                per_seed.append(row)
                logger.info(
                    "scenario=%s seed=%s rounds=%s proto_mse=%.6f indiv_mse=%.6f cond=%.2f",
                    scenario,
                    seed,
                    n_rounds,
                    row["prototype_mse"],
                    row["individual_mse_from_prototypes"],
                    row["design_condition_number"],
                )

    keys = [
        "prototype_mse",
        "dataset_mean_mse",
        "count_mae",
        "individual_mse_from_prototypes",
        "within_class_variance",
        "design_condition_number",
    ]
    by_scenario_round: list[dict[str, float | str]] = []
    for scenario in SCENARIOS:
        for n_rounds in SCENARIOS[scenario].probe_rounds:
            rows = [
                row
                for row in per_seed
                if row["scenario"] == scenario and row["probe_rounds"] == float(n_rounds)
            ]
            agg = aggregate(rows, keys)
            agg["scenario"] = scenario
            agg["probe_rounds"] = float(n_rounds)
            by_scenario_round.append(agg)

    scenario_headlines: list[dict[str, float | str]] = []
    for scenario in SCENARIOS:
        rows = [row for row in by_scenario_round if row["scenario"] == scenario]
        active_rows = [row for row in rows if row["probe_rounds"] >= 2.0]
        best = min(active_rows, key=lambda row: float(row["prototype_mse_mean"]))
        one = next(row for row in rows if row["probe_rounds"] == 1.0)
        scenario_headlines.append(
            {
                "scenario": scenario,
                "single_round_prototype_mse_mean": one["prototype_mse_mean"],
                "best_active_probe_rounds": best["probe_rounds"],
                "best_active_prototype_mse_mean": best["prototype_mse_mean"],
                "best_active_dataset_mean_mse_mean": best["dataset_mean_mse_mean"],
                "best_active_individual_mse_mean": best["individual_mse_from_prototypes_mean"],
                "within_class_variance_mean": best["within_class_variance_mean"],
                "best_active_condition_number_mean": best["design_condition_number_mean"],
                "active_vs_single_round_proto_gain_x": float(one["prototype_mse_mean"])
                / max(float(best["prototype_mse_mean"]), 1e-12),
            }
        )

    plot_path = write_svg_plot(scenario_headlines)
    balanced = next(row for row in scenario_headlines if row["scenario"] == "balanced_clean")
    noisy = next(row for row in scenario_headlines if row["scenario"] == "noisy_terminal_1e-3")
    weak_bias = next(row for row in scenario_headlines if row["scenario"] == "weak_bias_poor_condition")
    metrics: dict[str, object] = {
        "benchmark": "round08_tango_terminal_stress_and_tiered_limits",
        "method": "TANGO inside TALON: Terminal Aggregate Leakage with Observation tiers and Non-identifiability limits",
        "observation_model": {
            "observed": [
                "initial linear-head weights and server-chosen biases for each round",
                "terminal client model delta after local training",
                "public optimizer hyperparameters and architecture",
            ],
            "not_observed": [
                "intermediate minibatch gradients",
                "batch order",
                "batch incidence",
                "per-sample snapshots",
            ],
            "privacy_target": "class sums, class counts, dataset mean, and class prototypes; not individual samples",
        },
        "scenario_configs": {name: asdict(cfg) for name, cfg in SCENARIOS.items()},
        "per_seed": per_seed,
        "aggregate_by_scenario_round": by_scenario_round,
        "scenario_headlines": scenario_headlines,
        "headline": {
            "balanced_active_probe_rounds": balanced["best_active_probe_rounds"],
            "balanced_active_prototype_mse_mean": balanced["best_active_prototype_mse_mean"],
            "balanced_single_round_prototype_mse_mean": balanced["single_round_prototype_mse_mean"],
            "balanced_active_vs_single_round_proto_gain_x": balanced[
                "active_vs_single_round_proto_gain_x"
            ],
            "balanced_individual_mse_mean": balanced["best_active_individual_mse_mean"],
            "balanced_within_class_variance_mean": balanced["within_class_variance_mean"],
            "noisy_active_prototype_mse_mean": noisy["best_active_prototype_mse_mean"],
            "weak_bias_active_prototype_mse_mean": weak_bias["best_active_prototype_mse_mean"],
            "weak_bias_condition_number_mean": weak_bias["best_active_condition_number_mean"],
            "identifiability": "terminal aggregate probes identify class moments under rank conditions but not individual within-class deviations",
        },
        "plots": [str(plot_path.relative_to(RUN_ROOT))],
        "runtime_seconds": time.time() - started,
    }
    out_path = ARTIFACTS / "round08_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    logger.info("wrote %s", out_path)
    logger.info("wrote %s", plot_path)
    logger.info("headline=%s", metrics["headline"])
    return metrics


if __name__ == "__main__":
    run()
