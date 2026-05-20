#!/usr/bin/env python3
"""Round-07 benchmark: terminal-update aggregate inversion.

This experiment pivots away from SHARD's per-batch incidence recovery.  The
attacker observes only terminal client updates from several honest server-chosen
initial model states.  No intermediate minibatch gradients, no batch order, and
no incidence matrix are exposed.

The method, TANGO (Terminal Aggregate Neural Gradient Observation), uses the
first-order structure of a linear softmax head initialized with zero weights and
server-chosen biases.  Across two or more terminal updates with different
initial class probabilities, the aggregate weight update identifies class-level
feature sums.  That is enough to recover prototypes and dataset means, but not
individual samples inside each class.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


RUN_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = RUN_ROOT / "artifacts"
LOGS = RUN_ROOT / "logs"


@dataclass(frozen=True)
class SimConfig:
    n_classes: int = 3
    n_per_class: int = 8
    dim_x: int = 10
    prototype_scale: float = 2.0
    within_class_std: float = 0.55
    lr: float = 0.04
    local_steps: int = 3
    probe_rounds: tuple[int, ...] = (1, 2, 4, 8)
    seeds: tuple[int, ...] = (3, 7, 11, 19, 23, 29, 31, 37)

    @property
    def n_samples(self) -> int:
        return self.n_classes * self.n_per_class


def setup_logging() -> logging.Logger:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("round07")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(LOGS / "experiment_round07.log", mode="w")
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
    # Make separated but non-axis-aligned prototypes.
    q, _ = np.linalg.qr(raw.T)
    prototypes = cfg.prototype_scale * q[:, : cfg.n_classes].T
    labels = np.repeat(np.arange(cfg.n_classes), cfg.n_per_class)
    x = np.vstack(
        [
            prototypes[c]
            + cfg.within_class_std * rng.normal(size=(cfg.n_per_class, cfg.dim_x))
            for c in range(cfg.n_classes)
        ]
    )
    return x.astype(np.float64), labels.astype(np.int64), prototypes.astype(np.float64)


def local_terminal_update(
    x: np.ndarray,
    labels: np.ndarray,
    w0: np.ndarray,
    b0: np.ndarray,
    cfg: SimConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return terminal delta after full-batch local softmax training."""
    n = x.shape[0]
    c = cfg.n_classes
    y = np.eye(c, dtype=np.float64)[labels]
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
    probes = []
    # Include a neutral first round; later rounds are legal server initial biases.
    probes.append(np.zeros(cfg.n_classes, dtype=np.float64))
    while len(probes) < n_rounds:
        probes.append(1.2 * rng.normal(size=cfg.n_classes))
    return np.vstack(probes)


def observe_terminal_updates(
    x: np.ndarray,
    labels: np.ndarray,
    biases: np.ndarray,
    cfg: SimConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = cfg.n_classes
    d = cfg.dim_x
    deltas_w = []
    deltas_b = []
    probs = []
    w0 = np.zeros((d, c), dtype=np.float64)
    for bias in biases:
        delta_w, delta_b = local_terminal_update(x, labels, w0, bias, cfg)
        deltas_w.append(delta_w)
        deltas_b.append(delta_b)
        probs.append(class_prob_from_bias(bias))
    return np.stack(deltas_w), np.stack(deltas_b), np.stack(probs)


def estimate_counts(delta_b: np.ndarray, probs: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """Estimate class counts from terminal bias updates.

    For a first-order terminal update, -Delta_b / (lr * steps) = p - n_c / N.
    With small local lr this remains a good terminal approximation.
    """
    avg_grad_b = -delta_b / (cfg.lr * cfg.local_steps)
    counts_per_round = cfg.n_samples * (probs - avg_grad_b)
    counts = counts_per_round.mean(axis=0)
    counts = np.clip(counts, 1.0, cfg.n_samples)
    return counts


def tango_estimate_sums(
    delta_w: np.ndarray,
    delta_b: np.ndarray,
    probs: np.ndarray,
    cfg: SimConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover class sums from aggregate terminal updates.

    For each round r, class c, and feature k:
        N * g[r,c,k] = p[r,c] * S_total[k] - S_class[c,k]
    where g is the average weight gradient approximated from the terminal update.
    This is linear in S_total and S_class when at least two distinct probes are
    available.
    """
    r_count = delta_w.shape[0]
    c_count = cfg.n_classes
    d = cfg.dim_x
    avg_grad_w = -delta_w / (cfg.lr * cfg.local_steps)
    counts = estimate_counts(delta_b, probs, cfg)
    class_sums = np.zeros((c_count, d), dtype=np.float64)
    total = np.zeros(d, dtype=np.float64)

    for k in range(d):
        a_rows = []
        y_rows = []
        for r in range(r_count):
            for c in range(c_count):
                row = np.zeros(c_count + 1, dtype=np.float64)
                row[0] = probs[r, c]
                row[1 + c] = -1.0
                a_rows.append(row)
                y_rows.append(cfg.n_samples * avg_grad_w[r, k, c])
        # Enforce S_total = sum_c S_c, which resolves the rank deficiency.
        row = np.zeros(c_count + 1, dtype=np.float64)
        row[0] = 1.0
        row[1:] = -1.0
        a_rows.append(10.0 * row)
        y_rows.append(0.0)
        a = np.vstack(a_rows)
        y = np.asarray(y_rows)
        sol, *_ = np.linalg.lstsq(a, y, rcond=None)
        total[k] = sol[0]
        class_sums[:, k] = sol[1:]

    prototypes = class_sums / counts[:, None]
    return prototypes, class_sums, counts


def zero_probe_baseline(
    delta_w: np.ndarray,
    delta_b: np.ndarray,
    probs: np.ndarray,
    cfg: SimConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Single-neutral-update baseline: cannot identify absolute class sums."""
    avg_grad_w = -delta_w[:1] / (cfg.lr * cfg.local_steps)
    counts = estimate_counts(delta_b[:1], probs[:1], cfg)
    class_sums = np.zeros((cfg.n_classes, cfg.dim_x), dtype=np.float64)
    # Minimum-norm solution for g_c = p_c S_total / N - S_c / N assumes zero total.
    for c in range(cfg.n_classes):
        class_sums[c] = -cfg.n_samples * avg_grad_w[0, :, c]
    return class_sums / counts[:, None], counts


def nearest_neighbor_mse(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    """Greedy permutation-invariant matching MSE for small synthetic datasets."""
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


def update_residual(
    est_prototypes: np.ndarray,
    counts: np.ndarray,
    observed_delta_w: np.ndarray,
    probs: np.ndarray,
    cfg: SimConfig,
) -> float:
    class_sums = est_prototypes * counts[:, None]
    total = class_sums.sum(axis=0)
    pred_grad = np.zeros_like(observed_delta_w)
    for r in range(observed_delta_w.shape[0]):
        for c in range(cfg.n_classes):
            pred_grad[r, :, c] = (probs[r, c] * total - class_sums[c]) / cfg.n_samples
    pred_delta = -cfg.lr * cfg.local_steps * pred_grad
    return mean_squared(pred_delta, observed_delta_w)


def write_svg_plot(rows: list[dict[str, float]], cfg: SimConfig) -> Path:
    path = ARTIFACTS / "round07_terminal_update_metrics.svg"
    width, height = 760, 420
    margin = 56
    xs = [row["probe_rounds"] for row in rows]
    proto = [row["prototype_mse_mean"] for row in rows]
    indiv = [row["prototype_individual_mse_mean"] for row in rows]
    lower = [row["within_class_variance_mean"] for row in rows]
    max_y = max(max(proto), max(indiv), max(lower)) * 1.1
    min_x, max_x = min(xs), max(xs)

    def sx(x: float) -> float:
        if max_x == min_x:
            return margin
        return margin + (x - min_x) / (max_x - min_x) * (width - 2 * margin)

    def sy(y: float) -> float:
        return height - margin - y / max_y * (height - 2 * margin)

    def poly(values: list[float], color: str) -> str:
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, values))
        circles = "\n".join(
            f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{color}" />'
            for x, y in zip(xs, values)
        )
        return f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{pts}" />\n{circles}'

    ticks = "\n".join(
        f'<text x="{sx(x):.1f}" y="{height - 20}" text-anchor="middle" font-size="12">{int(x)}</text>'
        for x in xs
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
<text x="{width/2:.1f}" y="24" text-anchor="middle" font-size="16" font-family="sans-serif">Round 07: terminal-update aggregate inversion</text>
{''.join(y_ticks)}
<line x1="{margin}" x2="{width-margin}" y1="{height-margin}" y2="{height-margin}" stroke="#333"/>
<line x1="{margin}" x2="{margin}" y1="{margin}" y2="{height-margin}" stroke="#333"/>
{ticks}
<text x="{width/2:.1f}" y="{height-4}" text-anchor="middle" font-size="12">observed terminal server rounds</text>
<text transform="translate(16,{height/2:.1f}) rotate(-90)" text-anchor="middle" font-size="12">mean squared error</text>
{poly(proto, "#1f77b4")}
{poly(indiv, "#d62728")}
{poly(lower, "#2ca02c")}
<rect x="{width-255}" y="54" width="210" height="74" fill="white" stroke="#ddd"/>
<circle cx="{width-238}" cy="74" r="4" fill="#1f77b4"/><text x="{width-226}" y="78" font-size="12">prototype MSE</text>
<circle cx="{width-238}" cy="96" r="4" fill="#d62728"/><text x="{width-226}" y="100" font-size="12">individual via prototypes</text>
<circle cx="{width-238}" cy="118" r="4" fill="#2ca02c"/><text x="{width-226}" y="122" font-size="12">within-class lower bound</text>
</svg>
"""
    path.write_text(svg)
    return path


def run() -> dict[str, object]:
    logger = setup_logging()
    cfg = SimConfig()
    logger.info("Round 07 TANGO benchmark started")
    logger.info("config=%s", asdict(cfg))
    started = time.time()

    per_seed: list[dict[str, float]] = []
    for seed in cfg.seeds:
        x, labels, true_prototypes = make_dataset(cfg, seed)
        true_class_sums = np.vstack([x[labels == c].sum(axis=0) for c in range(cfg.n_classes)])
        true_empirical_prototypes = true_class_sums / cfg.n_per_class
        true_mean = x.mean(axis=0)
        within_class_var = float(
            np.mean(
                [
                    np.mean((x[labels == c] - true_empirical_prototypes[c]) ** 2)
                    for c in range(cfg.n_classes)
                ]
            )
        )
        for n_rounds in cfg.probe_rounds:
            biases = make_bias_probes(cfg, seed, n_rounds)
            delta_w, delta_b, probs = observe_terminal_updates(x, labels, biases, cfg)
            est_proto, est_sums, est_counts = tango_estimate_sums(delta_w, delta_b, probs, cfg)
            baseline_proto, baseline_counts = zero_probe_baseline(delta_w, delta_b, probs, cfg)
            proto_dataset = np.vstack(
                [np.repeat(est_proto[c][None, :], cfg.n_per_class, axis=0) for c in range(cfg.n_classes)]
            )
            baseline_dataset = np.vstack(
                [
                    np.repeat(baseline_proto[c][None, :], cfg.n_per_class, axis=0)
                    for c in range(cfg.n_classes)
                ]
            )
            per_seed.append(
                {
                    "seed": seed,
                    "probe_rounds": n_rounds,
                    "observed_terminal_updates": n_rounds,
                    "observed_intermediate_batch_gradients": 0,
                    "prototype_mse": mean_squared(est_proto, true_empirical_prototypes),
                    "prototype_to_true_center_mse": mean_squared(est_proto, true_prototypes),
                    "dataset_mean_mse": mean_squared(est_sums.sum(axis=0) / cfg.n_samples, true_mean),
                    "count_mae": float(np.mean(np.abs(est_counts - cfg.n_per_class))),
                    "terminal_update_residual_mse": update_residual(est_proto, est_counts, delta_w, probs, cfg),
                    "prototype_individual_mse": nearest_neighbor_mse(proto_dataset, x),
                    "single_round_baseline_proto_mse": mean_squared(
                        baseline_proto, true_empirical_prototypes
                    ),
                    "single_round_baseline_individual_mse": nearest_neighbor_mse(
                        baseline_dataset, x
                    ),
                    "within_class_variance": within_class_var,
                }
            )
            logger.info(
                "seed=%s rounds=%s proto_mse=%.6f mean_mse=%.6f indiv_mse=%.6f residual=%.8f",
                seed,
                n_rounds,
                per_seed[-1]["prototype_mse"],
                per_seed[-1]["dataset_mean_mse"],
                per_seed[-1]["prototype_individual_mse"],
                per_seed[-1]["terminal_update_residual_mse"],
            )

    aggregate = []
    for n_rounds in cfg.probe_rounds:
        rows = [row for row in per_seed if row["probe_rounds"] == n_rounds]
        agg: dict[str, float] = {
            "probe_rounds": float(n_rounds),
            "n_seeds": float(len(rows)),
        }
        keys = [
            "prototype_mse",
            "prototype_to_true_center_mse",
            "dataset_mean_mse",
            "count_mae",
            "terminal_update_residual_mse",
            "prototype_individual_mse",
            "single_round_baseline_proto_mse",
            "single_round_baseline_individual_mse",
            "within_class_variance",
        ]
        for key in keys:
            values = np.asarray([row[key] for row in rows], dtype=np.float64)
            agg[f"{key}_mean"] = float(values.mean())
            agg[f"{key}_std"] = float(values.std(ddof=0))
        aggregate.append(agg)

    plot_path = write_svg_plot(aggregate, cfg)
    best = min(aggregate, key=lambda row: row["prototype_mse_mean"])
    metrics: dict[str, object] = {
        "benchmark": "round07_terminal_update_aggregate_inversion",
        "method": "TANGO: Terminal Aggregate Neural Gradient Observation",
        "observation_model": {
            "observed": [
                "initial global linear-head weights and biases for each server round",
                "terminal client model delta after local full-batch training",
                "public optimizer hyperparameters and model architecture",
            ],
            "not_observed": [
                "intermediate minibatch gradients",
                "batch order",
                "per-batch incidence",
                "per-sample snapshots",
            ],
            "active_server_choice": "server chooses honest initial bias vectors before each round; no gradient tampering or protocol-breaking client changes",
        },
        "config": asdict(cfg),
        "per_seed": per_seed,
        "aggregate": aggregate,
        "comparison_vs_shard_same_observations": {
            "shard_stage2_status": "not applicable: same observations contain no batch-average rows or incidence matrix",
            "individual_identifiability": "not identifiable from class sums; all datasets with the same class sums produce the same first-order terminal updates",
            "best_defensible_target": "class prototypes, class counts, and dataset mean rather than individual samples",
        },
        "headline": {
            "best_probe_rounds": int(best["probe_rounds"]),
            "best_prototype_mse_mean": best["prototype_mse_mean"],
            "best_dataset_mean_mse_mean": best["dataset_mean_mse_mean"],
            "best_individual_mse_mean": best["prototype_individual_mse_mean"],
            "within_class_variance_mean": best["within_class_variance_mean"],
            "viability": "viable as a weaker aggregate/prototype leakage contribution; not viable as individual sample recovery",
        },
        "plots": [str(plot_path.relative_to(RUN_ROOT))],
        "runtime_seconds": time.time() - started,
    }
    out_path = ARTIFACTS / "round07_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    logger.info("wrote %s", out_path)
    logger.info("wrote %s", plot_path)
    logger.info("headline=%s", metrics["headline"])
    return metrics


if __name__ == "__main__":
    run()
