#!/usr/bin/env python3
"""Round-13 benchmark: fundamental minibatch directions beyond T_eff bookkeeping.

New directions (tested with code + metrics):
- TANGO-JOINT: joint ridge LS over (counts, class sums) from all probe rounds
- TANGO-COUPLED: fixed-point iteration on (W_hat, bias moments) with MB scaling
- TANGO-DOPT: joint inversion with round weights from probe conditioning
- passive_mb_scale_only: honest R11 scaling ablation (decoupled from MB count heuristic)
- trajectory_midpoint: average W0/W_M bias-gradient correction within one local step

Keeps minibatch_sgd as Phase-2 primary; median + IQR reporting.
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
    logger = logging.getLogger("round13")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(LOGS / "experiment_round13.log", mode="w")
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


def passive_mb_scale_only_estimate_sums(
    delta_w: np.ndarray,
    delta_b: np.ndarray,
    probs: np.ndarray,
    cfg: SimConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Round-11 honest scaling-only ablation: T_eff scale + vanilla count/LS."""
    dw, db = scale_terminal_deltas(delta_w, delta_b, cfg)
    return tango_estimate_sums(dw, db, probs, cfg)


def round_weights_dopt(probs: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """Inverse conditioning weight: rounds near uniform get higher weight."""
    uniform = np.ones(cfg.n_classes, dtype=np.float64) / cfg.n_classes
    dist = np.linalg.norm(probs - uniform, axis=1)
    w = 1.0 / (dist + 0.08)
    return w / w.sum()


def joint_counts_from_bias(
    delta_b_scaled: np.ndarray,
    probs: np.ndarray,
    cfg: SimConfig,
    ridge: float = 1e-3,
    round_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Joint ML over probe rounds for counts (bias moments only)."""
    eff = effective_gradient_steps(cfg)
    n = cfg.n_samples
    c = cfg.n_classes
    r_n = delta_b_scaled.shape[0]
    avg_gb = -delta_b_scaled / (cfg.lr * eff)
    if round_weights is None:
        round_weights = np.ones(r_n, dtype=np.float64) / r_n

    rows: list[np.ndarray] = []
    targets: list[float] = []
    for r in range(r_n):
        wr = float(round_weights[r])
        for cls in range(c):
            row = np.zeros(c, dtype=np.float64)
            row[cls] = wr
            rows.append(row)
            targets.append(wr * n * (probs[r, cls] - avg_gb[r, cls]))

    a_mat = np.vstack(rows)
    y_vec = np.asarray(targets, dtype=np.float64)
    sol, *_ = np.linalg.lstsq(a_mat, y_vec, rcond=None)
    sol = np.clip(sol, 1.0, n)
    sol = sol * (n / max(sol.sum(), 1e-9))
    return sol


def joint_mb_moment_invert(
    delta_w_scaled: np.ndarray,
    delta_b_scaled: np.ndarray,
    probs: np.ndarray,
    cfg: SimConfig,
    ridge: float = 1e-2,
    round_weights: np.ndarray | None = None,
    counts: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Jointly estimate counts and class sums from stacked bias + weight moments."""
    n = cfg.n_samples
    c = cfg.n_classes
    d = cfg.dim_x
    r_n = delta_w_scaled.shape[0]
    eff = effective_gradient_steps(cfg)
    avg_gw = -delta_w_scaled / (cfg.lr * eff)
    if round_weights is None:
        round_weights = np.ones(r_n, dtype=np.float64) / r_n
    if counts is None:
        counts = joint_counts_from_bias(
            delta_b_scaled, probs, cfg, round_weights=round_weights
        )

    n_vars = c * d
    rows: list[np.ndarray] = []
    targets: list[float] = []

    for k in range(d):
        for r in range(r_n):
            wr = float(round_weights[r])
            for cls in range(c):
                row = np.zeros(n_vars, dtype=np.float64)
                for cp in range(c):
                    row[cp * d + k] = probs[r, cls]
                row[cls * d + k] -= 1.0
                rows.append(wr * row)
                targets.append(wr * n * avg_gw[r, k, cls])
        row = np.zeros(n_vars, dtype=np.float64)
        for cls in range(c):
            row[cls * d + k] = 10.0
        rows.append(row)
        targets.append(0.0)

    a_mat = np.vstack(rows)
    y_vec = np.asarray(targets, dtype=np.float64)
    reg = ridge * 0.05 * np.eye(n_vars)
    sol = np.linalg.solve(a_mat.T @ a_mat + reg, a_mat.T @ y_vec)
    class_sums = sol.reshape(c, d)
    prototypes = class_sums / counts[:, None]
    return prototypes, class_sums, counts


def tango_joint_estimate_sums(
    delta_w: np.ndarray,
    delta_b: np.ndarray,
    probs: np.ndarray,
    cfg: SimConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TANGO-JOINT: joint counts from all bias rounds, then joint weight LS."""
    dw, db = scale_terminal_deltas(delta_w, delta_b, cfg)
    w = round_weights_dopt(probs, cfg)
    return joint_mb_moment_invert(dw, db, probs, cfg, round_weights=w)


def tango_dopt_estimate_sums(
    delta_w: np.ndarray,
    delta_b: np.ndarray,
    probs: np.ndarray,
    cfg: SimConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dw, db = scale_terminal_deltas(delta_w, delta_b, cfg)
    w = round_weights_dopt(probs, cfg)
    return joint_mb_moment_invert(dw, db, probs, cfg, round_weights=w)


def mean_prob_at_weights(
    x: np.ndarray, labels: np.ndarray, w: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    probs = softmax(x @ w + bias[None, :])
    out = np.zeros(probs.shape[1], dtype=np.float64)
    for cls in range(probs.shape[1]):
        mask = labels == cls
        if mask.any():
            out[cls] = probs[mask, cls].mean()
        else:
            out[cls] = probs[:, cls].mean()
    return out


def within_step_midpoint_weights(
    x: np.ndarray, labels: np.ndarray, cfg: SimConfig, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Terminal trajectory: W_mid = (W_0 + W_M)/2 after one minibatch pass."""
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
    w_mid = 0.5 * (w0 + w)
    return w0, w_mid


def tango_trajectory_midpoint_estimate_sums(
    delta_w: np.ndarray,
    delta_b: np.ndarray,
    probs: np.ndarray,
    cfg: SimConfig,
    x: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bias moments corrected using avg gradient at W0 and W_mid within one step."""
    dw_s, db_s = scale_terminal_deltas(delta_w, delta_b, cfg)
    w0, w_mid = within_step_midpoint_weights(x, labels, cfg, seed)
    db_corr = db_s.copy()
    beta = 0.25
    for r in range(delta_w.shape[0]):
        bias_r = np.log(np.maximum(probs[r], 1e-12))
        p0 = mean_prob_at_weights(x, labels, w0, bias_r)
        pm = mean_prob_at_weights(x, labels, w_mid, bias_r)
        g0 = -db_s[r] / (cfg.lr * cfg.local_steps)
        g_mid = g0 + beta * (0.5 * (pm + p0) - probs[r])
        db_corr[r] = -cfg.lr * cfg.local_steps * g_mid
    return joint_mb_moment_invert(dw_s, db_corr, probs, cfg)


def tango_coupled_estimate_sums(
    delta_w: np.ndarray,
    delta_b: np.ndarray,
    probs: np.ndarray,
    cfg: SimConfig,
    x: np.ndarray,
    labels: np.ndarray,
    seed: int,
    n_iter: int = 3,
    alpha: float = 0.18,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coupled drift iteration: alternate joint (S,n) and W_hat bias correction."""
    dw_s, db_s = scale_terminal_deltas(delta_w, delta_b, cfg)
    eff = effective_gradient_steps(cfg)
    db_work = db_s.copy()
    proto = np.zeros((cfg.n_classes, cfg.dim_x), dtype=np.float64)
    counts = np.full(cfg.n_classes, cfg.n_samples / cfg.n_classes)

    w0, _ = init_head(cfg, seed)
    for _ in range(n_iter):
        proto, _, counts = joint_mb_moment_invert(dw_s, db_work, probs, cfg)
        avg_grad_w = -dw_s / (cfg.lr * eff)
        w_hat = w0 - cfg.lr * eff * avg_grad_w.mean(axis=0)
        db_corr = db_work.copy()
        for r in range(delta_w.shape[0]):
            bias_r = np.log(np.maximum(probs[r], 1e-12))
            p_tilde = mean_prob_at_weights(x, labels, w_hat, bias_r)
            g0 = -db_work[r] / (cfg.lr * cfg.local_steps)
            g_corr = g0 + alpha * (p_tilde - probs[r])
            db_corr[r] = -cfg.lr * cfg.local_steps * g_corr
        db_work = db_corr

    _, sums, counts = joint_mb_moment_invert(dw_s, db_work, probs, cfg)
    return proto, sums, counts


def within_step_weight_drift(
    x: np.ndarray, labels: np.ndarray, cfg: SimConfig, seed: int
) -> float:
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
    "tango_joint",
    "tango_coupled",
    "tango_dopt",
    "tango_trajectory_midpoint",
    "passive_multi_round",
    "passive_mb_scale_only",
    "passive_mb_r12_coupled",
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
    if method == "tango_joint":
        proto, _, counts = tango_joint_estimate_sums(delta_w, delta_b, probs, cfg)
        return proto, counts
    if method == "tango_coupled":
        proto, _, counts = tango_coupled_estimate_sums(
            delta_w, delta_b, probs, cfg, x, labels, seed
        )
        return proto, counts
    if method == "tango_dopt":
        proto, _, counts = tango_dopt_estimate_sums(delta_w, delta_b, probs, cfg)
        return proto, counts
    if method == "tango_trajectory_midpoint":
        proto, _, counts = tango_trajectory_midpoint_estimate_sums(
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
    if method == "passive_mb_r12_coupled":
        dw, db = scale_terminal_deltas(delta_w, delta_b, cfg)
        proto, _, counts = tango_mb_from_scaled(dw, db, probs, cfg)
        return proto, counts
    if method == "public_prior":
        return public_prior_prototypes(delta_w[:1], probs[:1], cfg), np.full(
            cfg.n_classes, cfg.n_samples / cfg.n_classes
        )
    if method == "oracle_aggregate":
        return oracle_aggregate_prototypes(true_sums, true_counts), true_counts.copy()
    raise ValueError(f"unknown method: {method}")


def write_minibatch_svg(rows: list[dict[str, float | str]]) -> Path:
    path = ARTIFACTS / "round13_minibatch_methods.svg"
    width, height = 960, 460
    margin = 62
    method_keys = {
        "tango_mb": "tango_mb_prototype_mse_median",
        "tango_joint": "tango_joint_prototype_mse_median",
        "tango_coupled": "tango_coupled_prototype_mse_median",
        "passive_multi_round": "passive_multi_round_prototype_mse_median",
        "passive_mb_scale_only": "passive_mb_scale_only_prototype_mse_median",
    }
    colors = {
        "tango_mb": "#2ca02c",
        "tango_joint": "#1f77b4",
        "tango_coupled": "#9467bd",
        "passive_multi_round": "#ff7f0e",
        "passive_mb_scale_only": "#8c564b",
    }
    labels = {
        "tango_mb": "TANGO-MB",
        "tango_joint": "TANGO-JOINT",
        "tango_coupled": "TANGO-COUPLED",
        "passive_multi_round": "passive",
        "passive_mb_scale_only": "scale-only (R11)",
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
<text x="{width/2:.1f}" y="22" text-anchor="middle" font-size="14" font-family="sans-serif">Round 13: prototype MSE (median over seeds)</text>
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
    logger.info("Round 13 Phase-2 new-directions benchmark started")

    per_seed: list[dict[str, float | str]] = []
    drift_samples: list[float] = []

    for scenario, cfg in SCENARIOS.items():
        eff = effective_gradient_steps(cfg)
        logger.info(
            "scenario=%s effective_gradient_steps=%.4f",
            scenario,
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
                "tango_joint": (dw_a, db_a, probs_a),
                "tango_coupled": (dw_a, db_a, probs_a),
                "tango_dopt": (dw_a, db_a, probs_a),
                "tango_trajectory_midpoint": (dw_a, db_a, probs_a),
                "passive_multi_round": (dw_p, db_p, probs_p),
                "passive_mb_scale_only": (dw_p, db_p, probs_p),
                "passive_mb_r12_coupled": (dw_p, db_p, probs_p),
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
        scale_only_mse = float(
            entry["passive_mb_scale_only_prototype_mse_mean"]
        )
        tango_joint_mse = float(entry["tango_joint_prototype_mse_mean"])
        entry["tango_mb_vs_passive_gain_x"] = passive_mse / max(tango_mb_mse, 1e-12)
        entry["phase2_primary_win"] = float(tango_mb_mse < passive_mse)
        entry["active_over_r11_scaling_only_x"] = scale_only_mse / max(
            tango_mb_mse, 1e-12
        )
        entry["joint_vs_tango_mb_x"] = tango_mb_mse / max(tango_joint_mse, 1e-12)
        entry["passive_mb_scale_only_mse"] = scale_only_mse
        entry["passive_mb_r12_coupled_mse"] = float(
            entry["passive_mb_r12_coupled_prototype_mse_mean"]
        )
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
        "benchmark": "round13_phase2_new_directions",
        "phase2_goal": (
            "Test joint moment inversion, coupled drift iteration, honest "
            "passive_mb_scale_only; fix proof doc drift; keep minibatch_sgd primary"
        ),
        "new_directions": {
            "tango_joint": "Joint ridge LS on (counts, class sums) over all rounds",
            "tango_coupled": "3-iter fixed point: joint invert + Jacobian bias at W_hat",
            "tango_dopt": "Joint invert with D-optimal round weights (1/||p-uniform||)",
            "tango_trajectory_midpoint": (
                "W_mid=(W0+WM)/2 bias correction then joint invert"
            ),
            "passive_mb_scale_only": (
                "R11 honest ablation: scale_terminal_deltas + tango_estimate_sums"
            ),
            "passive_mb_r12_coupled": (
                "R12 degraded passive_mb (MB count heuristic coupled) for comparison"
            ),
        },
        "methods": {
            "tango_mb": "Round-12 primary (T_eff + uniform-round count)",
            "tango_joint": "TANGO-JOINT stacked moments",
            "tango_coupled": "TANGO-COUPLED drift iteration",
            "passive_mb_scale_only": "Scaling-only R11 path",
        },
        "lemma_mb_b_empirical": {
            "within_step_weight_drift_mean": drift_mean,
            "within_step_weight_drift_std": drift_std,
            "note": "Corrected from erroneous 0.42 in proofs.md Round 12",
        },
        "active_vs_scaling_primary": {
            "passive_mb_scale_only_prototype_mse": primary[
                "passive_mb_scale_only_prototype_mse_mean"
            ],
            "passive_mb_r12_coupled_prototype_mse": primary[
                "passive_mb_r12_coupled_prototype_mse_mean"
            ],
            "tango_mb_prototype_mse": primary["tango_mb_prototype_mse_mean"],
            "tango_joint_prototype_mse": primary["tango_joint_prototype_mse_mean"],
            "tango_coupled_prototype_mse": primary[
                "tango_coupled_prototype_mse_mean"
            ],
            "passive_prototype_mse": primary["passive_multi_round_prototype_mse_mean"],
            "gain_active_over_r11_scaling_only_x": primary[
                "active_over_r11_scaling_only_x"
            ],
            "gain_joint_vs_tango_mb_x": primary["joint_vs_tango_mb_x"],
        },
        "round12_comparison_primary": {
            "note": "Round 12 values from artifacts/round12_metrics.json",
            "r12_tango_mb_mse_mean": 0.011025348154678465,
            "r12_tango_mb_mse_median": 0.011406367115738615,
            "r12_passive_mse_mean": 0.7532224427330506,
            "r12_passive_mb_scale_only_proxy_stack_ridge": 0.1500,
            "r12_passive_mb_coupled_mse_mean": 0.2838866057759546,
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
            "passive_mb_r12_coupled_prototype_mse_mean": primary[
                "passive_mb_r12_coupled_prototype_mse_mean"
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
            "tango_joint_prototype_mse_iqr": primary["tango_joint_prototype_mse_iqr"],
            "tango_coupled_prototype_mse_mean": primary[
                "tango_coupled_prototype_mse_mean"
            ],
            "tango_coupled_prototype_mse_median": primary[
                "tango_coupled_prototype_mse_median"
            ],
            "tango_dopt_prototype_mse_mean": primary[
                "tango_dopt_prototype_mse_mean"
            ],
            "tango_trajectory_midpoint_prototype_mse_mean": primary[
                "tango_trajectory_midpoint_prototype_mse_mean"
            ],
            "tango_mb_count_mae_mean": primary["tango_mb_count_mae_mean"],
            "tango_joint_count_mae_mean": primary["tango_joint_count_mae_mean"],
            "passive_count_mae_mean": primary["passive_multi_round_count_mae_mean"],
            "count_mae_beats_passive": float(
                primary["tango_mb_count_mae_mean"]
                < primary["passive_multi_round_count_mae_mean"]
            ),
            "tango_mb_vs_passive_gain_x": primary["tango_mb_vs_passive_gain_x"],
            "active_over_r11_scaling_only_x": primary[
                "active_over_r11_scaling_only_x"
            ],
            "phase2_primary_win": primary["phase2_primary_win"],
            "best_new_direction": None,
        },
        "headline": {},
        "plots": [str(plot_path.relative_to(RUN_ROOT))],
        "runtime_seconds": time.time() - started,
    }

    pp = metrics["phase2_primary"]
    assert isinstance(pp, dict)
    tango_mb_mse = float(pp["tango_mb_prototype_mse_mean"])
    candidates = {
        "tango_joint": float(pp["tango_joint_prototype_mse_mean"]),
        "tango_coupled": float(pp["tango_coupled_prototype_mse_mean"]),
        "tango_dopt": float(pp["tango_dopt_prototype_mse_mean"]),
        "tango_trajectory_midpoint": float(
            pp["tango_trajectory_midpoint_prototype_mse_mean"]
        ),
    }
    least_bad = min(candidates, key=candidates.get)
    pp["best_new_direction"] = least_bad
    pp["any_new_direction_beats_tango_mb"] = float(
        any(v < tango_mb_mse for v in candidates.values())
    )
    pp["least_bad_new_direction_mse"] = candidates[least_bad]

    metrics["headline"] = {
        "minibatch_tango_mb_mse_mean": pp["tango_mb_prototype_mse_mean"],
        "minibatch_tango_mb_mse_median": pp["tango_mb_prototype_mse_median"],
        "minibatch_tango_joint_mse_mean": pp["tango_joint_prototype_mse_mean"],
        "minibatch_tango_joint_mse_median": pp["tango_joint_prototype_mse_median"],
        "minibatch_passive_mse_mean": pp["passive_prototype_mse_mean"],
        "minibatch_passive_mb_scale_only_mse_mean": pp[
            "passive_mb_scale_only_prototype_mse_mean"
        ],
        "minibatch_phase2_win": pp["phase2_primary_win"],
        "balanced_tango_mb_mse_mean": balanced["tango_mb_prototype_mse_mean"],
        "best_new_direction": pp["best_new_direction"],
    }

    out_path = ARTIFACTS / "round13_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    logger.info("wrote %s", out_path)
    logger.info("phase2_primary=%s", metrics["phase2_primary"])
    logger.info("headline=%s", metrics["headline"])
    logger.info("lemma_mb_b_empirical=%s", metrics["lemma_mb_b_empirical"])
    return metrics


if __name__ == "__main__":
    run()
