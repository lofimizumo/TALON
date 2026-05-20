#!/usr/bin/env python3
"""Round-10 benchmark: manuscript-closing experiments for TALON/TANGO.

Adds:
- Synthetic decoder / semantic probe (linear hidden -> pixel map)
- Frozen nonlinear backbone (random MLP features, head-only training)
- Re-run balanced_clean + minibatch_sgd for decoder comparison
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np

RUN_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = RUN_ROOT / "artifacts"
LOGS = RUN_ROOT / "logs"

# Reuse Round-09 simulator core
_spec = importlib.util.spec_from_file_location(
    "benchmark_round09",
    RUN_ROOT / "code" / "benchmark_round09.py",
)
_r09 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _r09
assert _spec.loader is not None
_spec.loader.exec_module(_r09)

SimConfig = _r09.SimConfig
make_dataset = _r09.make_dataset
init_head = _r09.init_head
make_bias_probes = _r09.make_bias_probes
observe_terminal_updates = _r09.observe_terminal_updates
tango_estimate_sums = _r09.tango_estimate_sums
mean_squared = _r09.mean_squared
count_metrics = _r09.count_metrics
aggregate = _r09.aggregate
evaluate_method = _r09.evaluate_method
nearest_neighbor_mse = _r09.nearest_neighbor_mse


def setup_logging() -> logging.Logger:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("round10")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(LOGS / "experiment_round10.log", mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def make_linear_decoder(
    cfg: SimConfig, seed: int, dim_pixel: int
) -> np.ndarray:
    """Fixed linear decoder D: (dim_x,) -> (dim_pixel,)."""
    rng = np.random.default_rng(seed + 88_000)
    raw = rng.normal(size=(cfg.dim_x, dim_pixel))
    q, _ = np.linalg.qr(raw)
    return q[:, :dim_pixel].astype(np.float64)


def decode_prototypes(decoder: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    return prototypes @ decoder


def prototype_correlation(est: np.ndarray, true: np.ndarray) -> float:
    """Mean per-class Pearson correlation across feature dimensions."""
    corrs = []
    for c in range(est.shape[0]):
        a = est[c] - est[c].mean()
        b = true[c] - true[c].mean()
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-12:
            corrs.append(1.0 if np.allclose(est[c], true[c]) else 0.0)
        else:
            corrs.append(float(np.dot(a, b) / denom))
    return float(np.mean(corrs))


def fit_linear_decoder_lstsq(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares D in y ≈ x @ D."""
    d_out = y.shape[1]
    d_in = x.shape[1]
    d_hat = np.zeros((d_in, d_out), dtype=np.float64)
    for j in range(d_out):
        coef, *_ = np.linalg.lstsq(x, y[:, j], rcond=None)
        d_hat[:, j] = coef
    return d_hat


def frozen_mlp_features(x: np.ndarray, seed: int, hidden_dim: int = 24) -> np.ndarray:
    """Two-layer tanh MLP with frozen random weights (representation only)."""
    rng = np.random.default_rng(seed + 66_000)
    n, d_in = x.shape
    w1 = rng.normal(scale=0.45, size=(d_in, hidden_dim))
    b1 = rng.normal(scale=0.05, size=(hidden_dim,))
    w2 = rng.normal(scale=0.35, size=(hidden_dim, d_in))
    b2 = rng.normal(scale=0.05, size=(d_in,))
    h = np.tanh(x @ w1 + b1)
    return (h @ w2 + b2).astype(np.float64)


def local_terminal_update_on_features(
    x: np.ndarray,
    labels: np.ndarray,
    w0: np.ndarray,
    b0: np.ndarray,
    cfg: SimConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Same as round09 local_terminal_update but on arbitrary feature matrix x."""
    n = x.shape[0]
    y = np.eye(cfg.n_classes, dtype=np.float64)[labels]
    w = w0.copy()
    b = b0.copy()
    rng = np.random.default_rng(seed + 77_000)
    for _ in range(cfg.local_steps):
        if cfg.use_minibatch:
            idx = rng.permutation(n)
            for start in range(0, n, cfg.minibatch_size):
                batch_idx = idx[start : start + cfg.minibatch_size]
                xb = x[batch_idx]
                yb = y[batch_idx]
                nb = xb.shape[0]
                probs = _r09.softmax(xb @ w + b[None, :])
                err = (probs - yb) / nb
                w -= cfg.lr * (xb.T @ err)
                b -= cfg.lr * err.sum(axis=0)
        else:
            probs = _r09.softmax(x @ w + b[None, :])
            err = (probs - y) / n
            w -= cfg.lr * (x.T @ err)
            b -= cfg.lr * err.sum(axis=0)
    return w - w0, b - b0


def observe_terminal_updates_features(
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
        dw, db = local_terminal_update_on_features(
            x, labels, w0, bias, cfg, seed + 100 * r
        )
        deltas_w.append(dw)
        deltas_b.append(db)
        probs.append(_r09.class_prob_from_bias(bias))
    return np.stack(deltas_w), np.stack(deltas_b), np.stack(probs)


def run_decoder_scenario(
    scenario: str,
    cfg: SimConfig,
    dim_pixel: int,
    logger: logging.Logger,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    n_rounds = max(cfg.probe_rounds)
    for seed in cfg.seeds:
        x, labels, true_centers = make_dataset(cfg, seed)
        decoder = make_linear_decoder(cfg, seed, dim_pixel)
        y_pixels = x @ decoder
        true_counts = np.asarray(cfg.class_counts, dtype=np.float64)
        true_sums = np.vstack([x[labels == c].sum(axis=0) for c in range(cfg.n_classes)])
        true_proto = true_sums / true_counts[:, None]
        true_pixel_proto = decode_prototypes(decoder, true_proto)

        biases = make_bias_probes(cfg, seed, n_rounds, passive=False)
        dw, db, probs = observe_terminal_updates(x, labels, biases, cfg, seed)
        est_proto, _, est_counts = tango_estimate_sums(dw, db, probs, cfg)
        est_pixel_proto = decode_prototypes(decoder, est_proto)

        # Decoder trained on all samples (oracle feature access — upper bound for semantics)
        d_oracle = fit_linear_decoder_lstsq(x, y_pixels)
        oracle_pixel_from_est = est_proto @ d_oracle
        oracle_pixel_from_true = true_proto @ d_oracle

        row = {
            "experiment": "decoder_probe",
            "scenario": scenario,
            "seed": float(seed),
            "dim_pixel": float(dim_pixel),
            "hidden_prototype_mse": mean_squared(est_proto, true_proto),
            "hidden_prototype_corr": prototype_correlation(est_proto, true_proto),
            "pixel_class_mean_mse": mean_squared(est_pixel_proto, true_pixel_proto),
            "pixel_class_mean_corr": prototype_correlation(
                est_pixel_proto, true_pixel_proto
            ),
            "decoder_lstsq_pixel_mse_from_est": mean_squared(
                oracle_pixel_from_est, true_pixel_proto
            ),
            "decoder_lstsq_pixel_mse_from_true": mean_squared(
                oracle_pixel_from_true, true_pixel_proto
            ),
            "count_mae": count_metrics(est_counts, true_counts)["count_mae"],
            "individual_mse_from_prototypes": nearest_neighbor_mse(
                np.vstack(
                    [
                        np.repeat(est_proto[c][None, :], cfg.class_counts[c], axis=0)
                        for c in range(cfg.n_classes)
                    ]
                ),
                x,
            ),
        }
        rows.append(row)
        logger.info(
            "decoder scenario=%s seed=%s hidden_mse=%.6f pixel_mse=%.6f corr=%.4f",
            scenario,
            seed,
            row["hidden_prototype_mse"],
            row["pixel_class_mean_mse"],
            row["hidden_prototype_corr"],
        )
    return rows


def run_frozen_mlp_scenario(
    scenario: str,
    cfg: SimConfig,
    logger: logging.Logger,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    n_rounds = max(cfg.probe_rounds)
    hidden_dim = 24
    for seed in cfg.seeds:
        x_raw, labels, _ = make_dataset(cfg, seed)
        x = frozen_mlp_features(x_raw, seed, hidden_dim=hidden_dim)
        true_counts = np.asarray(cfg.class_counts, dtype=np.float64)
        true_sums = np.vstack([x[labels == c].sum(axis=0) for c in range(cfg.n_classes)])
        true_proto = true_sums / true_counts[:, None]

        biases = make_bias_probes(cfg, seed, n_rounds, passive=False)
        dw, db, probs = observe_terminal_updates_features(
            x, labels, biases, cfg, seed
        )
        est_proto, _, est_counts = tango_estimate_sums(dw, db, probs, cfg)

        # Passive baseline in phi-space
        biases_p = make_bias_probes(cfg, seed, n_rounds, passive=True)
        dw_p, db_p, probs_p = observe_terminal_updates_features(
            x, labels, biases_p, cfg, seed + 1
        )
        est_proto_p, _, _ = tango_estimate_sums(dw_p, db_p, probs_p, cfg)

        row = {
            "experiment": "frozen_mlp_backbone",
            "scenario": scenario,
            "seed": float(seed),
            "mlp_hidden_dim": float(hidden_dim),
            "tango_prototype_mse": mean_squared(est_proto, true_proto),
            "passive_prototype_mse": mean_squared(est_proto_p, true_proto),
            "count_mae": count_metrics(est_counts, true_counts)["count_mae"],
            "tango_vs_passive_gain_x": float(
                mean_squared(est_proto_p, true_proto)
                / max(mean_squared(est_proto, true_proto), 1e-12)
            ),
        }
        rows.append(row)
        logger.info(
            "frozen_mlp scenario=%s seed=%s tango_mse=%.6f passive_mse=%.6f gain=%.1fx",
            scenario,
            seed,
            row["tango_prototype_mse"],
            row["passive_prototype_mse"],
            row["tango_vs_passive_gain_x"],
        )
    return rows


def write_decoder_svg(headlines: list[dict[str, float | str]]) -> Path:
    path = ARTIFACTS / "round10_decoder_probe.svg"
    width, height = 720, 400
    margin = 58
    scenarios = [h["scenario"] for h in headlines]
    max_y = max(
        float(h["pixel_class_mean_mse_mean"]) for h in headlines
    ) * 1.2

    def sx(i: int) -> float:
        return margin + i * (width - 2 * margin) / max(1, len(headlines) - 1)

    def sy(y: float) -> float:
        return height - margin - y / max(max_y, 1e-9) * (height - 2 * margin)

    pts_hidden = " ".join(
        f"{sx(i):.1f},{sy(float(h['hidden_prototype_mse_mean'])):.1f}"
        for i, h in enumerate(headlines)
    )
    pts_pixel = " ".join(
        f"{sx(i):.1f},{sy(float(h['pixel_class_mean_mse_mean'])):.1f}"
        for i, h in enumerate(headlines)
    )
    x_labels = "\n".join(
        f'<text x="{sx(i):.1f}" y="{height-28}" transform="rotate(20 {sx(i):.1f},{height-28})" '
        f'text-anchor="start" font-size="10">{s}</text>'
        for i, s in enumerate(scenarios)
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2:.1f}" y="22" text-anchor="middle" font-size="14" font-family="sans-serif">Round 10: decoder probe — hidden vs pixel class-mean MSE</text>
<line x1="{margin}" x2="{width-margin}" y1="{height-margin}" y2="{height-margin}" stroke="#333"/>
<line x1="{margin}" x2="{margin}" y1="{margin}" y2="{height-margin}" stroke="#333"/>
{x_labels}
<polyline fill="none" stroke="#1f77b4" stroke-width="2.5" points="{pts_hidden}"/>
<polyline fill="none" stroke="#d62728" stroke-width="2.5" points="{pts_pixel}"/>
<rect x="{width-200}" y="48" width="175" height="50" fill="white" stroke="#ddd"/>
<circle cx="{width-185}" cy="64" r="4" fill="#1f77b4"/><text x="{width-172}" y="68" font-size="11">hidden proto MSE</text>
<circle cx="{width-185}" cy="86" r="4" fill="#d62728"/><text x="{width-172}" y="90" font-size="11">pixel class-mean MSE</text>
</svg>"""
    path.write_text(svg)
    return path


def write_frozen_svg(headlines: list[dict[str, float | str]]) -> Path:
    path = ARTIFACTS / "round10_frozen_mlp.svg"
    width, height = 640, 360
    margin = 54
    max_y = max(
        max(float(h["tango_prototype_mse_mean"]), float(h["passive_prototype_mse_mean"]))
        for h in headlines
    ) * 1.2

    def sx(i: int) -> float:
        return margin + i * (width - 2 * margin) / max(1, len(headlines) - 1)

    def sy(y: float) -> float:
        return height - margin - y / max(max_y, 1e-9) * (height - 2 * margin)

    scenarios = [h["scenario"] for h in headlines]
    tango_pts = " ".join(
        f"{sx(i):.1f},{sy(float(h['tango_prototype_mse_mean'])):.1f}"
        for i, h in enumerate(headlines)
    )
    passive_pts = " ".join(
        f"{sx(i):.1f},{sy(float(h['passive_prototype_mse_mean'])):.1f}"
        for i, h in enumerate(headlines)
    )
    x_labels = "\n".join(
        f'<text x="{sx(i):.1f}" y="{height-26}" text-anchor="middle" font-size="10">{s}</text>'
        for i, s in enumerate(scenarios)
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2:.1f}" y="20" text-anchor="middle" font-size="14" font-family="sans-serif">Round 10: frozen MLP features — TANGO vs passive</text>
<line x1="{margin}" x2="{width-margin}" y1="{height-margin}" y2="{height-margin}" stroke="#333"/>
<line x1="{margin}" x2="{margin}" y1="{margin}" y2="{height-margin}" stroke="#333"/>
{x_labels}
<polyline fill="none" stroke="#1f77b4" stroke-width="2.5" points="{tango_pts}"/>
<polyline fill="none" stroke="#ff7f0e" stroke-width="2.5" points="{passive_pts}"/>
<circle cx="{width-170}" cy="58" r="4" fill="#1f77b4"/><text x="{width-158}" y="62" font-size="11">TANGO</text>
<circle cx="{width-170}" cy="78" r="4" fill="#ff7f0e"/><text x="{width-158}" y="82" font-size="11">passive</text>
</svg>"""
    path.write_text(svg)
    return path


def run() -> dict[str, object]:
    logger = setup_logging()
    started = time.time()
    logger.info("Round 10 TALON manuscript-closing benchmark started")

    dim_pixel = 28
    decoder_scenarios = {
        "decoder_balanced": SimConfig(),
        "decoder_minibatch": SimConfig(
            use_minibatch=True, minibatch_size=8, local_steps=6
        ),
    }
    frozen_scenarios = {
        "frozen_mlp_balanced": SimConfig(),
        "frozen_mlp_imbalanced": SimConfig(class_counts=(4, 8, 12)),
    }

    per_seed: list[dict[str, float | str]] = []
    for name, cfg in decoder_scenarios.items():
        per_seed.extend(run_decoder_scenario(name, cfg, dim_pixel, logger))
    for name, cfg in frozen_scenarios.items():
        per_seed.extend(run_frozen_mlp_scenario(name, cfg, logger))

    metric_keys_decoder = [
        "hidden_prototype_mse",
        "hidden_prototype_corr",
        "pixel_class_mean_mse",
        "pixel_class_mean_corr",
        "decoder_lstsq_pixel_mse_from_est",
        "decoder_lstsq_pixel_mse_from_true",
        "count_mae",
        "individual_mse_from_prototypes",
    ]
    metric_keys_frozen = [
        "tango_prototype_mse",
        "passive_prototype_mse",
        "count_mae",
        "tango_vs_passive_gain_x",
    ]

    decoder_headlines: list[dict[str, float | str]] = []
    frozen_headlines: list[dict[str, float | str]] = []
    for name in decoder_scenarios:
        rows = [r for r in per_seed if r["scenario"] == name]
        agg = aggregate(rows, metric_keys_decoder)
        agg["scenario"] = name
        agg["experiment"] = "decoder_probe"
        decoder_headlines.append(agg)
    for name in frozen_scenarios:
        rows = [r for r in per_seed if r["scenario"] == name]
        agg = aggregate(rows, metric_keys_frozen)
        agg["scenario"] = name
        agg["experiment"] = "frozen_mlp_backbone"
        frozen_headlines.append(agg)

    dec_plot = write_decoder_svg(decoder_headlines)
    frz_plot = write_frozen_svg(frozen_headlines)

    dec_bal = next(h for h in decoder_headlines if h["scenario"] == "decoder_balanced")
    dec_mb = next(h for h in decoder_headlines if h["scenario"] == "decoder_minibatch")
    frz_bal = next(h for h in frozen_headlines if h["scenario"] == "frozen_mlp_balanced")

    metrics: dict[str, object] = {
        "benchmark": "round10_talon_manuscript_closing",
        "method": "decoder semantic probe + frozen nonlinear backbone",
        "threat_model": (
            "active terminal probing via server-chosen classifier biases "
            "(not passive honest-but-curious)"
        ),
        "theorem_scope": {
            "exact": (
                "linear head, zero initial weights, full-batch local training, "
                "first-order terminal deltas, full-rank probe design; "
                "frozen_mlp uses fixed phi(x) during local training"
            ),
            "approximate_empirical": ["decoder_minibatch", "frozen_mlp_imbalanced"],
        },
        "decoder_setup": {
            "dim_pixel": dim_pixel,
            "map": "y = x @ D with fixed orthonormal D",
            "semantic_claim": (
                "Recovered hidden prototypes correlate with decodable pixel class means "
                "when Tier-1 succeeds; not raw-image inversion"
            ),
        },
        "per_seed": per_seed,
        "decoder_headlines": decoder_headlines,
        "frozen_mlp_headlines": frozen_headlines,
        "headline": {
            "decoder_balanced_hidden_mse": dec_bal["hidden_prototype_mse_mean"],
            "decoder_balanced_pixel_mse": dec_bal["pixel_class_mean_mse_mean"],
            "decoder_balanced_hidden_corr": dec_bal["hidden_prototype_corr_mean"],
            "decoder_balanced_pixel_corr": dec_bal["pixel_class_mean_corr_mean"],
            "decoder_balanced_lstsq_pixel_mse_from_est": dec_bal[
                "decoder_lstsq_pixel_mse_from_est_mean"
            ],
            "decoder_minibatch_pixel_mse": dec_mb["pixel_class_mean_mse_mean"],
            "decoder_minibatch_hidden_mse": dec_mb["hidden_prototype_mse_mean"],
            "frozen_mlp_balanced_tango_mse": frz_bal["tango_prototype_mse_mean"],
            "frozen_mlp_balanced_passive_mse": frz_bal["passive_prototype_mse_mean"],
            "frozen_mlp_balanced_gain_x": frz_bal["tango_vs_passive_gain_x_mean"],
        },
        "plots": [
            str(dec_plot.relative_to(RUN_ROOT)),
            str(frz_plot.relative_to(RUN_ROOT)),
        ],
        "proof_artifacts": [
            "paper/proofs.md",
            "paper/method.tex",
            "paper/draft.md",
        ],
        "runtime_seconds": time.time() - started,
    }
    out_path = ARTIFACTS / "round10_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    logger.info("wrote %s", out_path)
    logger.info("headline=%s", metrics["headline"])
    return metrics


if __name__ == "__main__":
    run()
