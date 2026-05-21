# LASA-QTERM — QFL terminal individual snapshot recovery

**Separate research run** from parent TALON (`/workspace`). Parent findings are preserved; this folder targets the **original QFL goal**.

## Goal

Recover **individual** LASA-VQC snapshots \(s_i \in \mathbb{R}^{\dim\mathfrak{g}}\) in quantum federated learning **without within-epoch intermediate minibatch gradients**, using the same linear structure SHARD exploits: \(g^{(e,k)} = A^{(e)} \bar{s}^{(e,k)}\).

SHARD Stage 2 achieves this with **full intermediate** batch-gradient rows; we study weaker **terminal** observation tiers.

## Status

**ACCEPT_WITH_MINOR** (Round 4, 2026-05-20) — method package **LASA-QTERM**

| Round | Verdict |
|-------|---------|
| 1 | REVISE_MAJOR (T1 fails; mean floor) |
| 2 | REVISE_MAJOR (B=1 works but budget mismatch) |
| 3 | ACCEPT_WITH_MINOR (T1 impossibility + honest tiers) |
| 4 | **ACCEPT_WITH_MINOR** (consolidated package) |

## Main scientific result

| Tier | Individual snapshots? | Notes |
|------|----------------------|--------|
| **T1** (epoch-averaged terminal only) | **No** — proved impossible | `paper/impossibility_t1.md` |
| **T1p** (last \(p\) minibatch terminals / epoch, honest) | **Yes** (~0.45 MSE vs SHARD ~0.48) | 70 rows, no imputation |
| **T1b** (B=1 per-client terminal / round) | **Yes** (~0.16 MSE @ 80 rows) | Matched budget vs SHARD |
| **T2** | **Yes** (SHARD oracle) | Needs all intermediate rows |

## Reproduce

```bash
cd /workspace/research-artifacts/qfl-terminal-snapshot
python3 code/benchmark_round04.py
```

Production API: `code/qterm_attack.py`

## Layout

| Path | Role |
|------|------|
| `config.json` | Acceptance criteria and tier status |
| `paper/` | method, scope, impossibility proof |
| `code/` | benchmarks + `qterm_attack.py` |
| `rounds/` | Scientist/supervisor rounds |
| `literature/parent_talon_bridge.md` | Link to TALON aggregate-leakage work |
