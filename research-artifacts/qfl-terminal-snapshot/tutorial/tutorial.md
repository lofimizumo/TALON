# Tutorial: LASA-QTERM — Terminal Snapshot Recovery in QFL

**Method:** **LASA-QTERM** (alias **Q-SNAP-T**)  
**Goal:** Recover individual LASA-VQC snapshots when the server does **not** see within-epoch minibatch gradients.

This tutorial is QFL-focused. Parent **TALON** work on terminal **class aggregates** is related but distinct—see [Parent TALON bridge](../literature/parent_talon_bridge.md).

---

## 1. Why this problem exists

**SHARD** (vendored in `/workspace/vendor/shard_sim`) recovers per-sample snapshots \(s_i\) in Stage 2 from **all** minibatch gradients each local epoch:

\[
g^{(e,k)} = A^{(e)} \bar{s}^{(e,k)}, \quad k = 1,\ldots,K.
\]

Many QFL deployments want to avoid publishing that full trajectory (bandwidth, privacy, API shape). They may publish only:

- **Epoch terminal** — one vector per round (mean over local batches), or
- **Partial terminal** — only the last few batch rows per epoch, or
- **Per-client terminal** — one gradient per participant per round (B=1).

**LASA-QTERM** is the packaged attack stack for those weaker channels, with honest tier labels and SHARD as oracle.

---

## 2. LASA linear structure (SurrogateQFL)

The simulator uses `SurrogateQFL`: each gradient is linear in a **batch-mean snapshot** (or individual snapshot for B=1):

| Observation | Formula |
|-------------|---------|
| Minibatch row | \(g^{(e,k)} = A^{(e)} \bar{s}^{(e,k)}\) |
| Epoch terminal | \(c^{(e)} = \frac{1}{K}\sum_k g^{(e,k)} = A^{(e)} \bar{s}\) |
| B=1 client row | \(g^{(e,i)} = A^{(e)} s_i\) |

Stacking epoch terminals **only** constrains the dataset mean \(\bar{s}\), not individual deviations \(s_i - \bar{s}\). That is **Proposition T1** (`paper/impossibility_t1.md`).

---

## 3. Observation tiers (read `paper/scope.md`)

| Tier | What the server sees | Within-epoch steps? | Individual recovery? |
|------|----------------------|---------------------|----------------------|
| **T1** | One epoch mean per round | No | **No** (impossible) |
| **T1p** | Last \(p\) batch rows / epoch (honest) | No (early rows not sent) | Partial |
| **T1b** | B=1 per-client rows | No | Yes (stronger channel) |
| **T2** | Full SHARD trajectory | Yes (intermediates) | Yes (oracle) |

Do not compare T1 (10 rows) to SHARD (80 rows) without stating the budget mismatch.

---

## 4. Production API

```python
from qterm_attack import QtermAttack, QtermConfig, QtermTier

# After Level-1 mean recovery (e_bar) and collecting observations:
attack = QtermAttack(QtermConfig(tier=QtermTier.T1P, partial_rows_per_epoch=7))
result = attack.recover(
    e_bar,
    coeff_matrices,
    batch_gradients=batch_gradients,  # full sim trajectories; attack keeps last p rows
)
S_hat = result.snapshots  # (N, dim_g)
```

| `QtermTier` | Required inputs | Production estimator |
|-------------|-----------------|----------------------|
| `T1` | `terminal_gradients` | Graph terminal MAP |
| `T1P` | `batch_gradients` | Honest partial disaggregate |
| `T1B` | `b1_gradients` | B=1 budget disaggregate |

Module: `code/qterm_attack.py`.

---

## 5. Reproduce the acceptance benchmark

From the run root:

```bash
cd research-artifacts/qfl-terminal-snapshot
python code/benchmark_round04.py
```

Outputs:

- `artifacts/round04_metrics.json` — smooth + **MNIST** tracks, `acceptance_table`
- `logs/experiment_round04.log`

Smooth-only (faster):

```bash
python code/benchmark_round04.py --smooth-only
```

---

## 6. How to read results

**Primary metric:** Hungarian snapshot MSE (permutation-aligned).

**Acceptance (this artifact):**

1. **T1:** Accept **impossibility** + report ~2× worse than SHARD at 10 vs 80 rows—not a failed SHARD clone.
2. **T1p:** Honest partial can approach SHARD on smooth at 70 rows; label as non-primary tier.
3. **T1b @ 80 rows:** Can beat ≤2× SHARD at **equal** row budget; still not the primary “epoch terminal only” deployment.

Example smooth headlines (5 seeds, Round 04 — see JSON for exact numbers):

| Tier | Method | MSE (approx) | vs SHARD |
|------|--------|--------------|----------|
| T2 | SHARD oracle | 0.48 | 1.0× |
| T1 | LASA-QTERM T1 | 0.99 | ~2.0× |
| T1p | LASA-QTERM T1p (p=7) | 0.44 | ~0.9× |
| T1b@80 | LASA-QTERM B=1 budget | 0.16 | 0.24× vs SHARD@80 |

MNIST: T1 stays at passive scale (~1.0); T1p and T1b patterns qualitatively similar.

---

## 7. Relation to parent TALON

| Parent (TALON) | This run (LASA-QTERM) |
|----------------|------------------------|
| Terminal probes → **class prototypes** | Terminals / partials → **per-sample snapshots** |
| Theorem: aggregates not individuals | Proposition T1: epoch means not individuals |
| GARD needs oracle assignment | T1p uses honest partial + soft assignment |
| PHASE2_SCOPED_ACCEPT | Scoped to LASA-QTERM tiers + SHARD oracle |

Bridge: `literature/parent_talon_bridge.md`.  
Do not overwrite parent artifacts under `/workspace`.

---

## 8. File map

| Path | Role |
|------|------|
| `paper/method.md` | Method definition |
| `paper/scope.md` | Tier table + acceptance scope |
| `paper/impossibility_t1.md` | T1 proof |
| `code/qterm_attack.py` | Production attack |
| `code/benchmark_round04.py` | Consolidated benchmark |
| `code/terminal_attacks.py` | Estimator primitives |
| `config.json` | Run goal and acceptance gates |

---

## 9. PDF export (optional)

```bash
pandoc tutorial/tutorial.md -o tutorial/tutorial.pdf
```

`config.json` does not require PDF for acceptance.

---

## 10. Scientific takeaway

**Strict epoch-terminal QFL (T1) cannot recover individual LASA snapshots** under linearity; graph priors spread rows but do not add new measurements. **LASA-QTERM** packages honest stronger tiers (partial, B=1) and SHARD oracle comparison so deployers can choose observation strength vs reconstruction fidelity without conflating row budgets.
