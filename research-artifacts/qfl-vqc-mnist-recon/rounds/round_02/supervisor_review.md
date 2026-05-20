# Round 02 — Supervisor Review

## Verdict

**REVISE_MAJOR**

Round 02 correctly moves the primary track to **28×28** (`d=784`), implements the requested **dim_g** sweep, exports **per-seed joint pass rates**, and reports an honest failure verdict in JSON. **Zero of three seeds** meet both acceptance targets (input MSE ≤ 0.05 **and** PSNR ≥ 18 dB) on **any** path × inverter × `dim_g` combination — **0/3 joint pass** on the headline resolution. The best mean compressive path remains **T1b@320 rows + JOLI @ dim_g=256** (mean input MSE **0.081**, PSNR **10.9 dB**), which is **~7 dB short** of the PSNR gate and still uses an **unfair observation budget** relative to oracle/T1p. Policy blocks **ACCEPT** before round 5 (`forbid_accept_before_round: 5`). Round 02 is a **valid scale-up experiment**, not a metric pass.

---

## Executive summary

| Audit question | Supervisor ruling |
|----------------|-----------------|
| 28×28 primary track delivered? | **Pass** — `mnist_choice.resolution: 28`, `d=784`, full E2E in `code/benchmark_round02.py` |
| Joint target pass on any seed? | **Fail** — **0/3** on all 18 `path__inverter` rows × 3 `dim_g` values in `table_28x28` |
| SHARD oracle defensible upper bound? | **Fail** — T1p **beats** oracle on mean input MSE with JOLI at every swept `dim_g`; oracle snapshot MSE **0.12–0.39** on seeds 7/11 |
| T1p as primary weak tier? | **Partial** — T1p tuned and reported, but headline “best” is still **T1b**; T1p never within 2× of targets |
| JOLI compressive sweep informative? | **Partial** — TV active when `dim_g < d`; JOLI helps vs `shard_l3`, but budget fixed; **LAPIN not run** |
| Round 01 supervisor demands met? | **Partial** — see §Round 01 demand checklist below |

**Headline:** Scientist `honest_verdict` is **accepted** — targets not met. Do **not** promote mean T1b@256 as weak-tier success; bottleneck is **L3 under-budget + oracle L2 failure**, not missing 28×28 infrastructure.

---

## Per-seed joint pass audit (MSE ≤ 0.05 **and** PSNR ≥ 18 dB)

Primary inverter for compressive regime: **`joli_l3`**. Source: `artifacts/round02_metrics.json` (`per_seed`).

### Best headline path: `lasa_qterm_T1b` @ `dim_g=256`

| Seed | Snapshot MSE | Input MSE | PSNR (dB) | Joint pass |
|------|-------------:|----------:|----------:|:----------:|
| 3 | ~0 | 0.085 | 10.69 | fail / fail |
| 7 | ~0 | 0.089 | 10.53 | fail / fail |
| 11 | ~0 | **0.069** | **11.58** | fail / fail |
| **Pass rate** | — | — | — | **0/3** |

Closest single draw: seed **11** (MSE gap **0.019**, PSNR gap **~6.4 dB**). No seed clears PSNR.

### Primary weak tier: `lasa_qterm_T1p` @ `dim_g=160` (supervisor reference)

| Seed | Snapshot MSE | Input MSE | PSNR (dB) | Joint pass |
|------|-------------:|----------:|----------:|:----------:|
| 3 | 0.012 | 0.114 | 9.44 | fail / fail |
| 7 | 0.049 | 0.135 | 8.68 | fail / fail |
| 11 | 0.071 | 0.172 | 7.64 | fail / fail |
| **Pass rate** | — | — | — | **0/3** |

T1p snapshot MSE is **within** the weak-path gate (≤ 0.15) on all seeds at `dim_g=160`, but L3 image metrics remain far from targets — **snapshot quality is not the sole bottleneck**.

### SHARD oracle @ `dim_g=100` (ceiling audit)

| Seed | Snapshot MSE | Input MSE (JOLI) | PSNR (dB) | Joint pass |
|------|-------------:|-----------------:|----------:|:----------:|
| 3 | 0.030 | 0.106 | 9.74 | fail / fail |
| 7 | **0.391** | 0.263 | 5.81 | fail / fail |
| 11 | **0.325** | 0.256 | 5.92 | fail / fail |
| **Pass rate** | — | — | — | **0/3** |

Logs: `SHARD did not converge within 500 iterations` on oracle L2 for seeds **7** and **11** (`logs/experiment_round02.log`); pervasive **L-BFGS did not converge** warnings on L3 polish.

---

## `table_28x28` pass-rate summary (mean over seeds)

All rows: **`pass_rate_both = 0.0`** (0/3). Best mean compressive JOLI rows:

| dim_g | Path | Input MSE (mean) | PSNR (mean) | Notes |
|------:|------|-----------------:|------------:|-------|
| 256 | `lasa_qterm_T1b` | **0.081** | **10.94** | Best mean; 320 terminal rows |
| 160 | `lasa_qterm_T1p` | 0.140 | 8.59 | Primary weak tier; 80 rows/epoch |
| 100 | `lasa_qterm_T1p` | 0.148 | 8.43 | Strongest compressive TV regime |
| 256 | `shard_oracle` | 0.222 | 6.75 | Oracle worsens vs 100/160 at some seeds |

**Supervisor ruling:** Increasing `dim_g` to 256 **helps T1b L3** but **hurts oracle** snapshot recovery on seed 3 (snapshot MSE **0.218** vs **0.030** at `dim_g=100`) — do not treat “larger dim_g” as uniformly better; Round 03 should **anchor compressive work at `dim_g=100`**.

---

## SHARD oracle audit (required)

| Check | Result | Evidence |
|-------|--------|----------|
| `true_snapshots` smuggled into recovery | **Pass (no leak)** | Same contract as Round 01 (`vendor/shard_sim/attacker.py`) |
| Graph L2 init + 500 iter | **Delivered** | `round02_tuning.shard_l2_init: graph_term_map`, `shard_max_iter: 500` |
| Oracle beats T1p/T1b on image MSE | **Fail** | At `dim_g=160` JOLI means: oracle **0.201** vs T1p **0.140** vs T1b **0.108** |
| Oracle snapshot MSE ≤ weak paths | **Fail on 2/3 seeds** | Seeds 7/11 oracle snap MSE **≫ 0.15** while T1p snap MSE **< 0.07** |
| L2 convergence exported to JSON | **Fail** | Warnings only in log; no `shard_l2_converged` / `||ΔS||_F` fields |
| `oracle_shard_required_as_upper_bound` | **Fail** | Config spirit violated on mean image metrics |

**Supervisor ruling:** Round 02 **improved** L2 vs Round 01 (graph init, more iterations) but the oracle remains **operationally broken** on shuffle-sensitive seeds — not evidence of cheating. **Do not** interpret “T1p beats oracle” as weak-tier superiority; it signals **baseline invalidity** for comparative claims.

---

## T1b vs T1p — fairness and triviality

| Path | Terminal / intermediate rows | Snapshot MSE (typical) | Role in Round 02 |
|------|------------------------------|------------------------|------------------|
| `shard_oracle` | 80 intermediate, 0 terminal | 0.03–0.39 | Broken ceiling on 2/3 seeds |
| `lasa_qterm_T1p` | 0 intermediate, **80** terminal (8/epoch) | 0.02–0.07 | **Primary weak tier** — still 0/3 pass |
| `lasa_qterm_T1b` | 0 intermediate, **320** terminal (B=1) | ~0 | Best L3 means; **4×** oracle row budget |

T1b near-zero snapshot MSE with **high** image MSE (e.g. seed 7: snap **~0**, input MSE **0.117** @ `dim_g=100` JOLI) shows the **L3 stage** dominates error even when snapshots are exact — supports Round 03 **L3 budget sweep**, not more T1b row flood.

---

## L3 / inverter audit

| Check | Result |
|-------|--------|
| JOLI vs SHARD L3 | **JOLI wins** on almost all rows when `dim_g < d` (TV in L-BFGS at `tv_lbfgs=0.005`) |
| L3 budget documented | **Partial** — fixed `adam_steps=2000`, `n_batch=shard_n_batch(784, dim_g)=500` in logs |
| L3 budget sweep | **Not done** — Round 01 demand unfulfilled |
| LAPIN arm | **Not run** — parent `code/lapin_invert.py` exists; omitted from `benchmark_round02.py` |
| Joint multi-snapshot polish | **Not run** — per-snapshot independence only (`literature/qfl_snapshot_inversion.md` open gap) |
| L-BFGS failure telemetry in JSON | **Fail** — thousands of log warnings, zero aggregate counts |
| `image_matching_acc` | **All 0.0** — likely threshold/permutation reporting issue; investigate but **not** blocking REVISE |

At `dim_g=100`, logs confirm compressive JOLI: `tv_lbfgs=0.0050`, `n_batch=500`, `adam_steps=2000`.

---

## Round 01 supervisor demand checklist

| # | Round 01 demand | Round 02 status |
|---|-----------------|-----------------|
| 1 | 28×28 pipeline | **Done** |
| 2 | Fix SHARD L2 + convergence JSON | **Partial** — graph init + 500 iter; **no JSON convergence fields** |
| 3 | T1p focus | **Partial** — tuned; headline still T1b-best |
| 4 | Recon grids (multi-path, multi-seed) | **Partial** — one PNG: `round02_recon_grid_dim160.png`, seed 7 only |
| 5 | Per-seed pass rates | **Done** — `table_28x28`, `per_seed[].targets` |
| 6 | Fair T1b@80 vs oracle@80 | **Not done** — full T1b@320 still in sweep |
| 7 | dim_g sweep for JOLI TV | **Done** — {100, 160, 256} |

---

## Config acceptance gates

| Gate | Round 02 | Ruling |
|------|----------|--------|
| `require_mnist_end_to_end` @ 28×28 | Full pipeline | **Pass** |
| `require_vqc_stack` | SurrogateQFL + SHARD + Qterm + L3 | **Pass** |
| `target_metrics` joint | 0/3 all configurations | **Fail** |
| `oracle_shard_required_as_upper_bound` | T1p < oracle image MSE (JOLI means) | **Fail** |
| `primary_weak_tier` T1p story | T1p 0/3 pass; best path T1b | **Fail for acceptance narrative** |
| `forbid_accept_before_round` 5 | Round 2 | **Fail (policy)** |
| `forbid_accept_on_snapshot_only` | L3 always run | **Pass** |

---

## Rubric (1–5)

| Criterion | Score | Note |
|-----------|------:|------|
| Novelty | 2 | Scale-up + dim_g sweep; science inherits terminal-snapshot + vendored optimizers |
| Soundness | 3 | Honest metrics and pass rates; oracle baseline and T1b budget undermine comparative claims |
| Feasibility | 3 | ~79 min CPU for full sweep; tractable with focused Round 03 budget |
| Impact | 2 | No defensible 28×28 target pass; identifies L3 budget as lever |
| Evaluability | 4 | Strong JSON/`table_28x28`; weak visual audit (single grid) |

---

## Critical issues

1. **0/3 joint pass at 28×28** — no path qualifies for acceptance discussion.
2. **Oracle not an upper bound** — violates `oracle_shard_required_as_upper_bound`; seeds 7/11 L2 non-convergence persists.
3. **Headline best path is T1b@320** — not budget-matched and not the configured primary weak tier (T1p).

## Major issues

4. **L3 budget not swept** — fixed Adam/L-BFGS budget despite snapshot→image gap on near-perfect T1b snaps.
5. **LAPIN omitted** — Jacobian GN inverter never compared at compressive `dim_g=100`.
6. **Visual audit insufficient** — one grid (seed 7, `dim_g=160`); no per-digit panels, no T1p-only diagnostic grids for seeds 7/11.

## Minor issues

7. **No T1b@80** fairness arm — Round 01 demand still open.
8. **README status table empty** — update after Round 03.
9. **`image_matching_acc` all zero** — report fix or document threshold in Round 03 JSON.

---

## Round 03 experiment demands (mandatory)

Round 03 is **L3-forward**: hold snapshot simulator fixed unless oracle L2 JSON proves convergence failure. Do **not** expand to new seeds or `dim_g` sweep until the budget/inverter arms below are run at **`dim_g=100`** (compressive TV on, `d=784`).

### 1. L3 budget sweep (`adam_steps`, `n_batch`)

- **Script:** `code/benchmark_round03.py` (new).
- **Fixed snapshot paths for sweep:** reuse Round 02 checkpoints **or** re-run recovery once per seed at `dim_g=100` only (do not triple `dim_g` cost again).
- **Sweep grid (minimum):**
  - `adam_steps ∈ {2000, 4000}` (supervisor priority: **4000** arm required)
  - `n_batch ∈ {500, 1000}` via `shard_n_batch` override and explicit cap logging
- **Inverters in sweep:** `joli_l3` (primary), `shard_l3` (diagnostic control).
- **JSON:** Pareto table `input_mse_mean` vs `l3_runtime_sec_mean` per budget cell; flag `lbfgs_nonconv_count` per run from log capture.

### 2. LAPIN + JOLI TV at `dim_g=100`

- Add **`lapin_l3`** arm using parent `code/lapin_invert.py` on the **same recovered snapshots** as JOLI/SHARD.
- **JOLI at `dim_g=100` only** for this arm (do not repeat 160/256 unless budget sweep wins at 100).
- Report side-by-side: `shard_l3`, `joli_l3`, `lapin_l3` for **`lasa_qterm_T1p`** and **`shard_oracle`** (T1b diagnostic only, not headline).
- Export `tv_lbfgs`, `n_batch`, `adam_steps` in metrics for reproducibility.

### 3. Joint polish

- Implement **multi-snapshot joint L-BFGS polish** after per-snapshot Adam (minimum viable: shared nothing, but **joint loss** \(\sum_i \|\cos(Wx_i+b)-s_i\|^2\) with optional graph regularizer from `graph_term_map` on snapshot residuals — see `literature/qfl_snapshot_inversion.md` §Open gaps).
- Label inverter `joli_joint_l3` or extend `joli_invert` with `joint_polish=True`.
- Compare **joint vs independent** polish on **T1p @ dim_g=100** for seeds 3/7/11 — JSON field `joint_polish_gain_db` (PSNR delta).

### 4. Per-digit reconstruction grids

- Save **`artifacts/round03_recon_grid_digit{0-9}_seed{s}_dim100.png`** (or one combined figure per seed with rows = digits 0–9, cols = oracle / T1p / truth).
- Use **Hungarian-aligned** reconstructions for display (match `evaluate_reconstruction` assignment).
- Required seeds: **3, 7, 11** (hard oracle failures must be visible).
- Include difference maps `|x_true - x_rec|` row under reconstructions for supervisor audit.

### 5. Oracle L2 closure (if time permits)

- Export `shard_l2_converged`, `shard_l2_delta_fro`, `shard_l2_iters` per seed to JSON.
- Optional **B=1 oracle diagnostic** arm (separate path key) — do not replace B=4 oracle in tables.

### 6. Reporting rules (unchanged)

- **Joint pass only** — no mean-only ACCEPT headline.
- **T1b@80** fairness row alongside T1p — full T1b@320 not in Round 03 headline table.
- `honest_verdict` must state pass rate explicitly: `joint_pass: k/3 seeds`.

---

## Supervisor ruling on researcher claims

| Claim | Ruling |
|-------|--------|
| H1: Graph SHARD L2 improves oracle at 28×28 | **Partial** — helps seed 3; **fails** seeds 7/11 (snap MSE still **> 0.12**) |
| H2: T1p + 8 rows/epoch narrows gap without T1b flood | **Partial** — good snap MSE; **image targets still 0/3** |
| H3: `dim_g=160` sweet spot | **Rejected** — best **mean** image metrics at **`dim_g=256`** for T1b JOLI; 160 is middle, not optimal |
| 28×28 E2E established | **Accepted** — keep stack |
| Targets met | **Rejected** — consistent with scientist `honest_verdict` |

---

## Honesty / reproducibility

| Check | Result |
|-------|--------|
| Metric injection | **Pass** |
| Oracle gradient smuggling | **Pass** |
| 28×28 + dim_g sweep disclosed | **Pass** |
| Runnable artifacts | **Pass** — `code/benchmark_round02.py`, `artifacts/round02_metrics.json`, `logs/experiment_round02.log` |
| Independent supervisor re-run | **Not required** — audit by committed artifacts + log spot-check |

No fraud detected. Risk is **over-interpreting** T1b@320 mean improvements while **0/3** joint pass and an **invalid oracle ceiling** remain.

---

## References (evidence paths)

- Benchmark: `research-artifacts/qfl-vqc-mnist-recon/code/benchmark_round02.py`
- Metrics: `research-artifacts/qfl-vqc-mnist-recon/artifacts/round02_metrics.json`
- Log: `research-artifacts/qfl-vqc-mnist-recon/logs/experiment_round02.log`
- Grid: `research-artifacts/qfl-vqc-mnist-recon/artifacts/round02_recon_grid_dim160.png`
- Config: `research-artifacts/qfl-vqc-mnist-recon/config.json`
- JOLI / LAPIN: `code/joli_invert.py`, `code/lapin_invert.py`
- Stack bridge: `research-artifacts/qfl-vqc-mnist-recon/literature/vqc_stack_bridge.md`
- Round 01 review: `research-artifacts/qfl-vqc-mnist-recon/rounds/round_01/supervisor_review.md`
