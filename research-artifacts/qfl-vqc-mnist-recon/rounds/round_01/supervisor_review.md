# Round 01 — Supervisor Review

## Verdict

**REVISE_MAJOR**

Round 01 delivers a **runnable, honest end-to-end stack** (real MNIST pixels → `SurrogateQFL` → snapshot recovery → L3 → image metrics) with reproducible JSON/logs. It does **not** satisfy `config.json` acceptance for this run: resolution is **8×8**, not the preferred **28×28**; the **SHARD oracle is not a defensible upper bound** on 2/3 seeds; headline target success rides on **T1b with 4× the oracle’s row budget** and is **not transferable** to a fair weak-tier claim. Policy also blocks **ACCEPT** before round 5 (`forbid_accept_before_round: 5`). Treat Round 01 as a **tractability scaffold**, not a metric pass.

---

## Executive summary

| Audit question | Supervisor ruling |
|----------------|-----------------|
| Is T1b success on 8×8 trivial? | **Partly yes** — not metric fraud, but **structurally easier** than B=4 SHARD (B=1 direct disaggregation) plus **320 vs 80** observed rows; with **d=64, dim_g=100** L3 is in an **overcomplete** regime (`dim_g > d`), so near-perfect snapshots (seeds 7, 11) collapse image error to ~0. **Not** evidence that weak-tier QFL recon “solved” MNIST. |
| Is SHARD oracle broken? | **Invalid as upper bound, not cheated** — `true_snapshots` is logging-only (`vendor/shard_sim/attacker.py:328–333`). Failure is **Stage-2 assignment / non-convergence** on seeds 7 and 11 (snapshot MSE 0.36 / 0.16 vs T1b ~10⁻⁷). Oracle mean input MSE **0.147** (PSNR **8.9 dB**) vs T1b **0.0026** (PSNR **48.3 dB**). |
| Are image metrics computed correctly? | **Pass (protocol)** — Hungarian assignment + mean squared error on flattened **[0,1]** pixels; PSNR with peak=1.0 (`code/benchmark_round01.py:195–212`, `vendor/shard_sim/metrics.py`). **Fail (reporting)** — headline uses **mean-only** pass; no per-seed joint pass table; JOLI ≡ SHARD L3 here because **TV is off** when `dim_g > d` (`code/joli_invert.py:151–155`). |
| Real VQC stack vs need 28×28? | **Stack real; resolution not** — correct SurrogateQFL + SHARD + LASA-QTERM + L3 wiring. **8×8 is an ablation**, not the config-preferred validation (`config.json` domain_notes; `mnist_choice.resolution: 8` in metrics). **28×28 required** before any acceptance discussion. |

**Headline correction:** JSON `honest_verdict` (“targets met (unexpected at 8x8 Round 1)”) is **arithmetically true for mean T1b** but **scientifically inadmissible** as run acceptance: wrong resolution, broken oracle ordering, unfair observation budget, round-index policy.

---

## Per-seed target pass audit (joint: input MSE ≤ 0.05 **and** PSNR ≥ 18 dB)

Primary inverter: `shard_l3` (JOLI identical on all seeds). Source: `artifacts/round01_metrics.json`.

| Path | Seed 3 | Seed 7 | Seed 11 | Pass rate |
|------|--------|--------|---------|-----------|
| `shard_oracle` | fail / fail | fail / fail | fail / fail | **0/3** |
| `lasa_qterm_T1p` | MSE ✓, PSNR ✗ | fail / fail | fail / fail | **0/3** |
| `lasa_qterm_T1b` | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | **3/3** |

Oracle misses even on the “easy” seed 3 (input MSE **0.0586** > 0.05; PSNR **12.3 dB** < 18). T1b **3/3** joint pass is driven by **near-zero snapshot error** on seeds 7 and 11 after B=1 recovery, not by L3 breakthrough on hard snapshots.

**Observation budget (fairness):**

| Path | Intermediate batch rows | Terminal rows | Total gradient rows |
|------|-------------------------|---------------|---------------------|
| `shard_oracle` | 80 | 0 | 80 |
| `lasa_qterm_T1p` | 0 | 70 | 70 |
| `lasa_qterm_T1b` | 0 | 320 | 320 |

T1b is **not comparable** to SHARD oracle at equal tier budget. Sibling `qfl-terminal-snapshot` Round 04 accepted **T1b@80** (80 rows); this run uses **full B=1** (`lasa_qterm_b1_full`) — supervisor treats Round 02 **T1p** as primary weak tier and any T1b claim must use **budget-matched** rows.

---

## SHARD oracle audit (required)

| Check | Result | Evidence |
|-------|--------|----------|
| `true_snapshots` smuggled into recovery | **Pass (no leak)** | Only used when `true_snapshots is not None` for **matching-accuracy logging** (`vendor/shard_sim/attacker.py:328–333`); assignment/update use batch means only |
| Oracle uses full intermediate rows | **Pass** | `observed_intermediate_batch_gradients: 80` = 10 epochs × 8 batches |
| Oracle beats weak paths (upper bound) | **Fail** | Mean input MSE: oracle **0.147** > T1p **0.090** > T1b **0.0026**; ranking inverted |
| Stage-2 stability | **Fail on 2/3 seeds** | Seed 3: snap MSE **0.014**; seeds 7/11: **0.357** / **0.161**; revision log notes `||ΔS||_F` ≫ ε after `SHARD_MAX_ITER=200` |
| L3 failure modes | **Major** | Logs show pervasive `L-BFGS did not converge` on oracle paths (`logs/experiment_round01.log`); confounds image MSE when snapshot assignment is wrong |

**Supervisor ruling:** The oracle path is **operationally broken as a baseline** for this shuffle/seed draw, not “dishonest code.” Round 02 must **repair SHARD L2** (initialization, `max_iter`, convergence diagnostics, optional B=1 oracle arm) before any claim that weak tiers beat—or approach—the ceiling.

---

## T1b on 8×8 — triviality analysis

1. **Problem size:** `d = 64` vs 28×28 `d = 784` — L3 search space shrinks ~12×; parent Round 03 Pareto on 28×28 is the relevant hardness reference (`researcher_proposal.md:20`).
2. **Encoding geometry:** `dim_g = 100 > d = 64` — JOLI TV disabled; inversion is **overdetermined in snapshot space**, easing perfect recovery when snapshots are exact.
3. **Algorithmic asymmetry:** T1b calls `b1_budget_disaggregate` (same family as SHARD `_level2_b1_direct`) on **per-client** gradients; SHARD oracle solves **B=4 assignment** alternating problem — strictly harder per sibling science.
4. **Empirical smoking gun:** Seeds 7 and 11: T1b snapshot MSE ~**10⁻⁷**, `image_matching_acc = 1.0`, input MSE ~**0**; oracle snapshot MSE **0.16–0.36**, matching acc **0.0**. Success is **snapshot identification**, not a surprising L3 win.

**Conclusion:** T1b “success” at 8×8 is **expected** under generous rows and B=1 structure; it does **not** validate H1 (oracle + L3 hits targets) and must not be cited as Round 01 acceptance.

---

## Image metrics & artifacts

| Check | Result |
|-------|--------|
| Metrics computed in code | **Pass** — no hardcoded headline JSON |
| Hungarian image MSE | **Pass** — `compute_matching_accuracy` + `compute_reconstruction_mse` |
| PSNR definition | **Pass** — `10 log10(1/mse)` for [0,1] peak |
| Snapshot vs image MSE separation | **Pass** — both reported; bottleneck visible (oracle seed 7: snap **0.36** → image **0.21**) |
| Recon grid | **Partial** — `artifacts/round01_recon_grid.png` exists (seed 7, 4 methods); caption admits **fixed index** (no Hungarian for display) — good honesty, insufficient for 28×28 qualitative review |
| Log ↔ JSON | **Pass** — spot-checked seed 3/7/11 lines vs `round01_metrics.json` |

**Minor implementation note:** `evaluate_reconstruction` calls `compute_matching_accuracy` twice for the same pair (`benchmark_round01.py:203–210`) — harmless duplication.

---

## Config acceptance gates

| Gate | Round 01 | Ruling |
|------|----------|--------|
| `require_mnist_end_to_end` | 8×8 MNIST, full L3 | **Partial** — E2E yes; **not preferred resolution** |
| `require_vqc_stack` | SurrogateQFL + SHARD + Qterm + L3 | **Pass** |
| `target_metrics` (MSE ≤ 0.05, PSNR ≥ 18) | Mean T1b only | **Fail for admissible path** — wrong res + unfair tier + invalid oracle |
| `oracle_shard_required_as_upper_bound` | T1b ≪ oracle MSE | **Fail** |
| `forbid_accept_on_snapshot_only_without_l3_images` | L3 run | **Pass** |
| `forbid_accept_before_round` 5 | Round 1 | **Fail (policy)** |
| `min_rounds_before_accept` 5 | Round 1 | **Fail (policy)** |

---

## Rubric (1–5)

| Criterion | Score | Note |
|-----------|------:|------|
| Novelty | 2 | First E2E image loop in this slug; science largely inherits terminal-snapshot + vendored SHARD |
| Soundness | 3 | Metrics coherent; oracle baseline and tier budgets undermine comparative claims |
| Feasibility | 4 | ~10 min CPU run, clear path to 28×28 |
| Impact | 2 | No defensible weak-tier win at fair budget; oracle failure blocks ceiling narrative |
| Evaluability | 4 | Good JSON/logs; missing per-seed pass matrix and SHARD convergence fields |

---

## Critical issues

1. **Wrong resolution for acceptance** — 8×8 tractability ablation does not meet “real VQC stack” validation intent in `config.json` / README (28×28 preferred).
2. **SHARD oracle not defensible** — Weaker-appearing T1b beats oracle on image metrics; Stage-2 fails on seeds 7 and 11.
3. **Headline targets misleading** — Mean-only T1b pass with **0/3** oracle joint pass; cannot ACCEPT on `honest_verdict`.
4. **Unfair weak-tier budget** — T1b@320 rows vs oracle@80; contradicts `oracle_shard_required_as_upper_bound` spirit.

## Major issues

5. **T1p neglected** — `config.json` `primary_weak_tier: T1p_or_T1b`; T1p is the honest partial-terminal probe (70 rows) and misses PSNR on all seeds; Round 02 must prioritize T1p, not celebrate full T1b.
6. **JOLI arm uninformative** — `dim_g > d` disables TV; shard and JOLI paths numerically identical — no polish ablation value at this setting.
7. **L3 under-convergence unreported in JSON** — logs contain mass L-BFGS warnings; metrics should export per-path failure counts.

## Minor issues

8. Only **3 seeds** — thin for shuffle-sensitive SHARD assignment.
9. Recon grid omits T1p and uses single seed — insufficient for supervisor visual audit at 28×28.

---

## Round 02 experiment demands (mandatory)

1. **28×28 pipeline** — `FederatedDataLoader(resize=None)` (or `resize=28`); document `d=784`, runtime budget, and L3 knobs (`n_batch`, `adam_steps`) explicitly; retain 8×8 as **diagnostic ablation** only, not headline.
2. **Fix SHARD L2** — Tune `max_iter`, initialization, convergence logging to JSON; verify oracle snapshot MSE **≤ T1p** and **≤ T1b@matched_budget** on **all** seeds; add `shard_l2_converged` and final `||ΔS||_F` per seed.
3. **T1p focus** — Primary weak path: `QtermTier.T1P` with `partial_rows` sweep; report snapshot MSE vs `snapshot_mse_max_for_weak_path` (0.15); goal is **visible** 28×28 recons, not oracle-beating MSE alone.
4. **Recon grids** — Per-seed (or worst/best/median) PNGs for oracle, T1p, T1b@80; include difference maps; label Hungarian vs fixed-order display.
5. **Per-seed pass rates** — JSON table: fraction of seeds meeting **joint** MSE+PSNR targets per `path__inverter`; forbid mean-only headline acceptance.
6. **Fair T1b budget** — Run **T1b@80** (or match oracle row count) alongside full T1b; never headline full 320-row T1b against 80-row oracle.
7. **dim_g sweep** — At 28×28, test `dim_g ∈ {100, 200, 400}` so JOLI TV arm (`dim_g < d`) is actually exercised.

---

## Supervisor ruling on researcher claims

| Claim | Ruling |
|-------|--------|
| H1: SHARD oracle + L3 hits targets on 8×8 | **Rejected** — oracle mean input MSE **0.147**, 0/3 joint pass |
| H2: T1b approaches oracle when snap MSE < 0.15 | **Misleading** — true only because oracle **failed**; T1b far exceeds oracle on seeds 7/11 |
| E2E stack established | **Accepted** — keep architecture; scale resolution and fix baselines |
| Targets met (JSON verdict) | **Rejected for acceptance** — see audits above |

---

## Honesty / reproducibility

| Check | Result |
|-------|--------|
| Metric injection | **Pass** |
| Oracle gradient smuggling | **Pass** |
| 8×8 choice disclosed | **Pass** — proposal, revision log, `mnist_choice` in JSON |
| Runnable artifacts | **Pass** — `code/benchmark_round01.py`, log, metrics, grid PNG |
| Independent supervisor re-run | **Not required** — audit by code + committed artifacts sufficient for Round 01 |

No evidence of fraud. The integrity risk is **over-interpreting** a T1b-dominated mean on a downsampled MNIST setup with a stalled SHARD oracle.

---

## References (evidence paths)

- Benchmark: `research-artifacts/qfl-vqc-mnist-recon/code/benchmark_round01.py`
- Metrics: `research-artifacts/qfl-vqc-mnist-recon/artifacts/round01_metrics.json`
- Log: `research-artifacts/qfl-vqc-mnist-recon/logs/experiment_round01.log`
- Config: `research-artifacts/qfl-vqc-mnist-recon/config.json`
- SHARD attacker: `vendor/shard_sim/attacker.py`
- Stack bridge: `research-artifacts/qfl-vqc-mnist-recon/literature/vqc_stack_bridge.md`
