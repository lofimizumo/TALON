# Round 04 — Supervisor Review

## Verdict

**REVISE_MAJOR**

Round 04 delivers credible **infrastructure** (three snapshot paths in full script, JOLI-R3 L3, per-image logging, stall-aware budgets) and an **honest negative** on joint targets in the artifact reviewed. The delivered experiment is **not** the protocol in `researcher_proposal.md`: it ran under default **`QFL_QUICK=1`** (14×14, N=12, 2 seeds, 2 paths, adam=120), **skipped 28×28 entirely**, and **never executed GARD-SPARSE**. `config.json` acceptance (28×28 E2E preferred, MSE≤0.05, PSNR≥18 on ≥2/3 seeds) is **not met** and cannot be claimed at round 4 (`forbid_accept_before_round: 5`).

---

## Executive summary

The scientist’s Round 04 plan—parent R3 `tv_lbfgs`, GARD-SPARSE oracle MAP, T1p+graph polish, then conditional 14×14 fallback—is directionally sound after the Round 03 L3 stall. `code/benchmark_round04.py` implements that stack for **full** mode (`QFL_QUICK=0 QFL_FULL=1`). The **only** completed run in `artifacts/round04_metrics.json` is a **~5 s** quick iteration (`logs/experiment_round04.log` line 2: `resize=14 N=12 paths=('lasa_qterm_T1p_graph', 'shard_oracle')`).

On that quick track, snapshot recovery is **strong** (T1p+graph mean snap MSE **0.022**, well under weak-path gate 0.15) but **image L3 fails all seeds**: **0/2** joint pass (MSE≤0.05 and PSNR≥18); best single draw seed 7 — input MSE **0.109**, PSNR **9.61 dB** (~8 dB below PSNR gate). Aggregates in JSON **match** per-seed rows (verified). `summary_28x28_by_dim_g` is **empty**; `fallback_14x14.ran_because_28_failed` is **misleading** (quick mode skips 28×28 by design, not after a failed 28×28 sweep). Proposal and revision log **do not** label results as quick-mode or document the GARD/path omission. Round 03 supervisor demands on **per-seed pass rates** and **stall logging** are satisfied for this artifact; demands on **28×28 completion** and **≥2/3 seed pass** are not.

---

## Honesty / reproducibility audit

| Check | Result | Evidence |
|-------|--------|----------|
| Hardcoded / injected metrics? | **Pass** | Log lines 17, 31, 45, 59 match `per_seed` MSE/PSNR; means recomputed from JSON ✓ |
| Claim vs script config? | **Fail (Major)** | Proposal: `N=32`, `E=10`, seeds `3/7/11`, `PARTIAL_ROWS=8`, 3 paths (`researcher_proposal.md` L23–24, L11–12). Artifact: `n_samples=12`, `n_epochs=5`, seeds `[3,7]`, `partial_rows_t1p=6`, 2 paths (`round04_metrics.json` L33–42, L84–218) |
| 28×28 primary track run? | **Fail (Critical for acceptance)** | `summary_28x28_by_dim_g: {}`; quick branch sets empty `block_28` (`benchmark_round04.py` L802–810) |
| GARD-SPARSE evaluated? | **Fail (Major)** | Not in `SNAPSHOT_PATHS` under quick (`quick_config.py` L73–74); absent from `per_seed` |
| `honest_verdict` text accurate? | **Partial fail** | JSON L224 references “summary_28x28_by_dim_g” for “best weak-path” but that object is empty |
| Train/test leakage? | **N/A / Pass** | Simulated federated replay; oracle paths labeled conditional |
| Single-seed miracle? | **Pass** | Both seeds fail consistently; no cherry-picked pass |
| Stall / runtime honesty? | **Pass** | Per-image `joli_r3 image i/N` in log; `root_cause_l3_stall.md` aligned with `l3_budget` / quick caps |

**File:line anchors**

- Quick mode skip of 28×28: `code/benchmark_round04.py` L802–817 with `is_quick_mode()`.
- Default quick env: `code/quick_config.py` L37–40 (`QFL_QUICK` default `"1"`).
- Misleading fallback flag when quick: same file L818–819 sets `run_14 = True` without running 28×28.
- R3 L3 budget mismatch: parent `round03_metrics.json` uses `adam_steps=250`, `n_batch=128` at d=784; Round 04 quick uses `adam_steps=120`, `n_batch_cap=48` at d=196 (`round04_metrics.json` L27–31 vs `round03_metrics.json` L86–91).

---

## Feasibility & resources

| Item | Assessment |
|------|------------|
| Full 28×28 × 3 paths × 3 seeds | **Feasible but hours** on CPU; stall doc and R3 resume (~25–50 min/seed L3 at d=784 with adam=250) justify sequential seeds and no LAPIN at d=784 |
| Quick iteration | **Delivered** — Round 04 log completes in **~5 s**; suitable for regression only |
| GARD-SPARSE cost | Low vs L3; omission in quick is policy choice, not compute limit |
| Memory / parallelism | Quick uses 2 seeds sequentially; no repeat of R3 triple `ProcessPoolExecutor` contention on this run |

**Resource recommendation:** Treat `QFL_QUICK=1` Round 04 metrics as **dev smoke test**, not round evidence for `experiment_focused_strict` acceptance. Next gate: one **documented** `QFL_FULL=1` 28×28 block with `dim_g=160`, paths including `gard_sparse_oracle`, before further 14×14 tuning.

---

## Theory & assumptions

| Assumption | Challenge |
|------------|-----------|
| **H1** GARD-SPARSE lowers snap MSE vs SHARD partial | Untested in artifact; privacy-breakthrough port may help but **no MNIST numbers** here |
| **H2** Graph polish before L3 narrows snap→image gap | **Weak support** — snap MSE ~0.01–0.03 at 14×14 but input PSNR ~9 dB; same pattern as R02/R03 (good snapshots, weak L3) |
| **H3** 14×14 may hit ≥2/3 pass | **Falsified** on quick run (0/2); even if 2/2 passed, **not** substitutable for 28×28 per proposal L36 and `config.json` |
| Oracle assignment + 8 rows/epoch = full epoch leak | Acknowledged in proposal L34–35; quick uses 6 rows × 5 epochs — **different threat budget** than proposal |
| Parent R3 “operating point” | Only **`tv_lbfgs=0.005`** carried over; **adam/lbfgs/n_batch** differ from R3 resume profile — conflating “R3 L3” label with full R3 budget |

No theorems claimed; no proof gaps.

---

## Rubric scores (1–5)

| Criterion | Score | Note |
|-----------|------:|------|
| Novelty | 3 | Sensible composition of existing GARD/T1p/graph/JOLI pieces; not a new inversion principle |
| Soundness | 3 | Implementation coherent; **evaluation protocol ≠ proposal** lowers score |
| Feasibility | 4 | Quick path proves pipeline; full path still bounded by documented L3 cost |
| Impact | 2 | No movement toward acceptance targets vs R03 T1p@28×28 (MSE ~0.17–0.20, PSNR ~7 dB) on comparable L3 budget axis |
| Evaluability | 2 | Artifact not comparable to R02/R03 full MNIST tables without env label and 28×28 block |

---

## Issues

### Critical

1. **No 28×28 end-to-end metrics in Round 04 artifact** — `require_mnist_end_to_end` / primary track in `config.json` not satisfied; `summary_28x28_by_dim_g` empty (`round04_metrics.json` L51, L908 path in code).
2. **Acceptance targets unmet** — `targets_met_any_config: false`, `accepting_configs: []`; 0 seeds pass both MSE and PSNR on any reported path.

### Major

3. **Proposal ↔ artifact protocol mismatch** — Documented N=32/E=10/3 seeds/3 paths vs quick N=12/E=5/2 seeds/2 paths; not disclosed in `researcher_proposal.md` or `revision_log.md`.
4. **GARD-SPARSE (H1) not evaluated** in completed run despite being a headline Round 04 addition.
5. **Misleading JSON semantics** — `fallback_14x14.ran_because_28_failed: true` and `honest_verdict` “28×28 + 14×14 fallback” imply a failed 28×28 sweep; actual cause is quick-mode skip (`benchmark_round04.py` L802–819).
6. **Per-seed pass gate vs config** — Quick lowers `PASS_MIN_SEEDS` to 2 (`quick_config.py` L166–167) but project acceptance is **≥2 of 3** seeds; 0/2 pass anyway, but reporting “2/3” in proposal without env context is ambiguous.
7. **L3 budget not aligned with “parent R3 operating point”** — Only TV weight matches; adam 120 vs R3’s 250 at 28×28 makes cross-round PSNR comparison invalid.

### Minor

8. **No recon grids** — `grid_png_28` / `grid_png_14` null (quick grid seed/dim mismatch with empty 28 block).
9. **`image_matching_acc` always 0** — Likely metric definition at N=12; worth one-line note in revision log.
10. **Revision log** states “no supervisor review” — process slip only.

---

## Actionable suggestions

1. **Re-run Round 04 with `QFL_QUICK=0 QFL_FULL=1`** (or dedicated `round04_metrics_full.json`) for **28×28**, `dim_g=160`, all three `SNAPSHOT_PATHS`, seeds 3/7/11, and log env + wall-clock in `revision_log.md`.
2. **Include `gard_sparse_oracle`** in every acceptance-grade run; report snap vs input MSE separately for H1.
3. **Fix JSON labeling in quick mode** — e.g. `mnist_tracks.primary_28x28.skipped_reason: "QFL_QUICK"` and do not set `ran_because_28_failed` when 28×28 was not attempted.
4. **Align L3 naming** — Distinguish “R3 tv_lbfgs” from “R3 L3 budget (adam=250, n_batch_cap=128)” in proposal tables.
5. **Report per-seed joint pass table** (as R02/R03) in proposal or revision log when full metrics exist; keep quick results in a separate “smoke” subsection.
6. **Before Round 05 full 28×28**, complete quick L3 adam sweep (Round 05 plan) only as a **trend** gate; do not relax `input_mse_max` / `psnr_min_db` for 14×14 acceptance.
7. **Optional:** Cache L3 across snapshot paths sharing identical `s_rec` to cut full-mode cost (`root_cause_l3_stall.md` root cause 2).

**Policy:** `forbid_accept_before_round: 5` — **no ACCEPT verdict** for this run folder regardless; Round 04 must still deliver honest 28×28 evidence before round 5 closeout under `experiment_focused_strict`.

---

## Round 03 demands checklist

| Demand (R03 review) | Round 04 status |
|---------------------|-----------------|
| Complete R3 metrics with fixed profile | **Done** — `round03_metrics.json`, 0/3 pass at 28×28 |
| R4 benchmark + per-image logging | **Done** — `experiment_round04.log` |
| Per-seed pass rates in artifact | **Done** for quick (2 seeds); **missing** 28×28 |
| No full LAPIN grid on d=784 | **Done** (quick + full budgets) |
| ≥2/3 seeds pass 28×28 | **Not met** — 28×28 not run in R4 artifact |

---

## Metrics reference (artifact under review)

**Environment:** `QFL_QUICK=1` (default), 14×14, N=12, E=5, seeds 3 & 7, dim_g=100, adam=120, paths: T1p+graph, shard_oracle.

| Path | Snap MSE (mean) | Input MSE (mean) | PSNR (mean) | Joint pass |
|------|----------------:|-----------------:|------------:|:----------:|
| `lasa_qterm_T1p_graph__joli_r3` | 0.022 | 0.122 | 9.16 dB | **0/2** |
| `shard_oracle__joli_r3` | 0.033 | 0.129 | 8.94 dB | **0/2** |

**Per-seed (T1p+graph):** seed 3 — MSE 0.135, PSNR 8.70; seed 7 — MSE 0.109, PSNR 9.61 (best; still fails both gates).

**Comparison (28×28, R3 T1p, adam=250):** mean input MSE **0.190**, PSNR **7.21 dB** — R4 quick 14×14 improves PSNR ~2 dB at lower d but remains **~9 dB** below PSNR target.
