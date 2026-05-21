# Round 05 — Supervisor Review

## Verdict

**REVISE_MAJOR**

Round 05 closes most Round 04 **process and honesty** gaps: three snapshot paths including GARD-SPARSE, explicit `run_env` / `mnist_tracks` labeling, adam {500,1000} sweep on quick 14×14, `StallWatchdog` with per-image L3 logs, and log↔JSON metric alignment (verified). The delivered artifact is still **quick-mode only** (`QFL_QUICK=1`, 14×14, 2 seeds); **28×28 end-to-end was not run** (`mnist_tracks.primary_28x28.ran: false`). **0/2** seeds meet joint acceptance (MSE≤0.05 **and** PSNR≥18 dB); best draw peaks at **14.09 dB** (~4 dB below gate). Under `config.json` (`require_mnist_end_to_end`, `forbid_accept_on_snapshot_only_without_l3_images`, targets `input_mse_max=0.05`, `psnr_min_db=18.0`), **ACCEPT is not available** at round 5 despite `forbid_accept_before_round: 5` lifting the round gate—metrics and track are insufficient.

---

## Executive summary

The scientist implemented Round 04 supervisor demands in code and ran an honest **~96 s** quick benchmark (`logs/experiment_round05.log`, `artifacts/round05_metrics.json`). Infrastructure is acceptance-ready for a full run: `mnist28_quality` profile, `full_28x28_runtime_estimate` (~11.1 h), stall monitor (0 warnings), and no misleading `ran_because_28_failed` flag. GARD-SPARSE oracle MAP improves snapshot MSE vs T1p at dim_g=160 (mean snap ~0.006 vs ~0.014) but the snap→image gap persists (best input PSNR **14.09 dB**, one seed with MSE≤0.05 only). Adam 500→1000 is **not monotonic** (verified regressions on seed 3 paths; proposal mislabels seed/path in H1 narrative). Scientist correctly defers full 28×28 and does not claim quick results satisfy `require_mnist_end_to_end`. **Next blocking work:** supervised launch of `QFL_QUICK=0 QFL_FULL=1` with seeds 3/7/11—not further 14×14 tuning alone.

---

## Honesty / reproducibility audit

| Check | Result | Evidence |
|-------|--------|----------|
| Hardcoded / injected metrics? | **Pass** | Summary lines in log match `per_seed` L3 rows (24/24 configs; automated diff 0 mismatches) |
| Log ↔ `per_seed_pass_best_adam`? | **Pass** | Rows derived from `_best_l3_row` (min input MSE per path); e.g. GARD seed 3 dim_g=160 adam=1000 → MSE 0.0390, PSNR 14.09 (`round05_metrics.json` L281–289; log L448) |
| Claim vs executed env? | **Pass** | `run_env.mode=quick`, `resize=14`, label “not acceptance-grade” (`round05_metrics.json` L4–12); proposal L24–25 states 14×14 does not satisfy E2E |
| 28×28 primary track run? | **Fail (acceptance)** | `primary_28x28.ran: false`, `skipped_reason: "QFL_QUICK"` (L15–22); estimate only in `full_28x28_runtime_estimate` (L34–54) |
| GARD-SPARSE evaluated? | **Pass** | In `SNAPSHOT_PATHS` and all `per_seed` blocks (`quick_config.py` L58; JSON `per_seed`) |
| Misleading fallback JSON (R04)? | **Pass** | Uses `mnist_tracks`; no `ran_because_28_failed` |
| Stall monitor used? | **Pass** | `StallWatchdog[round05]` at log L2; per-image progress L31+; **0** `STALL?` warnings |
| Adam sweep honest? | **Pass (data); Partial fail (narrative)** | Both adam values logged per path; regressions exist but **mis-cited** in proposal (see Issues) |
| Acceptance on 14×14 alone? | **Pass (not attempted)** | `accepting_configs: []`, `targets_met_any_config: false` (L938–939); no config relaxation |
| Train/test leakage? | **N/A / Pass** | Simulated FL replay; oracle paths labeled |
| Single-seed miracle? | **Pass** | Both seeds fail joint gate; one partial MSE pass only |

**File:line anchors**

- Quick patch: `code/benchmark_round05.py` L539–544 `patch_module_globals(..., research_round=5)`.
- Track labeling: L690–704 `mnist_tracks` with `skipped_reason: "QFL_QUICK"`.
- Best-row selection (MSE, not PSNR): L471–479 `_best_l3_row` → `min(..., key=input_mse)`; exposed as `per_seed_pass_best_adam` L670–687.
- Aggregate means: L482–501 `aggregate_method` uses same best-by-MSE rows.
- Stall wiring: L558–566, L267–268 `watchdog.on_image_done`; `code/run_monitor.py` L33–57.
- Round 5 quick profile: `code/quick_config.py` L46–60 adam_grid `(500, 1000)`, three paths.
- Grid title mismatch: `benchmark_round05.py` L655–657 title says “28×28” while quick `RESIZE=14` (L569–572).

**Verified regressions (adam 1000 vs 500 PSNR ↓)** — use these in H1 text, not proposal’s seed 7 / GARD seed 7 examples:

| seed | dim_g | path | Δ PSNR (dB) |
|------|------:|------|------------:|
| 3 | 100 | `gard_sparse_oracle` | −0.19 |
| 3 | 100 | `lasa_qterm_T1p` | −0.01 |
| 3 | 160 | `shard_oracle` | −0.02 |

**T1p seed 7:** adam 1000 **improves** PSNR at dim_g 100 (+0.72 dB) and 160 (+0.32 dB)—contradicts `revision_log.md` L54 and proposal L83 “GARD seed 7 regresses” (no GARD seed 7 PSNR regression in artifact).

---

## Feasibility & resources

| Item | Assessment |
|------|------------|
| Quick run delivered | **Yes** — wall **96.3 s** vs est JOLI **234 s** (`round05_metrics.json` L100–101; log L659) |
| Full 28×28 × 3 paths × 3 seeds | **Feasible, ~11 h CPU** per `full_28x28_runtime_estimate` (L51–52); stall monitor + per-image logs ready |
| GARD-SPARSE cost | Negligible vs L3; included in both quick and full configs |
| dim_g sweep on quick | **Done** {100,160}; H3 inconclusive for joint pass (both 0/2) |
| Memory / parallelism | Sequential seeds, `parallel_workers=1`; no R3 pool stall repeat |

**Resource recommendation:** Treat Round 05 quick metrics as **adam-sweep / regression smoke**, not acceptance evidence. Approve wall clock for one **`QFL_FULL=1`** block before more 14×14 grids or hypothesis expansion.

---

## Theory & assumptions

| Assumption | Assessment |
|------------|------------|
| **H1** Monotonic PSNR vs adam | **Rejected** on observed cells; narrative must fix seed/path citations |
| **H2** GARD lowers snap MSE, narrows image gap | **Partial** — snap MSE advantage clear at dim_g=160; image PSNR still ~4 dB below gate on best draw |
| **H3** dim_g=160 beats 100 on joint metrics | **Weak / inconclusive** — best PSNR at 160 (14.09 dB) but 0/2 pass at both; means at 160 slightly better for GARD/T1p |
| **H4** mnist28_quality reaches acceptance | **Untested** — correctly deferred; speculative ~10–12 dB in proposal is not evidenced |
| Oracle GARD assignment | Acknowledged; acceptable as upper-bound path only |
| 14×14 → 28×28 transfer for adam | **Unknown** — different d, N, n_batch cap; quick sweep cannot justify skipping full track |

No theorems claimed; no proof gaps.

---

## Rubric scores (1–5)

| Criterion | Score | Note |
|-----------|------:|------|
| Novelty | 3 | Compositional (GARD + JOLI quality + adam sweep); not a new inversion principle |
| Soundness | 4 | Pipeline and metrics traceable; H1 write-up errors lower confidence |
| Feasibility | 4 | Quick delivered; full path budgeted and instrumented |
| Impact | 3 | ~+5 dB vs R4 quick on best draw; still far from acceptance targets |
| Evaluability | 4 | Strong env labeling and JSON structure; 28×28 block still empty |

---

## Issues

### Critical (acceptance blockers)

1. **No 28×28 end-to-end metrics** — `require_mnist_end_to_end` / `forbid_accept_on_snapshot_only_without_l3_images` not satisfied (`config.json`; `mnist_tracks.primary_28x28.ran: false`).
2. **Acceptance targets unmet** — `targets_met_any_config: false`; **0/2** joint pass on quick (PSNR ceiling **14.09 dB** vs **18.0 dB** gate).

### Major

3. **Full 28×28 run still mandatory** — Round 05 improved labeling and estimates but did not execute `full_28x28_runtime_estimate.command` (scientist defers; supervisor concurs deferral, not deferral of requirement).
4. **H1 narrative misattributes regressions** — Proposal L83 (“GARD seed 7 regresses”) and revision log L54 (“T1p seed 7 … worse”) do not match artifact; undermines adam-sweep conclusions unless corrected.
5. **`per_seed_pass_best_adam` naming vs selection** — Field name implies PSNR-best adam; code picks **lowest input MSE** (`benchmark_round05.py` L471–479). Summaries and proposal tables follow MSE-best (consistent internally but easy to misread for H1).
6. **Pass gate semantics on quick** — `PASS_MIN_SEEDS=2` with only 2 seeds (`quick_config.py` L166–167); project acceptance is **≥2 of 3** at 28×28 (`config.json` L18–19). Quick 0/2 pass is honest but not substitutable.

### Minor

7. **Recon grid title** — PNG title says “28×28” under 14×14 quick run (`benchmark_round05.py` L655–657).
8. **`targets.pass_min_seeds_of_3: 2` in JSON** — Correct for full mode intent; clarify in revision log that quick used 2 seeds only.
9. **H3 not decisively tested** — dim_g sweep present but no joint-pass separation; optional report snap-vs-input gap per dim_g in one table.

---

## Actionable suggestions

1. **Run acceptance block** (blocking): `QFL_QUICK=0 QFL_FULL=1 PYTHONPATH=/workspace/vendor:/workspace/code python3 code/benchmark_round05.py` — seeds 3/7/11, dim_g=160 default, three paths; log wall clock and stall count in `revision_log.md`.
2. **Fix H1 documentation** — Replace erroneous seed 7 examples with verified regression table (seed 3, dim_g=100 GARD −0.19 dB, etc.); state H1 **rejected**, not “partial” without caveats.
3. **Rename or document best-adam policy** — e.g. `per_seed_pass_best_mse_adam` or footnote that aggregates optimize MSE, not PSNR.
4. **Fix grid title** — Use actual `RESIZE` in `plot_recon_grid` title for quick runs.
5. **Report full-mode per-seed joint table** in proposal when 28×28 exists (MSE and PSNR per seed/path, ≥2/3 pass column).
6. **If full run stalls** — Use existing `QFL_ABORT_ON_STALL=1` and `paper/root_cause_l3_stall.md`; do not lower `psnr_min_db` / `input_mse_max` for 14×14 acceptance.
7. **Optional cost cut** — Cache L3 across paths sharing identical `s_rec` only after full-mode baseline exists (R04 suggestion).

**Policy:** Round 5 is the first round **eligible** for accept discussion (`forbid_accept_before_round: 5`), but this artifact **does not** meet metric or E2E requirements. **Do not ACCEPT** on quick 14×14 or snapshot-only evidence.

---

## Round 04 demands checklist

| Demand (R04 review) | Round 05 status |
|---------------------|-----------------|
| Honest `mnist_tracks` / no false 28×28 failure flag | **Done** |
| GARD-SPARSE in acceptance paths | **Done** (quick + full script) |
| Per-seed pass / best-config reporting | **Done** (`per_seed_pass_best_adam`) |
| Stall + per-image L3 logs | **Done** |
| L3 naming (tv_lbfgs vs mnist28_quality vs quick sweep) | **Done** |
| Quick L3 adam sweep before full 28×28 | **Done** |
| **28×28 run with ≥2/3 seed joint pass** | **Not done** |
| Align proposal ↔ artifact env | **Done** for R05 run |

---

## Metrics reference (artifact under review)

**Environment:** `QFL_QUICK=1`, 14×14, N=12, E=5, seeds 3 & 7, dim_g∈{100,160}, adam∈{500,1000}, paths: T1p, GARD-SPARSE, SHARD oracle, profile `quick_r5_l3_sweep`.

| Path (dim_g=160, mean over seeds, best-by-MSE per seed) | Snap MSE (mean) | Input MSE (mean) | PSNR (mean) | Joint pass |
|--------------------------------------------------------|----------------:|-----------------:|------------:|:----------:|
| `gard_sparse_oracle__joli_quality` | 0.0059 | 0.0615 | **12.43 dB** | **0/2** |
| `lasa_qterm_T1p__joli_quality` | 0.0140 | 0.0659 | 11.88 dB | **0/2** |
| `shard_oracle__joli_quality` | 0.0311 | 0.1118 | 9.62 dB | **0/2** |

**Best single draw:** seed 3, dim_g=160, GARD, adam=1000 — input MSE **0.039**, PSNR **14.09 dB** (MSE ok, PSNR fail).

**Comparison:** R4 quick best PSNR ~9.6 dB → R5 quick best **+4.5 dB** on best draw; still **~4 dB** below PSNR target. R3 28×28 T1p mean ~7.2 dB — not comparable budget (adam 250, d=784).
