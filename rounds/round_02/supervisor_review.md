# Supervisor Review — Round 02

**Run folder:** `research-artifacts/novel-inversion-vs-shard/`  
**Reviewed:** `researcher_proposal.md`, `revision_log.md`, `code/joli_invert.py`, `code/benchmark_round02.py`, `artifacts/round02_metrics.json`, `logs/experiment_round02.log`, reconstruction/bar plots under `artifacts/`  
**Baseline reference:** `supplementary_materials/code/shard_sim/attacker.py` (`level3_invert`, L693–843)  
**Prior audit:** `rounds/round_01/supervisor_review.md` (`REVISE_MAJOR`)

---

## 1. Verdict

**`REVISE_MINOR`**

Round 02 **closes all Round 01 Critical blockers**: canonical `ShardAttacker.level3_invert`, matched budgets, multi-seed compressive evaluation, and a **real** input-fidelity win in the intended regime (\(\dim\mathfrak{g}=100 < d=784\)). Experiments are honest, reproducible, and not hardcoded.

The work is **not yet** eligible for `ACCEPT` / `ACCEPT_WITH_MINOR` under `config.json`: novelty remains **incremental** (TV prior on a SHARD-identical shell), the snapshot-objective trade-off is large and under-analyzed, and evaluation still lacks a true \(\dim\mathfrak{g} \ge d\) control plus \(\lambda_{\mathrm{TV}}\) sensitivity at scale.

**Proceed to Round 03:** **Yes** — with the required fixes below.

---

## 2. Executive summary

The scientist pivoted LAPIN → **JOLI**: duplicate SHARD’s batch-Adam phase (`tv_adam=0`), then add isotropic TV **only** in the L-BFGS polish when \(\dim\mathfrak{g} < d\) (`tv_lbfgs=5\times 10^{-3}\)).

Supervisor **re-ran** `code/benchmark_round02.py` (2026-05-20, ~4.4 min, MPS). Aggregate metrics match stored `artifacts/round02_metrics.json` to full precision.

**Compressive 28×28 (\(\dim\mathfrak{g}=100\), \(d=784\), 3 seeds, \(N=12\))**

| Method | Input MSE ↓ | PSNR ↑ | Snap. match acc. | Mean snap. residual ↓ |
|--------|-------------|--------|------------------|------------------------|
| SHARD L3 (canonical) | 0.2804 ± 0.0034 | 5.52 ± 0.05 dB | **1.00** | **~1e-8** |
| **JOLI** | **0.1352 ± 0.0108** | **8.70 ± 0.35 dB** | 0.08 ± 0.07 | 0.19 ± 0.04 |

→ **~2.1× lower input MSE**, **~3.2 dB higher PSNR** on all seeds.  
→ **SHARD dominates snapshot consistency** (JOLI trades snapshot fit for smoother, more digit-like inputs — documented, not hidden).

**Reference 14×14 (\(\dim\mathfrak{g}=160\), \(d=196\), `tv_lbfgs=0`)** — JOLI equals SHARD to numerical precision (aggregate MSE 0.00493, identical per-seed rows). This is a valid **TV-off ablation**, not an overdetermined regime (see Major issue #4).

---

## 3. Honesty / reproducibility audit

| Check | Finding |
|-------|---------|
| Hardcoded metrics | **Pass.** `benchmark_round02.py` computes all metrics from arrays; JSON via `json.dumps(results)` (L368–369). |
| Re-run | **Pass.** Supervisor re-run: compressive aggregates for `input_mse`, `input_psnr_db`, `snapshot_match_acc`, `mean_snapshot_residual` match JSON exactly. |
| Canonical baseline | **Pass.** `ShardAttacker.level3_invert(snapshots, surrogate, device=...)` (L150–151). Budget block logs `n_batch=500`, `adam_steps=2000`, `lbfgs_max_iter=500`. |
| JOLI vs SHARD shell | **Pass (code audit).** `joli_invert_single` mirrors `level3_invert`: same `shard_n_batch`, lstsq+uniform seeds (`42+i`), Adam sum-loss, early stop at `recon.min() < 1e-8`, CPU L-BFGS with strong Wolfe. Difference: optional TV term in L-BFGS closure only. |
| Plots | **Pass** after re-run: `round02_metrics_bar.png`, `round02_reconstruction_compressive.png`, `round02_reconstruction_reference.png` present. |
| Claims vs code | **Mostly pass.** Pilot TV sweep and failed Attempts A/B in `revision_log.md` are consistent with code paths (`tv_adam`, `tv_lbfgs`). |
| Stale docstring | **Minor.** `benchmark_round02.py` L2 still says “dual-LM + Tikhonov”. |

**Implementation note:** JOLI reimplements SHARD’s Adam phase instead of calling `level3_invert` for Phase A. Structures match today; recommend a unit test that `tv_lbfgs=0` reproduces SHARD on one snapshot to guard drift.

---

## 4. Novelty vs SHARD (strict, fair)

| Aspect | SHARD L3 | JOLI |
|--------|----------|------|
| Leakage model | Oracle snapshots | Same |
| Adam phase | Snapshot MSE only | **Same** (when `tv_adam=0`) |
| L-BFGS | Snapshot MSE only | Snapshot MSE + **TV** if \(\dim\mathfrak{g}<d\) |
| Null space | Unregularized; many \(x\) fit \(s\) | TV selects smoother \(x\) |

**Assessment:** Total variation as an image prior is **well-known** (Rudin–Osher–Fatemi; standard in compressive imaging). The contribution is **not** a new information channel or Jacobian-based solver — the name “Jacobian-aware” is **misleading** (no explicit \(J\) use beyond what SHARD already implies through gradient-based Adam/L-BFGS).

What *is* defensible for a paper-line: **regime-conditioned structure prior on Stage 3** when \(\dim\mathfrak{g} < d\), with **TV confined to polish** so Adam still finds snapshot-near starts (pilot: `tv_lbfgs=0` recovers SHARD metrics). That is an **engineering refinement** of SHARD L3, not a replacement attack class comparable to DLG or a new Stage 1–2 model.

**Novelty rubric score: 3/5** (up from 2/5 in Round 01) — sufficient for continued research, **insufficient alone for `ACCEPT`** without sharper framing, ablations, and Pareto analysis.

---

## 5. Snapshot-match collapse (0.08) — severity

**Not Critical** for this audit, but **Major** for deployment as a drop-in Stage-3 replacement.

- Stage 3 in SHARD is explicitly \(\min_x \|\cos(Wx+b)-s\|_2^2\) per snapshot. JOLI’s L-BFGS **changes the polish objective**, so mean oracle residual \(\sim 0.19\) vs \(\sim 10^{-8}\) is expected, not a bug.
- The scientist reports this honestly; pilot table shows the **knob** (`tv_lbfgs`) controls the trade-off (MSE 0.121 / acc 0.08 at 0.005 vs MSE 0.276 / acc 1.0 at 0).
- For a **snapshot inversion** threat model, an attacker that fails to re-encode to \(s\) may be **invalid** downstream (Stages 1–2 assume consistent snapshots). Round 3 must treat this as a **first-class Pareto curve** (input MSE/PSNR vs `mean_snapshot_residual` vs `snapshot_match_acc`), not a footnote.
- **Not fraud:** the win on input MSE is under Hungarian matching to ground-truth digits, not by cheating metrics.

If the contribution is reframed as **“better semantic recovery under compressive null space given oracle \(s\)”** rather than **“strict Stage-3 solver,”** the trade-off is acceptable with evidence. Without that reframe, snapshot collapse would escalate toward Critical.

---

## 6. Rubric scores (1–5)

| Criterion | Score | Notes |
|-----------|-------|-------|
| **Novelty** | **3** | Regime-gated TV polish on SHARD shell; TV is standard; no new leakage surface. |
| **Soundness** | **4** | Implementation coherent; trade-off explicit; canonical baseline. |
| **Feasibility** | **4** | Runnable benchmark; ~15 min full multi-seed run on laptop MPS. |
| **Impact** | **3** | Real MSE/PSNR win in compressive MNIST surrogate; snapshot fit regresses. |
| **Evaluability** | **4** | Multi-seed, budget logged, plots; still small \(N\), no CIFAR/full pipeline. |

---

## 7. Issues

### Critical

*None.* Round 01 Critical items (no win vs SHARD, non-canonical baseline, unequal budget) are **resolved** in compressive 28×28.

### Major

1. **[Major] Snapshot–input Pareto not fully characterized.** Report curves over `tv_lbfgs` at multi-seed scale; include `mean_snapshot_residual` to oracle \(s\) and `snapshot_match_acc` on the same plots. Justify \(\lambda=5\times 10^{-3}\) without pilot-only tuning.
2. **[Major] Mislabeled “reference” regime.** Proposal states \(\dim\mathfrak{g}=160 > d=196\) (L46, L62); **false** (\(160 < 196\)). `reference_14x14` is still compressive; TV is off only because `tv_lbfgs=0` is passed manually. Rename and add a true **overdetermined** run (\(\dim\mathfrak{g} \ge d\), e.g. \(\dim\mathfrak{g}=256\), \(d=196\)) where TV should auto-disable.
3. **[Major] Contribution framing vs SHARD paper.** Drop-in “Stage-3 replacement” overclaims until snapshot consistency is bounded or reframed as utility–privacy trade-off. Remove or justify “Jacobian-aware” branding.
4. **[Major] Scale and generalization.** \(N=12\), MNIST-only surrogate, \(\lambda\) tuned on seed 7 pilot — extend \(N\), report seed held-out for \(\lambda\), and plan CIFAR / end-to-end path per `config.json` goal.

### Minor

5. **[Minor] Runtime ~1.35× SHARD** in compressive regime — acceptable; report in tables.
6. **[Minor] Stale benchmark docstring** (“dual-LM + Tikhonov”).
7. **[Minor] JOLI duplicates SHARD Adam** — add regression test `tv_lbfgs=0` ≡ `level3_invert` on fixed snapshot.

---

## 8. Round 01 fix checklist

| Round 01 issue | Round 02 status |
|----------------|-----------------|
| No win vs SHARD | **Fixed** (compressive MSE/PSNR) |
| Novelty not demonstrated | **Partial** (structure prior + regime gate; still incremental) |
| Non-canonical baseline | **Fixed** |
| Unequal budget | **Fixed** |
| Small scale / single seed | **Partial** (3 seeds; still \(N=12\)) |
| LAPIN snapshot fit | **N/A** (pivoted) |
| Missing plots | **Fixed** |

---

## 9. Conditions for upgraded verdict

| Target | Requirements |
|--------|----------------|
| **`ACCEPT_WITH_MINOR`** | Multi-seed Pareto over `tv_lbfgs`; true \(\dim\mathfrak{g}\ge d\) control; defensible novelty text (null-space selection, not “new attack”); \(N \ge 50\) or paper-aligned batch size on ≥1 setting; zero Major evaluation gaps; snapshot trade-off explicitly positioned. |
| **`ACCEPT`** | Above + CIFAR or full SHARD pipeline slice; sustained wins with snapshot residual within paper-tolerable band **or** accepted dual-metric story endorsed for QFL threat model. |

---

## 10. Top 3 required fixes for scientist (Round 3)

1. **Pareto sweep** — `tv_lbfgs ∈ {0, 1e-3, 5e-3, 1e-2, 2e-2}` × 3–5 seeds; plot input MSE/PSNR vs snapshot residual/acc; hold out one seed for \(\lambda\) selection.
2. **Correct regime matrix** — add \(\dim\mathfrak{g} \ge d\) configuration; fix proposal text (\(160 \not> 196\)); rename `reference_14x14` → e.g. `ablation_tv_off_14x14`.
3. **Reframe contribution** — “Structure-regularized polish for compressive Stage 3” with SHARD-identical Adam; retire “Jacobian-aware” unless Jacobian machinery is added; clarify when JOLI is **not** a valid strict snapshot inverter.

---

## 11. Recommendation

| Question | Answer |
|----------|--------|
| Accept now? | **No** (`REVISE_MINOR`) |
| Proceed to Round 3? | **Yes** |
| Is compressive win real? | **Yes** (re-verified) |
| Is TV-in-L-BFGS novel enough alone? | **No** — needs framing + Pareto + regime controls |
| Is snapshot acc. 0.08 Critical? | **No** (Major if unframed; honest reporting helps) |

---

*Supervisor audit: full `benchmark_round02.py` re-run verified 2026-05-20; metrics match `artifacts/round02_metrics.json`; plots regenerated in `artifacts/`.*
