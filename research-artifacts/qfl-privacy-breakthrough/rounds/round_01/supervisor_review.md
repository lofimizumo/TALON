# Round 01 — Supervisor Review

## Verdict

**REVISE_MAJOR**

Round 01 is a credible three-lane survey with runnable artifacts, but it does **not** satisfy `config.json` acceptance for this run. Policy blocks **ACCEPT** before round 8; criterion **B** and **C** fail; criterion **A** is flagged only under oracle-heavy GARD-SPARSE with a laxer threat-reduction statistic than parent TALON Round 05. Treat GARD-SPARSE as **replication of a known conditional prior**, not a new breakthrough without assignment/graph/mean-oracle removal.

---

## Executive summary

The scientist delivered an honest multi-lane benchmark (`code/benchmark_round01.py`), reproducible logs, and explicit `honesty_notes` in JSON. Negative lanes (QFL-SHIELD, LEAK-CERT) are reported correctly. The headline claim—**85% row reduction @ snapshot MSE ≤ 0.10**—rests on GARD-SPARSE under **oracle incidence**, **oracle chain graph**, and **true snapshot mean anchor**, the same favorable setting parent Round 05 already characterized as a **75% reduction at 15/60 rows** when threat reduction uses **mean MSE across seeds**.

Round 01’s `aggregate_stage2()` instead takes the minimum row count over **any** seed×fraction run with per-run MSE ≤ 0.10. At 9 observed rows, **4/5 seeds exceed 0.10** (mean GARD MSE ≈ 0.130); only seed 11 passes (0.088). Under parent-style aggregation, GARD would need **15 rows** (mean MSE ≈ 0.050 at 15 rows), i.e. **75% reduction**, not 85%. The inflated headline is a **metric-definition regression**, not new science.

**Breakthrough gates (config):**

| Criterion | Claimed | Supervisor ruling |
|-----------|---------|-------------------|
| A — ≥25% row reduction @ equal snapshot MSE | `breakthrough_A_25pct: true` | **Conditional / not admissible as run breakthrough** — oracle cheats + per-seed min-rows; parent-equivalent ~75% only |
| B — defense ≥50% attack MSE cut @ ≤10% utility | false | **Fail** (defense worsens T1p on 4/5 seeds) |
| C — cert ≥2× tighter than naive | false | **Fail** (dof term zero at T1p row count) |

Primary lane recommendation (GARD-SPARSE) is reasonable for **research focus**, not for **acceptance**.

---

## GARD-SPARSE audit (required)

### 1. Oracle assignment — **CONFIRMED (critical)**

- `make_incidence()` builds the **true** batch membership matrix `h_full`; `run_stage2_sparse()` sets `h_true = h_full[keep]` and solves with that matrix directly (`benchmark_round01.py:198–214`).
- There is **no** `wrong20`, `wrong40`, or `unknown_random` regime (parent `benchmark_round05.py` had these).
- Proposal and `honesty_notes` disclose oracle assignment; disclosure does **not** remove the oracle cheat for breakthrough claims.

**Implication:** Row-reduction numbers describe an attacker who already knows which samples co-occur in each observed batch gradient. This is the opposite of SHARD’s unknown-assignment bottleneck (parent Round 05 supervisor: assignment error dominates graph prior).

### 2. Oracle graph — **CONFIRMED (critical)**

- `lap = chain_laplacian(cfg.n_samples)` is a fixed path graph over sample index `0…N−1` (`benchmark_round01.py:118–124, 211`).
- Snapshots are synthetic smooth signals over the **same index order** (`make_low_rank_snapshots`: sinusoids on `t = linspace(0,1,N)`).
- No noisy, permuted, co-occurrence, or LS-kNN graph arms (all tested in parent Round 05).

**Implication:** Generator geometry and prior are **aligned by construction**. Gains measure nullspace filling under matched side information, not recoverable graph inference from FL observations.

### 3. Cherry-picked / gamed MSE threshold — **PARTIAL (major)**

- `target_snapshot_mse = 0.10` matches parent Round 05 and `config.json` — **not** an ad-hoc threshold invented in Round 01.
- **However**, threat-reduction accounting is **more lenient than parent**:
  - Parent `compute_threat_reduction()` requires `snapshot_mse_mean <= 0.10` over seeds at a budget (`code/benchmark_round05.py:553–559`).
  - Round 01 requires only **one** `(seed, fraction)` pair with `gard_sparse_mse <= 0.10` (`benchmark_round01.py:248–251`).
- At **9 rows** (fraction 0.15): per-seed GARD MSE = {0.137, 0.158, **0.088**, 0.134, 0.132}; **mean 0.130 > 0.10**.
- At **15 rows**: all seeds pass (mean ≈ 0.050) → parent-consistent **75%** reduction, not 85%.

**Additional oracle side channel:** `mean_snapshot = s_true.mean(axis=0)` is supplied to every solver (`benchmark_round01.py:197, 136–140`) — same issue flagged in parent Round 05 supervisor review.

**Validation:** λ chosen on held-out rows sharing the **same** (oracle) incidence; does not test assignment error.

### 4. Baseline fairness under oracle setting

SHARD-style baseline is plain `solve_map` without graph (`s_ls`). Under shared oracle incidence and mean anchor, comparison is **internally fair** but **externally optimistic** for both methods.

### 5. End-to-end SHARD track

QFL lane SHARD Stage-2 MSE mean ≈ 0.58 with high seed variance (seed 19 ≈ 1.38). Log does not show convergence failure spam in saved round01 log; proposal notes unreliable convergence — **not** used to justify GARD-SPARSE numbers (synthetic Stage-2 lane is separate). Acceptable separation, but Round 02 should not conflate synthetic Stage-2 wins with vendor SHARD until oracle path is fixed or declared diagnostic-only.

---

## Other lanes (brief)

### QFL-SHIELD — fail (correctly reported)

- Mean attack MSE reduction **−27%** (defense helps attacker).
- Utility loss fraction **≫ 10%** (~12.4 vs ‖S‖² scale).
- Rank-2 SVD + σ=0.15 not validated; fixed seed 42 in noise (`apply_shield`) is a minor reproducibility quirk.

### LEAK-CERT — fail (correctly reported)

- T1p observes 70 terminal rows vs N=32 → `cert_dof = 0`; certificate collapses to naive mean broadcast (~1.0).
- Not a proved privacy certificate; heuristic only.

---

## Honesty / reproducibility

| Check | Result |
|-------|--------|
| Metrics computed in code | **Pass** — no hardcoded breakthrough JSON |
| Log ↔ JSON consistency | **Pass** — spot-checked GARD-SPARSE lines vs `artifacts/round01_metrics.json` |
| Oracle setting disclosed | **Pass** — proposal, `honesty_notes`, literature notes |
| Runnable path | **Pass** — `logs/experiment_round01.log`, ~24 s runtime |
| Parent prior cited | **Pass** — Round 05 bridge in `literature/prior_findings_bridge.md` |
| Threat-reduction statistic | **Fail** — per-run min vs parent mean-at-budget; inflates criterion A |
| Independent full rerun (supervisor) | **Not executed** — audit by code + saved artifacts |

No evidence of metric injection. The main integrity issue is **evaluation protocol**, not fraud.

---

## Rubric (1–5)

| Criterion | Score | Note |
|-----------|------:|------|
| Novelty | 2 | GARD-SPARSE replicates parent favorable regime; shield/cert are first-pass negatives |
| Soundness | 3 | Synthetic Stage-2 logic coherent; breakthrough A statistic and oracle stack weaken claims |
| Feasibility | 4 | Three lanes in one script; clear Round 02 extensions in revision log |
| Impact | 2 | No defense or certificate win; row reduction not transferable without assignment stress |
| Evaluability | 4 | Rich per-run JSON; needs aligned threat-reduction tables and assignment ablations |

---

## Critical issues

1. **Criterion A not admissible as breakthrough:** Oracle assignment + oracle graph + oracle mean anchor; parent already reported conditional 75% reduction.
2. **Threat-reduction gaming:** `gard_min_rows_at_target = 9` driven by **one** seed; mean-at-budget would yield **15** rows (75%).
3. **No assignment stress:** Cannot claim QFL/SHARD Stage-2 improvement under realistic unknown membership.
4. **Criteria B and C failed:** Cannot accept on lane A alone; `forbid_accept_on_negative_result_only` applies to treating survey negatives as contribution.

## Major issues

5. **No graph mismatch ablation** in Round 01 (noisy / co-occurrence / wrong graph).
6. **No ridge / low-rank baselines** on Stage-2 lane (parent had these).
7. **QFL-SHIELD utility metric** unnormalized; single (rank, σ) point — not a fair defense study.
8. **LEAK-CERT** criterion C implementation in code checks `tight_over_naive >= 2` (larger cert), while prose wants *tighter* bounds — clarify in Round 02.

## Minor issues

9. Only five seeds; no CIs on row-reduction table.
10. `README.md` supervisor column still empty — update after this review.

---

## Config / policy compliance

- `forbid_accept_before_round: 8` → **ACCEPT forbidden** regardless of lane A flag.
- `require_measurable_breakthrough` → **Not met** for a non-oracle, deployment-relevant threat model.
- `forbid_accept_on_negative_result_only` → Lanes B/C negative; lane A is replication, not independent breakthrough signal.
- `early_accept_requires_two_independent_breakthrough_signals` → **Not satisfied**.

---

## Round 02 demands (mandatory)

### GARD-SPARSE / Stage-2 (primary)

1. **Port parent assignment stress:** `wrong20`, `wrong40`, `unknown_random` with observations from true H, solver from assumed H (same semantics as `code/benchmark_round05.py`).
2. **Port graph stress:** at minimum `gard_noisy_graph`, `gard_cooccurrence_graph`, and `gard_wrong_graph` (or equivalent) alongside oracle chain.
3. **Fix threat-reduction metric:** report both (a) parent-style **mean MSE ≤ 0.10 across seeds** at minimum rows, and (b) per-seed pass rate at each row budget; **pre-register** which gate counts for criterion A.
4. **Mean-anchor ablation:** no anchor / estimated mean from Level-1 only / noisy mean — justify or remove oracle `s_true.mean`.
5. **Baselines:** ridge + low-rank PCA on the same sparse Stage-2 simulator as Round 01.
6. **Sensitivity:** MSE targets {0.05, 0.10, 0.15} in artifacts — show row reduction is not threshold-tuned to 0.10 only.

### ASSIGN-LOCK (new implementation required)

7. Minimal **assignment-hiding** stub: permuted batch-index secure aggregation on published batch means; measure attacker row recovery vs GARD-SPARSE oracle upper bound.

### QFL-SHIELD (redesign or demote)

8. If kept: utility-aware grid on rank and σ with **≤10%** utility constraint under normalized metric; report attack MSE vs undefended T1p **and** vs SHARD Stage-2 on same track.
9. If still harmful after grid: **deprioritize** in proposal with one-paragraph kill criteria.

### LEAK-CERT (redesign or park)

10. Tier-specific rank/sensitivity bound for T1p at **actual** partial row budget (7 rows/epoch), or park until Round 03+.

### Deliverables

11. `rounds/round_02/researcher_proposal.md` addressing every item above with pass/fail table vs config A/B/C.
12. Re-run `python3 code/benchmark_round02.py` (or extend round01 with flags) → new JSON/log; **do not** edit Round 01 artifacts retroactively.
13. `rounds/round_02/revision_log.md` must state whether 85% claim is **withdrawn** under parent-aligned aggregation.

---

## Scientist alignment

Round 01 proposal correctly deprioritizes SHIELD and LEAK-CERT and flags assignment risk. The overclaim is labeling criterion **A met** as “breakthrough” without assignment/graph/oracle-mean removal and without parent-aligned threat accounting. Round 02 should **narrow** the claim to: “conditional replication + assignment-first stress test,” or produce a **new** mechanism (e.g. ASSIGN-LOCK + robust graph) that survives wrong incidence.

---

## Supervisor sign-off

| Field | Value |
|-------|-------|
| Round | 1 |
| Verdict | **REVISE_MAJOR** |
| Criterion A (config) | Flagged true in JSON; **supervisor: not breakthrough-grade** |
| Criterion B | Fail |
| Criterion C | Fail |
| Accept | **No** (policy + substance) |
| Next round focus | Assignment stress + honest threat reduction + ASSIGN-LOCK stub |
