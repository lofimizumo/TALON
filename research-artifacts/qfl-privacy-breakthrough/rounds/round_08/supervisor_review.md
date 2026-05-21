# Round 08 — Supervisor Review

## Verdict

**REVISE_MAJOR**

Round 08 delivers **QFL-PRIVACY-MAP** — a reproducible defender/auditor integration (`paper/privacy_map.md`, `code/qprivacy_map_core.py`, `code/qprivacy_audit.py`, `artifacts/round08_metrics.json`). The round **correctly retires Round 07 criterion-C gaming** (honest naive **≈ 0.93×** cert, **0/5** seeds at ≥2×) and documents tier status without relitigating T1. **This is a valid privacy *audit* deliverable, not a privacy-preserving *breakthrough*.** Oracle **GARD-SPARSE 75%** row reduction is **conditional threat quantification** (Round 01 replay under oracle incidence), **not** a wrong40 deployment attack win. **ACCEPT is forbidden** — `config.json` honest **C** still fails, criterion **A** does not transfer to `wrong40`, and `early_accept_requires_two_independent_breakthrough_signals` is unmet on breakthrough grounds.

---

## Executive summary

Deliverables: `paper/privacy_map.md`, `code/qprivacy_map_core.py`, `code/qprivacy_audit.py`, `code/benchmark_round08.py`, `artifacts/round08_metrics.json`, `rounds/round_08/researcher_proposal.md`, `rounds/round_08/revision_log.md`. Supervisor re-ran `python3 code/benchmark_round08.py` (~6.8s); `config_breakthrough` flags match JSON: `T1_impossibility_confirmed=true`, `A_gard_oracle_75pct_reduction_at_mse_0.10=true`, `C_leak_cert_2x_honest_naive=false`, `ABT1_wrong40_barrier=true`.

| Question | Supervisor ruling |
|----------|-------------------|
| Is **QFL-PRIVACY-MAP** a valid privacy-preserving deliverable? | **Valid audit / limits map** — **not** a deployable defense package |
| Oracle GARD **75%** @ MSE ≤ 0.10 — attack or audit? | **Audit / conditional upper bound** — oracle + chain graph; **not** wrong40 breakthrough |
| Honest **C** fail acknowledged? | **Yes** — proposal, `privacy_map.md`, JSON, and `leak_cert_fix` block align |

---

## QFL-PRIVACY-MAP — privacy deliverable vs audit framework

**Ruling: audit framework (acceptable Round 08 mission); not breakthrough defense.**

The map correctly separates **defender posture** (what is published) from **auditor channel** (strongest attack allowed for that tier). That is scientifically useful for a **tiered leakage paper** and matches Round 07’s demand for honest integration. It does **not** constitute:

- A new privacy-preserving mechanism satisfying **B** (killed R05),
- Honest **C** (cert not ≥2× tighter than mean-only naive),
- Production **A** on `wrong40` (JASPER still **≈ 0.91** @ 50% rows in the same bundle).

**Forbidden framing:** “QFL-PRIVACY-MAP solves QFL privacy” or “Round 08 passes breakthrough gates.” **Allowed framing:** “Integrated reproducible audit of what each observation class permits, with explicit conditional vs barrier tiers.”

---

## Oracle GARD-SPARSE 75% — attack or audit?

**Ruling: conditional threat quantification (audit), not deployment attack success.**

From `privacy_map.audits.S2-oracle` (supervisor re-run):

| Metric | Value | Interpretation |
|--------|------:|----------------|
| Fraction | **0.25** | 15/60 rows |
| Row reduction vs full SHARD LS | **75%** | Arithmetic on oracle lane only |
| GARD-SPARSE mean MSE | **0.0499** | ≤ target **0.10** |
| Assignment | **Oracle** + chain graph | Trusted incidence — **not** `wrong40` |
| `parent_mean_reaches_target` | **true** | Parent-mean gate under oracle |

This **replicates Round 01** under the integration bundle. It is the **upper bound** tier in the map: “if assignment and graph are correct, sparse co-occurrence MAP can reach low MSE at 25% rows.” It is **not** evidence that SHARD/JASPER under **wrong40** can be beaten at the same row budget — `S2-wrong40` shows JASPER **≈ 0.907** @ 50% with barrier **holds**.

**Supervisor mapping to `config.json` criterion A:**

| Reading | Ruling |
|---------|--------|
| **A** as “≥25% row reduction @ equal snapshot MSE (any regime)” | **Pass only under oracle** — label **`A_oracle_conditional`**, not global **A** |
| **A** as production wrong40 / level1 breakthrough | **Fail** — unchanged from Rounds 04–07 |

Do **not** promote `A_gard_oracle_75pct_reduction_at_mse_0.10: true` to README or accept memos without the **oracle + graph** qualifier.

---

## Honest criterion C — acknowledged?

**Ruling: yes — honestly reported and correctly gated.**

`T1p` audit (all seeds):

| Metric | Mean | Pass? |
|--------|-----:|:-----:|
| Empirical T1p MSE | **0.430** | — |
| Cert tight upper | **1.080** | Coverage **5/5** |
| Honest naive (broadcast) | **1.000** | — |
| `naive_broadcast / cert` | **0.926** | **Fail** 2× (**0/5**) |
| Trace-inflated / cert | **14.81** | Diagnostic only; `round07_trace_inflated_gaming_detected: true` |

Round 08 fixes the Round 07 supervisor finding: trace naive is retained in JSON for **gaming detection**, not as primary **C**. Scientist proposal and `leak_cert_fix` in metrics JSON are **aligned** with Round 07 §4.

**Formal kill path:** Per Round 07 demands, LEAK-CERT as a **breakthrough C lane** should be **killed for rounds 9–10** unless mean-only 2× is achieved — Round 08 confirms **0/5**; team should add one-paragraph kill in Round 10 `revision_log.md` (carry demand).

---

## Breakthrough gates (`config.json`)

| Criterion | Round 08 | Supervisor ruling |
|-----------|----------|-------------------|
| **A** (measurable breakthrough) | Oracle GARD pass only | **Conditional audit signal** — **not** wrong40 **A** |
| **B** | Not attempted (R05 kill) | **Fail / N/A** |
| **C** (honest mean-only ≥2×) | `C_leak_cert_2x_honest_naive: false` | **Fail** — correctly documented |
| `require_measurable_breakthrough` | Not met on honest **C** or wrong40 **A** | **Fail** |
| `forbid_accept_on_limits_essay_only` | Map + ABT-1 + oracle quant | **Limits-heavy** — needs Round 10 second signal |
| `forbid_accept_on_negative_result_only` | Integration positive | **Not negative-only**, but **not breakthrough** |

---

## Honesty / reproducibility audit

1. **Benchmark re-run** — Gates and per-seed LEAK-CERT ratios match stored JSON (minor float drift on emp T1p only).
2. **No injected metrics** — `benchmark_round08.py` computes tiers from shared attack primitives; CLI reads bundle only.
3. **`hard_overlap` mislabel** — Not present in Round 08 aggregates (Round 07 fix carried).
4. **Scientist-only charter** — Round 08 omitted `supervisor_review.md` at run time; this document closes the gap.
5. **Tier table honesty** — `privacy_map.md` gate table lists **C fail** and **A pass on S2-oracle** with qualifiers; acceptable if external copy keeps oracle qualifier.

**Code note (Minor):** `cert_tight_upper_bound` still **≈ 1.08** constant across seeds (`cert_trace_slack`) — coverage holds but bound is not seed-adaptive (Round 04/07 carry-over).

---

## Feasibility & resources

- Runtime **~7s** for full map bundle — excellent for replication.
- CLI `qprivacy_audit.py` enables third-party re-audit without re-running sweeps.
- Dependencies trace to Rounds 01, 06, 07 modules — integration scope is proportionate.

---

## Theory & assumptions

| Tier | Theory status |
|------|----------------|
| **T1** | Cites `qfl-terminal-snapshot` impossibility — **not relitigated**; empirical MSE **≈ 1.0** consistent |
| **T1p** | Ky-Fan + trace floor **sketch**; honest **C** fails → cert is **not** proved 2× tighter than contract naive |
| **S2-oracle** | Empirical conditional bound; assumes oracle incidence |
| **S2-wrong40** | **ABT-1** sketch + empirical floor — **limits object**, not defense |

---

## Critical issues

1. **Gate A label overload** — JSON flag `A_gard_oracle_75pct_reduction_at_mse_0.10` can be misread as global **config.json A** pass; must be tier-scoped in all external claims.
2. **No second independent breakthrough signal** — Oracle quant + ABT-1 barrier are **complementary limits**, not **C** or wrong40 **A**; `early_accept_requires_two_independent_breakthrough_signals` still unmet for breakthrough accept.
3. **Round 07 Path C-honest v3 not attempted** — Integration only; **0/5** on mean-only 2× should trigger formal LEAK-CERT kill paragraph by Round 10.

## Major issues

4. **S2-wrong40 uses mixed epoch/sample configs** from upstream rounds — map documents barrier but does not unify GARD/JASPER incidence grids (Round 06/07 carry-over).
5. **Missing supervisor review at delivery** — Policy violation for rounds 8–10 window; corrected here.

## Minor issues

6. `T1p` status string **CERT_WEAK** vs paper **CERT** — align nomenclature (`CERT_COVERAGE_ONLY`).
7. `README.md` supervisor table likely still stale — Round 10 demand.

---

## Rubric scores (1–5)

| Dimension | Score | Note |
|-----------|------:|------|
| Novelty | 3 | First integrated map + CLI; tiers mostly recomposed |
| Soundness | 4 | Honest **C** fix; oracle **A** correctly scoped |
| Feasibility | 5 | Fast reproducible bundle |
| Impact | 3 | Strong audit artifact; no new defense |
| Evaluability | 4 | JSON + CLI; gate naming needs care |

---

## Round 10 path — integrated audit ACCEPT bar (pre-registered)

**Default verdict for Round 10 remains REVISE_MAJOR** unless the team earns **ACCEPT** on the **integrated privacy audit framework** with **two independent signals** (not fake **C**, not oracle-only headline):

| Signal pair (choose one primary pair) | Requirements |
|---------------------------------------|--------------|
| **Pair 1** | **ABT-1** (written theorem + falsifiable wrong40 checks @ pre-registered fractions) **+** **oracle sparse threat quant** (GARD-SPARSE 75% with full protocol spec: rows, graph λ, incidence source) |
| **Pair 2** | **T1 impossibility cite** (parent proof + reproduced MSE floor) **+** **cert coverage** (all seeds, empirical ≤ cert, explicit statement that **C 2× fails** and is out of scope) |

**Additional Round 10 mandatory items:**

1. `rounds/round_10/revision_log.md` **kill list**: Snapshot-DP (R05), ASSIGN-LOCK (R04), trace-inflated **C** (R07), honest **C** as breakthrough (R08), PROBE-RAND (R09), Path 2 @ 0.15 (R06–07).
2. `rounds/round_10/acceptance_decision.md` only if round index ≥ 8, **0 Critical** on final replication, and pair above is complete.
3. `README.md` supervisor table rounds **1–10**.
4. Held-out seed replication **{5, 13, 17, 29, 31}** for any metric cited in accept memo (Round 07 demand — **not done** in Round 08/09).
5. **Forbidden:** ACCEPT on trace-inflated **C**; ACCEPT on oracle **A** without wrong40 disclaimer; ACCEPT on PROBE-RAND negative only; ACCEPT on JASPER diagnostic alone.

---

## Actionable suggestions (prioritized)

1. Add `config_breakthrough.A_wrong40` (explicit **false**) beside `A_gard_oracle_*` in JSON and `privacy_map.md`.
2. Publish one-paragraph **LEAK-CERT breakthrough kill** in Round 09/10 `revision_log.md`.
3. Run held-out seed bundle for map tiers before Round 10 accept memo.
4. Rename `A_gard_oracle_75pct_reduction_at_mse_0.10` → `A_oracle_conditional_pass` in code/JSON to prevent claim drift.

---

## Supervisor sign-off

| Field | Value |
|-------|-------|
| Round | 8 |
| Verdict | **REVISE_MAJOR** |
| QFL-PRIVACY-MAP | **Valid audit framework** — not defense breakthrough |
| Oracle GARD 75% | **Conditional threat quant (audit)** |
| Honest **C** fail | **Acknowledged — correct** |
| Breakthrough **A/B/C** | **Not met** (oracle **A** only, scoped) |
| Accept | **No** |
| Round 10 | **Integrated audit + two-signal pair or REVISE_MAJOR** |
