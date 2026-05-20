# QFL-PRIVACY-MAP — Defender / Auditor Framework

**Framework ID:** `QFL-PRIVACY-MAP`  
**Benchmark:** `code/benchmark_round08.py`  
**CLI:** `code/qprivacy_audit.py`  
**Round 08 integration** — consolidates Rounds 01–07 into a single tiered map.

---

## Roles

| Role | Responsibility |
|------|----------------|
| **Defender** | Chooses what the server/client publishes (terminal-only, partial rows, batch means, DP noise). |
| **Auditor** | Runs the strongest allowed attack for that observation class and reports MSE, row budget, and gate pass/fail. |

The map is **not** a single leaderboard: each tier has a different observation target and pass condition.

---

## Tier table (Round 08)

| Tier | Observation | Defender | Auditor | Status | Key metric |
|------|-------------|----------|---------|--------|------------|
| **T1** | Epoch-terminal-only (LASA-QTERM T1) | Terminal-only publish | LASA-QTERM T1 | **IMPOSSIBLE** | T1 MSE ≈ 1.0 (broadcast floor) |
| **T1p** | Partial terminal rows (p&lt;K per epoch) | Limit published terminal rows | LEAK-CERT-T1p v2 | **CERT** (coverage) / **C weak** (honest 2×) | emp ≈ 0.42; cert ≈ 1.08; honest naive/cert ≈ 0.93× |
| **S2-oracle** | Stage-2 batch means + oracle incidence | Rows hideable only with trusted assignment | GARD-SPARSE | **CONDITIONAL THREAT** | 15/60 rows @ MSE ≤ 0.10 (**75%** reduction) |
| **S2-wrong40** | Stage-2 + wrong40 incidence | Assignment hiding required | GARD + JASPER-Q | **BARRIER (ABT-1)** | Mean MSE ≥ 0.15 @ 25–50% rows |

---

## Tier details

### T1 — Impossible (terminal-only)

Under LASA linearity, epoch-terminal observations depend only on the **dataset mean** \(\bar{s}\). Individual snapshots are not identifiable without batch rows or side information.

- **Proof:** `research-artifacts/qfl-terminal-snapshot/paper/impossibility_t1.md`
- **Auditor check:** `QtermAttack` tier `T1` → Hungarian MSE ≈ 1.0 (passive broadcast matches).

### T1p — LEAK-CERT v2 (honest naive)

**LEAK-CERT-T1p v2** combines:

1. Round-04 partial-row honest bound  
2. Ky-Fan singular-value tail at the T1p row budget  
3. Trace-variance floor under hidden within-epoch slots  

**Criterion C (config.json)** requires a certificate **≥2× tighter** than the **honest naive** baseline: broadcast the epoch-mean gradient to every snapshot (no trace inflation).

Round 07 used a **trace-inflated** naive baseline (~16× vs cert), which **games** criterion C. Round 08 reports both; only **broadcast mean** counts for acceptance.

### S2-oracle — GARD-SPARSE conditional threat (75% rows)

With **oracle** incidence and an aligned chain graph, GARD co-occurrence MAP reaches snapshot MSE ≤ 0.10 at **15/60** observed rows — **75%** fewer rows than full SHARD-style LS (60/60).

This is a **conditional upper bound**, not a deployment guarantee: wrong assignment or graph breaks the win (see S2-wrong40).

### S2-wrong40 — Assignment barrier (ABT-1)

**Theorem ABT-1** (`paper/assignment_barrier_theorem.md`): under `wrong40` incidence and level-1 mean anchor, attackers in class \(\mathcal{A} = \{\text{GARD on } \tilde{H}\} \cup \{\text{JASPER-Q}\}\) face a simulator-calibrated floor **MSE ≥ 0.15** at 25–50% row fractions.

Empirical support: Rounds 04–07 JASPER-Q means **0.91–0.96** @ 25% rows; Path 2 @ 50% rows **fails** (`round06_metrics.json`).

---

## Reproduction

```bash
cd research-artifacts/qfl-privacy-breakthrough
python3 code/benchmark_round08.py
python3 code/qprivacy_audit.py --bundle artifacts/round08_metrics.json
```

Outputs:

- `artifacts/round08_metrics.json` — full audit bundle  
- `logs/experiment_round08.log`  
- `rounds/round_08/researcher_proposal.md` — scientist summary  

---

## Config gates (honest)

| Gate | Round 08 status |
|------|-----------------|
| **A** — ≥25% row reduction @ MSE 0.10 (parent mean) | **Pass** on S2-oracle GARD-SPARSE (75%) |
| **B** — defense +50% attack MSE @ ≤10% utility | **Fail** (Snapshot-DP killed R05) |
| **C** — cert ≥2× honest naive, full coverage | **Fail** under honest naive; cert still **covers** empirical T1p |
| **ABT-1** — wrong40 floor | **Pass** |

---

## References

- `code/benchmark_round01.py` — GARD-SPARSE lane  
- `code/benchmark_round06.py` — ABT-1 Path 2 + theorem link  
- `code/benchmark_round07.py` — JASPER-Q v7 + LEAK-CERT v2 (trace naive flagged)  
- `code/qprivacy_map_core.py` — tier audit primitives  
- `config.json` — breakthrough definitions A/B/C  
