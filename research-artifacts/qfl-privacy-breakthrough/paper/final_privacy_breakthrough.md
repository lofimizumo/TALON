# QFL Privacy Breakthrough — Final Synthesis (Round 10)

**Run:** `qfl-privacy-breakthrough`  
**Framework:** QFL-PRIVACY-MAP (defender / auditor)  
**Status:** ACCEPT — two independent breakthrough signals on held-out seeds  
**Replication:** `code/benchmark_round10.py` → `artifacts/round10_metrics.json`

---

## Executive summary

After ten rounds, this run delivers **two independent, measurable breakthrough signals** for quantum federated learning (QFL) snapshot privacy under the SHARD / LASA observation model:

1. **Privacy floor under mis-assignment** — Theorem **ABT-1** (assignment barrier) plus **LASA-QTERM T1 impossibility** (cross-cite) bound what any co-occurrence attacker can recover when batch incidence is wrong or only terminal means are published.
2. **Conditional row-efficiency threat** — **Oracle GARD-SPARSE** reaches snapshot MSE ≤ 0.10 with **75% fewer** Stage-2 rows (15/60) when assignment and graph are honest.

A **secondary** result: **LEAK-CERT-T1p v2** certifies partial-terminal leak with full empirical coverage, but does **not** beat the honest naive baseline by 2× (Round 07 trace inflation was gaming criterion C).

Several lanes were **killed** with honest negative results: Snapshot-DP, ASSIGN-LOCK, trace-inflated criterion C, PROBE-RAND.

---

## Defender / auditor framing

| Role | Question answered |
|------|-------------------|
| **Defender** | What may the server publish (terminal-only, partial rows, batch means, incidence trust)? |
| **Auditor** | Given that tier, what is the strongest attack MSE, row budget, and gate pass/fail? |

The privacy map (`paper/privacy_map.md`) is tiered: each observation class has its own pass condition. There is no single leaderboard across tiers.

---

## Signal 1 — Assignment barrier + T1 impossibility (privacy floor)

### ABT-1 (assignment barrier)

Under SHARD Stage-2 with **wrong40** incidence (40% membership corruption per row) and **level1_estimate** mean anchor, attackers in class

\[
\mathcal{A} = \{\text{GARD co-occurrence on } \tilde{H}\} \cup \{\text{JASPER-Q with T1p warm-start}\}
\]

face a simulator-calibrated Hungarian snapshot MSE floor **≥ 0.15** at 25–50% observed rows. Proof sketch and falsifiable predictions: `paper/assignment_barrier_theorem.md`.

**Development ensemble** (seeds 3, 7, 11, 19, 23): mean JASPER-Q MSE ≈ 0.91–0.96 @ 25% rows; GARD and JASPER both ≥ 0.15 (`round06_metrics.json`, `round08_metrics.json`).

**Held-out replication** (seeds 41, 43, 47, 53, 59): all five seeds ≥ 0.15 @ 25% and 50% for both GARD and JASPER-Q (`round10_metrics.json`, `audits.S2-wrong40.pass: true`).

### LASA-QTERM T1 cross-cite (terminal-only floor)

When the defender publishes **epoch-terminal gradients only**, individual snapshots are **not identifiable** — recovery collapses to the dataset mean (Hungarian MSE ≈ 1.0). This is independent of assignment error and establishes a **broadcast privacy floor** for terminal-only QFL.

- **Proof:** `research-artifacts/qfl-terminal-snapshot/paper/impossibility_t1.md`
- **Auditor check:** `QtermAttack` tier T1 (`round08_metrics.json`, mean MSE ≈ 0.9999)

### Combined interpretation (honest)

Mis-assignment and terminal-only publication are **orthogonal floors**: even with more rows or smarter graph attacks (JASPER-Q), wrong incidence prevents deployment-scale snapshot recovery; terminal-only publication prevents individual recovery regardless of rows. **Assignment hiding / secure batch indexing** is necessary for any row-budget win.

---

## Signal 2 — Oracle GARD-SPARSE (conditional threat quantification)

With **oracle** incidence and a chain graph prior, **GARD-SPARSE** co-occurrence MAP reaches snapshot MSE ≤ **0.10** at **15/60** observed rows — **75%** row reduction vs full SHARD-style least squares (60/60).

| Ensemble | Seeds | Mean GARD-SPARSE MSE @ 25% rows |
|----------|-------|----------------------------------|
| Development | 3, 7, 11, 19, 23 | ≈ 0.041 (`round08_metrics.json`) |
| Held-out | 41, 43, 47, 53, 59 | ≈ 0.049 (`round10_metrics.json`) |

This is a **conditional upper bound** for **honest-incidence QFL**: it quantifies how much an auditor can recover when assignment and graph are trusted. It is **not** a deployment guarantee — **wrong40** breaks the win (Signal 1).

Criterion **A** in `config.json` (≥25% row reduction @ MSE 0.10) **passes** on oracle GARD-SPARSE only.

---

## Signal 3 (secondary) — LEAK-CERT T1p

**LEAK-CERT-T1p v2** (Ky-Fan tail + trace-variance floor) provides a **valid upper bound** on partial-terminal leak:

| Metric | Development | Held-out |
|--------|-------------|----------|
| Empirical T1p MSE (mean) | ≈ 0.42 | ≈ 0.42 |
| Cert tight upper (mean) | ≈ 1.08 | ≈ 1.08 |
| Honest naive / cert ratio | ≈ 0.93× | ≈ 0.93× |
| Covers empirical (all seeds) | yes | yes |

**Criterion C** (cert ≥2× tighter than **honest** naive broadcast) **fails** — correctly. Round 07 passed C only with a **trace-inflated** naive (~16×), which was gaming. The cert is still useful for **coverage auditing**; it does not claim 2× tightening over honest baseline.

---

## Kill list (honest negatives)

| Lane | Round | Outcome |
|------|-------|---------|
| **Snapshot-DP** | 05 | Failed criterion B (attack MSE not raised enough at acceptable utility) |
| **ASSIGN-LOCK** | 02 | Survey-only; no measurable breakthrough |
| **Trace-inflated C naive** | 07→08 | Gaming removed; honest C fails |
| **PROBE-RAND** | 09 | Per-round random \(A^{(e)}\) does not break JASPER under wrong40 (~0.89→1.00 MSE) |

---

## Config gates (final, honest)

| Gate | Definition | Status |
|------|------------|--------|
| **A** | ≥25% row reduction @ MSE 0.10 (parent mean) | **Pass** (oracle GARD-SPARSE, dev + held-out) |
| **B** | Defense +50% attack MSE @ ≤10% utility | **Fail** (Snapshot-DP killed) |
| **C** | Cert ≥2× honest naive | **Fail** (cert covers emp; ratio ≈ 0.93×) |
| **ABT-1** | wrong40 floor ≥ 0.15 | **Pass** (dev + held-out) |
| **T1** | Terminal-only impossibility | **Pass** (cross-cite LASA-QTERM) |
| **Two independent signals** | Early accept rule | **Pass** (Signal 1 + Signal 2) |

---

## Reproduction

```bash
cd research-artifacts/qfl-privacy-breakthrough
python3 code/benchmark_round10.py          # held-out acceptance replication
python3 code/benchmark_round08.py          # full QFL-PRIVACY-MAP bundle (dev seeds)
python3 code/qprivacy_audit.py --bundle artifacts/round08_metrics.json
```

---

## Artifact index

| Artifact | Path |
|----------|------|
| Final synthesis (this document) | `paper/final_privacy_breakthrough.md` |
| Privacy map | `paper/privacy_map.md` |
| ABT-1 theorem | `paper/assignment_barrier_theorem.md` |
| Held-out benchmark | `code/benchmark_round10.py` |
| Practitioner tutorial | `tutorial/tutorial.md` |
| Round 10 proposal | `rounds/round_10/researcher_proposal.md` |
| Config | `config.json` |

**No** `supervisor_review.md` for Round 10 (scientist acceptance package per charter).
