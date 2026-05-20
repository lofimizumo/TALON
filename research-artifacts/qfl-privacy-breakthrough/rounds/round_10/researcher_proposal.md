# Round 10 — Researcher Proposal: Acceptance Package (Final)

## Mission

Synthesize the 10-round `qfl-privacy-breakthrough` run into an **acceptance package** with:

- Two **independent** breakthrough signals (config early-accept rule)
- Held-out replication (seeds never used in Rounds 01–09)
- Honest kill-list documentation
- Practitioner tutorial and final paper

**No** `supervisor_review.md` (Round 10 scientist-only per charter).

## Held-out replication

| Check | Method | Seeds |
|-------|--------|-------|
| Oracle GARD-SPARSE @ 25% rows | `run_stage2_sparse` (Round 01) | 41, 43, 47, 53, 59 |
| LEAK-CERT-T1p v2 (honest naive) | `run_leak_cert_t1p_honest` (Round 08) | same |
| ABT-1 wrong40 barrier @ 25%, 50% | GARD + JASPER-Q (Round 06) | same |

Development seeds `{3, 7, 11, 19, 23}` are **excluded** from Round 10 metrics.

## Breakthrough signals (honest)

### Signal 1 — Privacy floor (independent)

- **ABT-1** assignment barrier: GARD + JASPER-Q mean MSE ≥ 0.15 @ wrong40, 25% and 50% rows — **held-out pass**
- **LASA-QTERM T1** impossibility cross-cite: terminal-only MSE ≈ 1.0 (not re-run; cited from parent + Round 08)

### Signal 2 — Conditional threat (independent)

- **Oracle GARD-SPARSE**: all held-out seeds MSE ≤ 0.10 @ 15/60 rows (75% reduction) — **held-out pass**
- Conditional on honest incidence; wrong40 negates deployment claim

### Signal 3 — Secondary (not breakthrough)

- LEAK-CERT covers empirical T1p on all held-out seeds
- Honest naive / cert ≈ 0.93× — criterion **C** correctly **fails**

## Kill list (confirmed, not relitigated)

- Snapshot-DP (R05)
- ASSIGN-LOCK (R02 survey)
- Trace-inflated criterion C (R07→R08 fix)
- PROBE-RAND (R09 honest failure)

## Deliverables

| Artifact | Path |
|----------|------|
| Final paper | `paper/final_privacy_breakthrough.md` |
| Held-out benchmark | `code/benchmark_round10.py` |
| Metrics | `artifacts/round10_metrics.json` |
| Log | `logs/experiment_round10.log` |
| Tutorial | `tutorial/tutorial.md` |
| Config update | `config.json` |
| This proposal | `rounds/round_10/researcher_proposal.md` |
| Revision log | `rounds/round_10/revision_log.md` |

## Results (post-run)

| Audit | Held-out verdict |
|-------|------------------|
| S2-oracle GARD-SPARSE | **Pass** — mean MSE ≈ 0.049 @ 15/60 rows |
| S2-wrong40 ABT-1 | **Pass** — all seeds ≥ 0.15 @ 25% and 50% |
| T1p LEAK-CERT | Coverage **pass**; honest 2× **fail** (expected) |

**Acceptance:** `ACCEPT` — `two_independent_breakthrough_signals: true`

## Reproduction

```bash
cd research-artifacts/qfl-privacy-breakthrough
python3 code/benchmark_round10.py
```
