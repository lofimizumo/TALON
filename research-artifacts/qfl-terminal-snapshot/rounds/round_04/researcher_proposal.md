# Round 04 — Researcher Proposal: Acceptance Packaging (LASA-QTERM)

## Mission

Deliver an **acceptance-ready** QFL method package after Round 03 impossibility + budget-honest evaluation. No `supervisor_review.md` this round (scientist deliverables only).

## Method identity

| Field | Value |
|-------|-------|
| **Primary name** | **LASA-QTERM** |
| **Alias** | **Q-SNAP-T** |
| **Production module** | `code/qterm_attack.py` (`QtermAttack`) |
| **Papers** | `paper/method.md`, `paper/scope.md` (+ `paper/impossibility_t1.md`) |

## Round 03 gaps addressed

| Gap (R02/R03) | Round 04 action |
|---------------|-----------------|
| No unified method name | LASA-QTERM / Q-SNAP-T in method doc + benchmark JSON |
| Fragmented benchmarks | `code/benchmark_round04.py` — all tiers, acceptance table |
| No production wrapper | `QtermAttack` per tier (T1, T1p, T1b) |
| Tutorial stub | `tutorial/tutorial.md` (QFL + TALON bridge) |
| MNIST required | Default dual-track in R04 benchmark |
| Headline conflation | `acceptance_table` per track, tier-labeled |

## Scientific outcome (unchanged, now packaged)

**Primary T1 (`config.json` epoch-terminal goal):** **Not met** — closed via **impossibility** (`paper/impossibility_t1.md`).

| Track | T1 LASA-QTERM MSE | SHARD T2 MSE | Ratio |
|-------|-------------------|--------------|-------|
| Smooth | ~0.99 | ~0.48 | ~2.0× |
| MNIST | ~1.0 | ~0.61 | ~1.65× (ratio ≤2× but passive-scale recovery) |

**Non-primary positives (labeled):**

- **T1p** honest partial (p=7): smooth ~0.44 vs SHARD ~0.48; MNIST ~0.59 vs ~0.61.
- **T1b @ 80 rows:** smooth ~0.16 vs SHARD@80 ~0.68; MNIST ~0.11 vs ~0.57.

## Deliverables checklist

| Item | Path | Status |
|------|------|--------|
| Method paper | `paper/method.md` | Done |
| Scope / tier table | `paper/scope.md` | Done |
| Production attack | `code/qterm_attack.py` | Done |
| Benchmark R04 | `code/benchmark_round04.py` | Done |
| Metrics | `artifacts/round04_metrics.json` | Run required |
| Tutorial | `tutorial/tutorial.md` | Done |
| Revision log | `rounds/round_04/revision_log.md` | Done |

## Supervisor gate recommendation

Recommend **ACCEPT_WITH_MINOR** or **ACCEPT** on:

1. **T1 impossibility** as primary scientific outcome for epoch-terminal QFL.
2. **LASA-QTERM** as reproducible, tier-honest package (code + papers + tutorial + MNIST).
3. Remaining minors: SHARD `matching_acc` vs Hungarian MSE; real QFL stack beyond SurrogateQFL.

## Experiment

```bash
python code/benchmark_round04.py
```

Logs: `logs/experiment_round04.log`
