# TALON research run (`06.TALON`)

**Goal:** Find a stronger direction than SHARD by prioritizing Stage 2 / full-method redesign and reducing the requirement for intermediate minibatch gradients. Stage-3-only improvements are background.

**Status:** **Phase 1 ACCEPT** (Round 10) + **Phase 2 scoped ACCEPT** (Round 14, minibatch fix, 2026-05-20)

| Round | Scientist | Supervisor | Verdict |
|-------|-----------|------------|---------|
| 1 | done | done | REVISE_MAJOR (LAPIN loses to SHARD) |
| 2 | done | done | REVISE_MINOR (JOLI wins MSE in compressive regime) |
| 3 | done | done | **ACCEPT_WITH_MINOR** |
| 4 | done | done | REVISE_MAJOR (GARD promising but oracle-driven) |
| 5 | done | done | REVISE_MAJOR / PIVOT_REQUIRED (assignment dominates) |
| 6 | done | done | REVISE_MAJOR / PIVOT_REQUIRED (JASPER negative result) |
| 7 | done | done | ACCEPT_WITH_MAJOR_LIMITATIONS / REFINE_AND_INTEGRATE |
| 8 | done | done | ACCEPT AS FINAL PAPER DIRECTION |
| 9 | done | done | **ACCEPT_WITH_MINOR** (evaluation closure) |
| 10 | done | done | **ACCEPT** (paper-ready: proofs, decoder, frozen MLP) |
| 11 | done | done | **REVISE_MAJOR** (TANGO-MB scaling fix; not yet fundamental) |
| 12 | done | done | **REVISE_MAJOR** (proofs, counts, ablations) |
| 13 | done | done | **REVISE_MINOR** (JOINT/COUPLED/DOPT failed honestly) |
| 14 | done | done | **ACCEPT_WITH_MINOR** (Phase 2 scoped accept) |

## Final Outcome: TALON / TANGO

**TALON**: Terminal Aggregate Leakage with Observation tiers and Non-identifiability limits.

**TANGO**: Terminal Aggregate Neural Gradient Observation — active server bias probes recover class-level hidden prototypes without intermediate minibatch gradients.

Supported thesis:

> Active terminal probes identify class-level hidden prototypes under terminal-only observation. **TANGO-MB** corrects minibatch \(1/B\) gradient scaling (\(T_{\mathrm{eff}}=T\cdot N/B\)) at \(W_0=0\); harder joint/trajectory estimators do not beat it. Individual samples remain non-identifiable; SHARD Tier-3 needs intermediate gradients.

See **`paper/phase2_scope.md`** for scoped Phase-2 claims and documented failures.

## Paper artifacts

| Path | Role |
|------|------|
| `paper/proofs.md` | Formal proofs (Theorems 1–2, Lemmas A–B) |
| `paper/method.tex` | LaTeX method + limits |
| `paper/draft.md` | Limits-first manuscript draft |
| `paper/outline.md` | Section skeleton |

## Reproduce (Cloud / Linux)

```bash
cd /workspace
python3 code/benchmark_round08.py   # TANGO stress tests
python3 code/benchmark_round09.py   # baselines + extended stress
python3 code/benchmark_round10.py   # decoder probe + frozen MLP
python3 code/benchmark_round11.py   # TANGO-MB minibatch fix
python3 code/benchmark_round14.py   # Phase 2 primary + SHARD cross
python3 code/shard_cross_round14.py # SHARD level3_invert tier demo
```

## Key results (Round 10)

| Experiment | Headline |
|------------|----------|
| TANGO balanced (R09) | prototype MSE `0.000151`, vs passive `0.143` (**950×**) |
| Decoder balanced | hidden/pixel MSE `0.000151`, corr ≈ 1.0 |
| Decoder minibatch | pixel MSE `6.59` (honest failure) |
| Frozen MLP balanced | TANGO `0.00302` vs passive `0.367` (**218×**) |
| Individual recovery | MSE ≈ within-class floor (non-identifiable) |

## Layout

| Path | Role |
|------|------|
| `config.json` | Run metadata and acceptance criteria |
| `rounds/round_0N/` | Proposals, reviews, revision logs |
| `code/` | Benchmarks and methods |
| `artifacts/` | Metrics and plots |
| `tutorial/` | Idea tutorial (`tutorial.md`) |
| `paper/` | Proofs and manuscript draft |
| `vendor/shard_sim/` | Vendored SHARD simulator (rounds 01–03) |

## Cursor Cloud

Skills and subagents: `.cursor/skills/autonomous-research/`, `.cursor/agents/`. See `AGENTS.md`.
