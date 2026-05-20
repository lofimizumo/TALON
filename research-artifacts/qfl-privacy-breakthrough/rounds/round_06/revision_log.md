# Round 06 Revision Log

## Scope

Round 05 killed Snapshot-DP (0/420 feasible B cells). Round 06 executes **Path 2** (50% rows milestone) and **Path 3** (assignment barrier theorem), per Round 05 supervisor demands.

## Changes

- Added `code/benchmark_round06.py` — GARD vs JASPER-Q @ `wrong40`, fraction 0.50, `level1_estimate`; used-vs-true residuals; T1p level1 audit; oracle warm-start off.
- Added `paper/assignment_barrier_theorem.md` (`ABT-1`) — wrong incidence rank deficiency + falsifiable MSE floor.
- Updated `config.json` — `current_round: 6`, criterion B wording.
- Pre-registered `rounds/round_06/researcher_proposal.md`.

## Results

| Gate | Outcome |
|------|---------|
| `A_wrong40_mse_0.15_at_50pct_rows` | **Fail** (0/5 seeds; best attacker JASPER-Q mean **0.908**) |
| GARD @ 50% mean MSE | **1.156** |
| JASPER-Q @ 50% mean MSE | **0.908** |
| Theorem P1 @ 25% (JASPER mean) | **0.947** ≥ 0.15 — prediction holds |
| T1p `level1_mean_recovery` mean | **0.435** (Round 05 flat-mean ref 0.427) |
| Oracle JASPER warm-start off | **Yes** (`t1p_warm_blend=0`) |

Runtime ~7.3s. Log: `logs/experiment_round06.log`.

## Honesty

- Path 2 is an **intermediate** milestone, not criterion A acceptance.
- Theorem is a **proof sketch** tied to the simulator; empirical gate is separate.
