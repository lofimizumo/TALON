# Round 10 revision log

## Executed

- Authored `paper/final_privacy_breakthrough.md` — defender/auditor synthesis with Signals 1–3 and kill list.
- Implemented `code/benchmark_round10.py` — held-out seeds 41, 43, 47, 53, 59.
- Ran replication (~8.7s); wrote `artifacts/round10_metrics.json` and `logs/experiment_round10.log`.
- Expanded `tutorial/tutorial.md` for practitioners.
- Updated `config.json` status fields to Round 10 ACCEPT.
- Added `rounds/round_10/researcher_proposal.md` (this round).

## Outcome

| Gate | Result |
|------|--------|
| Two independent breakthrough signals | **Pass** |
| Oracle GARD-SPARSE (Signal 2) | **Pass** (held-out) |
| ABT-1 wrong40 (Signal 1 partial) | **Pass** (held-out) |
| LEAK-CERT honest 2× (Signal 3) | **Fail** (documented) |
| Acceptance status | **ACCEPT** |

## Not done (by charter)

- No `supervisor_review.md`
- No relitigation of PROBE-RAND, Snapshot-DP, or trace-inflated C

## 10-round arc (one line each)

| Round | Headline |
|-------|----------|
| 01 | GARD-SPARSE oracle 75% row win; multi-lane survey |
| 02 | Assignment stress regimes; wrong40 barrier emerges |
| 03 | Barrier stress + DP hybrid — no breakthrough |
| 04 | JASPER-Q diagnostic under wrong40 |
| 05 | Snapshot-DP killed; criterion B fail |
| 06 | ABT-1 theorem + Path 2 milestone |
| 07 | JASPER-Q v7 + LEAK-CERT (trace C gaming) |
| 08 | QFL-PRIVACY-MAP integration; honest C fix |
| 09 | PROBE-RAND honest failure |
| 10 | Held-out replication + acceptance package |
