# Round 09 — PROBE-RAND (high-risk breakthrough)

## Lane chosen

**PROBE-RAND** (not SUBSPACE-MIX): server draws fresh probe matrix \(A^{(e)}\) each round, publishes only to clients; SHARD/JASPER attacker sees gradients + wrong40 incidence but not per-round \(A^{(e)}\).

## Hypothesis

Randomizing \(A^{(e)}\) destroys cross-epoch co-occurrence consistency in decoded snapshot coordinates, blocking JASPER under wrong40 while clients still train with published \(A^{(e)}\).

## Setup

- Anchor: JASPER-Q v7 @ **wrong40**, **50%** rows, **level1_estimate** (Round 06/07 parity).
- Decode ablations for attacker:
  - `oracle_per_epoch` — knows each \(A^{(e)}\) (cheating upper bound)
  - `stale_first_epoch` — uses \(A^{(0)}\) for all rounds (legacy single-probe)
  - `pooled_lstsq` — mean of epoch matrices
  - `raw_gradient` — treats gradient vectors as pseudo-snapshots
- Client utility: mean squared error of \(A^{(e)} g\) vs published gradients (should be noise-only).

## Results (5 seeds, mean snapshot MSE)

| Condition | JASPER MSE |
|-----------|------------|
| Round 07 baseline (snapshot rows) | **0.894** |
| PROBE-RAND + oracle decode | **0.897** |
| PROBE-RAND + stale \(A^{(0)}\) decode | **1.002** |
| PROBE-RAND + pooled mean \(A\) | **231.6** (catastrophic) |
| True batch means (no probe noise) | **0.894** |
| Client grad MSE (PROBE-RAND) | **9.9e-5** |

## Verdict — honest failure

1. **Clients train:** published per-round \(A^{(e)}\) keeps gradient MSE at noise floor.
2. **JASPER does not break:** stale decode worsens MSE only ~12% (0.894 → 1.002); still \(\mathcal{O}(1)\) under wrong40, far from barrier gates (0.10–0.15).
3. **Co-occurrence not destroyed:** with correct per-epoch decode, JASPER matches baseline; wrong40 assignment remains the dominant error.
4. **Pooled-A attacker fails hard** — useful negative for naive defenses, not a practical server protocol.

PROBE-RAND is **not** a breakthrough under this threat model. A logging server that records \(A^{(e)}\) would collapse to oracle decode. SUBSPACE-MIX remains deferred.

## Artifacts

- `code/benchmark_round09.py`
- `artifacts/round09_metrics.json`
- `logs/experiment_round09.log`
