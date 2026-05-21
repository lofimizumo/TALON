# Practitioner guide: QFL privacy breakthrough

This tutorial explains how to use the **QFL-PRIVACY-MAP** defender/auditor framework produced by the 10-round `qfl-privacy-breakthrough` run. It is written for engineers auditing federated QFL snapshot leakage, not for reproducing every intermediate round.

---

## What you get

| Piece | Purpose |
|-------|---------|
| `paper/final_privacy_breakthrough.md` | Final synthesis — two breakthrough signals + kill list |
| `paper/privacy_map.md` | Tier table (T1, T1p, S2-oracle, S2-wrong40) |
| `paper/assignment_barrier_theorem.md` | ABT-1 proof sketch + falsifiable predictions |
| `code/benchmark_round10.py` | Held-out replication (acceptance) |
| `code/benchmark_round08.py` | Full privacy-map audit (development seeds) |
| `code/qprivacy_audit.py` | CLI over metrics JSON |
| `config.json` | Gates, parent runs, acceptance status |

---

## Quick start (acceptance replication)

```bash
cd research-artifacts/qfl-privacy-breakthrough
python3 code/benchmark_round10.py
```

Expected (~10s): writes `artifacts/round10_metrics.json` and logs to `logs/experiment_round10.log`.

Check acceptance:

```bash
python3 -c "
import json
m = json.load(open('artifacts/round10_metrics.json'))
print(m['acceptance']['status'], m['breakthrough_signals']['two_independent_breakthrough_signals'])
"
```

You should see `ACCEPT True`.

---

## Defender / auditor workflow

1. **Choose a publish tier** (what the server releases):
   - Terminal-only epochs → **T1**
   - Partial terminal rows → **T1p**
   - Stage-2 batch means + trusted assignment → **S2-oracle**
   - Stage-2 batch means + untrusted / wrong incidence → **S2-wrong40**

2. **Run the matching auditor** (strongest attack allowed for that tier):
   - T1: `QtermAttack` tier T1
   - T1p: LEAK-CERT v2 + `QtermAttack` T1p
   - S2-oracle: GARD-SPARSE (`run_stage2_sparse`)
   - S2-wrong40: GARD + JASPER-Q on `wrong40` incidence

3. **Read gates** in the metrics JSON `audits` or `privacy_map.config_breakthrough`.

Do **not** compare MSE across tiers — observation targets differ.

---

## Full privacy-map audit (development seeds)

```bash
python3 code/benchmark_round08.py
python3 code/qprivacy_audit.py --bundle artifacts/round08_metrics.json
```

Filter tiers:

```bash
python3 code/qprivacy_audit.py --bundle artifacts/round08_metrics.json --tiers S2-oracle S2-wrong40
```

---

## Interpreting the two breakthrough signals

### Signal 1 — Privacy floor

- **ABT-1:** Under 40% wrong batch membership, snapshot MSE stays ≥ ~0.15 even with 25–50% of rows and JASPER-Q refinements.
- **T1 impossibility:** Terminal-only gradients leak at most the dataset mean (MSE ≈ 1.0 per-sample). See `research-artifacts/qfl-terminal-snapshot/paper/impossibility_t1.md`.

**Defender action:** Require secure batch indexing / assignment hiding before trusting any row-reduction argument.

### Signal 2 — Conditional threat (oracle)

- GARD-SPARSE can hit MSE ≤ 0.10 with **15/60** rows (75% cut) **only** when incidence is oracle-correct.

**Defender action:** Treat sparse Stage-2 publication as safe **only** with verified assignment; otherwise assume Signal 1 floor.

### Signal 3 — LEAK-CERT (secondary)

- Certificate **covers** measured T1p leak on all seeds.
- Honest naive baseline is **not** 2× worse than the cert (~0.93×) — do not claim criterion C.

---

## What not to use (kill list)

| Lane | Why killed |
|------|------------|
| Snapshot-DP | Does not raise attack MSE enough at acceptable utility (R05) |
| ASSIGN-LOCK | Not implemented; no measured win (R02) |
| Trace-inflated LEAK-CERT naive | Games criterion C (~16×); use broadcast mean only (R08) |
| PROBE-RAND | Random per-round \(A^{(e)}\) does not break JASPER under wrong40 (R09) |

---

## Paths and dependencies

- Vendor SHARD sim: resolved via `code/_paths.py` → parent TALON `vendor/shard_sim`
- LASA-QTERM attacks: `research-artifacts/qfl-terminal-snapshot/code`
- Parent TALON workspace: `/workspace`

---

## PDF export (optional)

```bash
pandoc tutorial/tutorial.md -o tutorial/tutorial.pdf
pandoc paper/final_privacy_breakthrough.md -o paper/final_privacy_breakthrough.pdf
```

---

## Round history pointer

Per-round scientist notes live under `rounds/round_01` … `rounds/round_10`. Rounds 07–10 are scientist-only (no supervisor review files). Final acceptance is documented in `rounds/round_10/researcher_proposal.md` and `config.json`.
