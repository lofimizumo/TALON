# Prior findings (do not re-discover without citation)

## LASA-QTERM (`qfl-terminal-snapshot`)

- **T1 epoch-terminal only:** Individual snapshot recovery **impossible** (only \(\bar{s}\) identifiable).
- **T1p:** Honest partial minibatch terminals can recover individuals (~0.45 MSE vs SHARD ~0.48).
- **T1b:** Per-client B=1 terminals recover individuals strongly at matched row budget.
- **Implication for privacy research:** Defenses should target **row budget** and **assignment structure**, not only mean obfuscation.

## Parent TALON (`/workspace`)

- Terminal aggregate prototypes (TANGO-MB); not individual snapshots.
- GARD: graph prior helps Stage-2 only with oracle assignment.
- JASPER: unknown assignment is a hard barrier.

## Candidate breakthrough lanes (this run)

1. **SHARD-SPARSE / GARD+:** Fewer intermediate rows at equal MSE (extend parent Round 05).
2. **QFL-SHIELD:** Client-side snapshot noise / subspace projection before gradient publish.
3. **LEAK-CERT:** Provable ε-style bound on snapshot MSE given published tier (T1/T1p/T1b).
4. **PROBE-DETECT:** Server active probes detectable by clients.
5. **ASSIGN-LOCK:** Assignment-hiding via permuted secure aggregation on batch means.
