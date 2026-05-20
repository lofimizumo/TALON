# Vendored dependencies

Copied from `01.SHARD/supplementary_materials/code/shard_sim/` so TALON benchmarks run without importing from the SHARD repo.

| Package | Used by |
|---------|---------|
| `shard_sim/` | `benchmark_round01.py`–`03.py`, `joli_invert.py`, `lapin_invert.py` |

Rounds 04–08 are self-contained (numpy/matplotlib only).

To refresh from SHARD:

```bash
rsync -a --delete \
  "../01.SHARD/supplementary_materials/code/shard_sim/" \
  "vendor/shard_sim/"
```
