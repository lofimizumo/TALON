# VQC stack bridge (prior runs)

## Stack definition (this run)

1. **Data:** MNIST via `FederatedDataLoader` or torchvision (real pixels).
2. **Encode:** `SurrogateQFL.encode` = \(\cos(Wx+b)\) (LASA surrogate).
3. **Observe:** batch gradients \(g^{(e,k)} = A^{(e)} \bar{s}^{(e,k)}\) or weaker terminal tiers (LASA-QTERM).
4. **Recover snapshots:** SHARD L2, LASA-QTERM T1p/T1b, GARD-SPARSE (oracle assignment).
5. **Invert:** `ShardAttacker.level3_invert` or `joli_invert` → image \(x \in [0,1]^d\).

## Prior snapshot-only MNIST (not sufficient)

`qfl-terminal-snapshot` benchmark compared **snapshot MSE** on encoded MNIST vectors without full L3 image reconstruction. This run requires **image MSE / PSNR** after Level 3.

## Targets (`config.json`)

- Input MSE ≤ **0.05** (Hungarian-aligned or fixed-order per seed policy in benchmark)
- PSNR ≥ **18 dB**
- Weak path (T1p/T1b) must produce **visible** reconstructions, not only oracle SHARD

## Experiment levers (supervisor should push)

- `dim_g` (100–400), `n_epochs`, `batch_size`, image size 8 vs 28
- L3: `n_batch`, Adam steps, JOLI TV, lstsq seeds
- Snapshot path quality before L3 (dominant bottleneck)
