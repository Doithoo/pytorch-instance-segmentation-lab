# Penn-Fudan Pedestrian

The official archive contains 170 street images and pedestrian instance-ID masks from Fudan and Penn sources. The download script verifies archive SHA-256 before safe extraction.

Protocol-v2 preparation groups IDs by their nonnumeric source prefix, hash-orders each group with seed 42, and fills the fixed 136 train, 17 valid, and 17 test totals. The committed source composition is:

| Split | Fudan | Penn | Total |
|---|---:|---:|---:|
| train | 59 | 77 | 136 |
| valid | 7 | 10 | 17 |
| test | 8 | 9 | 17 |

Every manifest row stores image/mask paths, dimensions, instance count, and both file hashes. `dataset.yaml` stores format version, strategy, seed, split hashes, label schema, ranges, and dataset identity. Training and checkpoint evaluation reject changed manifests.
