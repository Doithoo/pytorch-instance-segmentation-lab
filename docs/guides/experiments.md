# Experiments

Use a new `run.name` per independent experiment. A run records resolved config, split hashes, structured environment, git commit/dirty state, `uv.lock` hash, JSONL lifecycle events, epoch losses/validation metrics, learning rate, duration, peak CUDA memory, and checkpoints.

Resume from `last.pt`, not `best.pt`, when continuing the same trajectory. Immutable data/model/optimization fields and the metrics tail are checked before writing. A deliberate fork may change run/output, total epochs, device, and workers while retaining recorded lineage.

Use `instance-segment compare-runs RUN_A RUN_B --metric valid_mask_map` for model selection metrics or `--metric mask_map` when each run has an evaluation report. Comparison rejects incompatible split hashes, metric protocols, score floors, mask thresholds, or class counts by default. `--allow-incompatible` is for diagnostics, not published rankings.

Do not repeatedly inspect test results while tuning. Select with validation, evaluate the fixed test split once, publish confidence intervals for small datasets, and record checkpoint/source hashes with any reported number.
