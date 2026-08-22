# Checkpoint Schema

Format version 1 stores model, optimizer and scheduler state; completed/best epoch; best validation metric; label schema; resolved config; split hashes; Python/Torch versions; and Python, NumPy, Torch, and optional CUDA RNG state. Writes use a temporary file and atomic replace.

Resume validates model name, tensor shapes, label schema, exact manifest hashes, and immutable configuration fields. Only run name/output, target epoch count, device, and worker count may change. Existing `metrics.csv` must end at the checkpoint epoch; resuming into a new directory creates a valid header and records checkpoint lineage in `events.jsonl` and `environment.json`.

Dataset-backed evaluation also checks checkpoint split hashes before inference. Single-image prediction does not access a dataset, so consumers must verify the published checkpoint SHA-256 themselves.

Security: resume checkpoints require `torch.load(..., weights_only=False)` for optimizer and RNG state and therefore may execute pickle payloads. Load only trusted checkpoints. For deployment distribution, prefer a separately published weights-only artifact plus signed metadata.
