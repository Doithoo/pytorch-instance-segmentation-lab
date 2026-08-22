# Examples

Run examples from the repository root with `uv run python examples/<file>.py`.

1. `01_instance_target.py`: boxes, labels, independent masks, and area.
2. `02_mask_to_instances.py`: sparse instance IDs are not merged.
3. `03_detection_collate.py`: variable image and instance counts remain lists.
4. `04_minimal_training_loop.py`: a contract model completes one update.
5. `05_checkpoint_prediction.py`: a checkpoint creates JSON, masks, and overlay.
