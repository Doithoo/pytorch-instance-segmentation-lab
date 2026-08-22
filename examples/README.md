# Examples

Run examples from the repository root with `uv run python examples/<file>.py`. Each program demonstrates one contract and avoids duplicating the full trainer.

| Example | Focus | Expected lesson |
| --- | --- | --- |
| `01_instance_target.py` | Boxes, labels, masks, area | The six target fields are aligned and typed |
| `02_mask_to_instances.py` | Indexed masks | Sparse IDs and touching instances remain separate |
| `03_detection_collate.py` | Batching | Variable image sizes/counts stay in lists |
| `04_minimal_training_loop.py` | Training contract | A model can complete one real update |
| `05_checkpoint_prediction.py` | Inference artifacts | JSON, binary masks, and an overlay form one result |

The extension examples in [`extensions/`](extensions/) show the trusted dataset and model factory shapes. Use them as test fixtures and starting points, not as a replacement for registering a supported provider/model.
