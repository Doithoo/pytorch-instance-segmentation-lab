# 示例

从仓库根目录使用 `uv run python examples/<file>.py` 运行。

1. `01_instance_target.py`：boxes、labels、独立 masks 与 area。
2. `02_mask_to_instances.py`：稀疏实例 ID 不会被合并。
3. `03_detection_collate.py`：不同图像尺寸和实例数保持 list。
4. `04_minimal_training_loop.py`：契约模型完成一次参数更新。
5. `05_checkpoint_prediction.py`：checkpoint 输出 JSON、masks 和 overlay。
