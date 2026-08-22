# 示例

从仓库根目录使用 `uv run python examples/<file>.py` 运行示例。每个程序只展示一个契约，不重复实现完整 trainer。

| 示例 | 重点 | 应理解的内容 |
| --- | --- | --- |
| `01_instance_target.py` | boxes、labels、masks、area | 六个 target 字段必须对齐且 dtype 正确 |
| `02_mask_to_instances.py` | 索引 mask | 稀疏 ID 和接触实例仍保持独立 |
| `03_detection_collate.py` | batch | 不同图片尺寸/实例数保留为 list |
| `04_minimal_training_loop.py` | 训练契约 | 模型可以完成一次真实更新 |
| `05_checkpoint_prediction.py` | 推理产物 | JSON、二值 mask 和 overlay 组成一次结果 |

[`extensions/`](extensions/) 下的扩展示例展示可信 dataset/model factory 的形状。请把它们作为测试 fixture 和起点，而不是替代正式注册 provider/model 的方式。
