# 训练

[English](04-training.md) | [文档导航](../README.zh-CN.md)

先使用小配置，确认数据契约后再去掉 limit 并选择 GPU 配置：

```bash
uv run instance-segment train --config configs/learning_minimal.yaml --dry-run --device cpu
uv run instance-segment train --config configs/reference_maskrcnn.yaml --device cuda
uv run instance-segment train --config configs/reference_maskrcnn.yaml --resume artifacts/reference-maskrcnn/last.pt
```

## 训练会记录什么

完整运行会校验 manifest hash，解析配置，记录 Python/Torch/设备/git/lock provenance，并写入 `metrics.csv`、`events.jsonl`、`best.pt` 和 `last.pt`。指标包含五个分项 loss、验证 bbox/mask AP 与 AR、学习率、每轮耗时和 CUDA 峰值显存。

`best.pt` 只根据 valid 的 `mask_map` 选择。即使最后一轮不满足 `evaluate_every`，也会强制评估最后一轮。训练器不会加载 test split；选模完成后再运行 `evaluate --split test`。

## Dry-run 与恢复

`--dry-run` 会构造配置中的模型并执行一次真实 optimizer update，但不创建运行目录，能尽早捕获 target dtype、图片 shape、预测头、loss 和 backward 错误。

使用 `--resume artifacts/<run>/last.pt` 继续同一轨迹。恢复会还原 model、optimizer、scheduler 和 RNG state，并拒绝变化的不可变配置、manifest hash 或不一致的 metrics 尾部。只有运行名称/输出目录、目标 epoch 总数、device 和 worker 数可以修改。要有意分叉实验，请不要使用 `--resume`，而是创建新运行。

长时间运行前请阅读[配置参考](../reference/config-reference.zh-CN.md)、[Checkpoint 结构](../reference/checkpoint-schema.zh-CN.md)和[实验管理](../guides/experiments.zh-CN.md)。
