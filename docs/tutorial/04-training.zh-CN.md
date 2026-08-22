# 训练

```bash
uv run instance-segment train --config configs/learning_minimal.yaml --dry-run --device cpu
uv run instance-segment train --config configs/reference_maskrcnn.yaml --device cuda
uv run instance-segment train --config configs/reference_maskrcnn.yaml --resume artifacts/reference-maskrcnn/last.pt
```

dry-run 会执行一次真实 optimizer 更新，但不写运行目录。完整训练会校验 split hash，记录解析配置、环境、git/lock provenance，并写入 `metrics.csv`、`events.jsonl`、`best.pt` 和 `last.pt`。指标包含分项 loss、验证 bbox/mask AP 与 AR、学习率、每轮耗时和 CUDA 峰值显存。

`best.pt` 由验证 mask AP 选出，最后一轮始终执行评估。训练 orchestrator 不会加载 test。恢复训练会还原 optimizer、scheduler 和 RNG，并拒绝不可变配置变化、split hash 不匹配或 metrics 尾部不一致。

已完成协议 v2 参考训练，复现步骤见 [Kaggle 指南](../guides/kaggle.zh-CN.md)。
