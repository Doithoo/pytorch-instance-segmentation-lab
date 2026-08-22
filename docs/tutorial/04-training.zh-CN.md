# 训练

先在本地执行一次真实参数更新：

```bash
uv run instance-segment train --config configs/learning_minimal.yaml --dry-run --device cpu
```

完整运行保存解析后的配置、环境、manifest hash、epoch loss、验证指标、`best.pt` 和 `last.pt`。只用验证集 `mask_map` 选择 `best.pt`；`train` 绝不读取 test split。

必须完成的 20 epoch GPU 参考流程见 [Kaggle 指南](../guides/kaggle.zh-CN.md)。
