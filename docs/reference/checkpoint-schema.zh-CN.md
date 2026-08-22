# Checkpoint 格式

格式版本 1 保存 model、optimizer、scheduler 状态，完成/最佳 epoch、最佳验证指标、label schema、解析配置、split hash、Python/Torch 版本，以及 Python、NumPy、Torch 和可选 CUDA RNG 状态。写入使用临时文件和原子替换。

恢复训练会校验模型名、tensor shape、label schema、精确 manifest hash 和不可变配置。只允许修改 run 名称/输出目录、目标 epoch 数、device 和 worker 数。已有 `metrics.csv` 必须结束于 checkpoint epoch；恢复到新目录时会创建合法表头，并在 `events.jsonl` 与 `environment.json` 中记录来源。

依赖数据集的评估也会在推理前检查 checkpoint split hash。单图推理不读取数据集，使用者必须自行验证发布的 checkpoint SHA-256。

安全提示：恢复 checkpoint 为读取 optimizer 和 RNG 状态，需要执行 `torch.load(..., weights_only=False)`，因此 pickle payload 可能执行代码。只能加载可信 checkpoint。部署分发建议另外发布 weights-only 文件和签名 metadata。
