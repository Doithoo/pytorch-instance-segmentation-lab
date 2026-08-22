# Checkpoint Schema

格式版本 1 保存模型/optimizer/scheduler state、完成 epoch、最佳验证指标/epoch、label schema、解析配置、manifest hash、Python/Torch 版本和 RNG state。加载会拒绝不兼容的格式、模型名、label schema 或 tensor shape。
