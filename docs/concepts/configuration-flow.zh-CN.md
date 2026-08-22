# 配置流

[English](configuration-flow.md) | [文档导航](../README.zh-CN.md)

配置采用严格模式，避免拼写错误静默改变实验。

## 优先级

loader 先使用 dataclass 默认值，再合并 YAML，最后按顺序应用重复的 `--set KEY VALUE`：

```text
默认 dataclass < YAML < CLI --set
```

`train` 的 `--device` 是对顶层 device 字段的命令行替换。`show-config` 会输出解析后的 YAML 以及 `sources` 映射，说明每个叶子字段来自默认值、YAML 还是 CLI。

```bash
uv run instance-segment show-config --config configs/learning_minimal.yaml \
  --set training.epochs 3 --set model.params.min_size 160
```

YAML 和 CLI 覆盖只能使用已知的顶层字段和 section 字段。`model.params` 是有意保留的模型构造参数扩展口，但这些参数仍会传给可信 model factory。

## 记录内容

完整运行会把解析配置写入 `config.yaml`，并嵌入两个 checkpoint。同时记录 manifest split hash、环境、源码状态和 lockfile hash，因此不需要猜测当时启用了哪些默认值。

## 重要 section

- `run`：`name`、`seed`、`output_dir`；运行目录是 `output_dir/name`。
- `data`：provider/factory、数据根目录、manifest 目录、图片 resize、loader 设置、增强概率和 split limit。
- `model`：注册名称或可信 factory、权重策略、类别数和构造参数。
- `training`：优化器/scheduler、loss 优化值、AMP、裁剪、验证频率和阈值。
- `device`：`auto`、`cpu`、`cuda` 或 `mps`。

`training.best_metric` 刻意固定为 `mask_map`。`training.evaluation_score_floor` 控制哪些预测进入 AP；`training.score_threshold` 控制可读的错误分析和推理展示；`training.mask_threshold` 将 mask 概率二值化。三者不能混用。

默认值和校验规则见[配置参考](../reference/config-reference.zh-CN.md)。
