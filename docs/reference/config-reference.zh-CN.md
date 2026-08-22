# 配置参考

[English](config-reference.md) | [文档导航](../README.zh-CN.md)

loader 接受与 `AppConfig` 对应的 YAML 字段，未知字段会被拒绝。路径相对于当前工作目录解释。

| Section | 字段 | 默认值 / 说明 |
| --- | --- | --- |
| `run` | `name`、`seed`、`output_dir` | `learning-minimal`、`42`、`artifacts`；运行路径为 `output_dir/name` |
| `data` | `provider`、`factory`、`root`、`manifest_dir` | provider 为 `pennfudan` 或 `coco`；factory 是可信的 `module:callable` |
| `data` | `image_size`、`batch_size`、`num_workers` | 默认 `[128,128]`、batch `1`、worker `0`；`null` 保留源尺寸 |
| `data` | `horizontal_flip`、`train_limit`、`valid_limit`、`test_limit` | 翻转概率 `0.0`；limit 为正数或 `null`（全部行） |
| `model` | `name`、`factory`、`weights`、`num_classes`、`params` | 两个类别（含 background）；params 会传给 factory |
| `training` | `epochs`、`optimizer`、`learning_rate`、`momentum`、`weight_decay` | SGD、2 epoch、`0.005`、`0.9`、`0.0005` |
| `training` | `scheduler`、`step_size`、`gamma` | `none`、`6`、`0.1`；scheduler 可设为 `step` |
| `training` | `amp`、`grad_clip_norm`、`evaluate_every` | `auto`、不裁剪、每轮评估 |
| `training` | `best_metric` | 固定为 `mask_map` |
| `training` | `evaluation_score_floor` | `0.0`；AP 的可选输入 floor |
| `training` | `score_threshold`、`mask_threshold` | `0.5`、`0.5`；展示/错误分析和二值 mask 阈值 |
| 根级 | `device` | `auto`；可选 `auto`、`cpu`、`cuda`、`mps` |

## 模型参数

`model.params` 是唯一开放 mapping。内置 factory 通常接受 `min_size` 和 `max_size`；运行 `instance-segment model-info NAME` 查看模型说明。权重名称由所选模型校验，`coco_v1` 不能与 `imagenet_v2` 互换。

## CLI 覆盖

每个覆盖使用一组 `--set KEY VALUE`。值按 YAML 解析，因此 `false`、`null`、数字、列表和字符串会保留对应类型：

```bash
uv run instance-segment show-config --config configs/learning_minimal.yaml \
  --set data.image_size null --set training.amp true --set run.name full-size
```

CLI 覆盖优先级最高。`show-config` 还会报告每个解析字段的来源。`train --device` 只覆盖当前运行设备。

## 可复现字段

做实验比较时，应明确固定 `run.seed`、provider、数据根目录内容、manifest identity、label schema、模型名称/权重/参数、图片尺寸策略、loader/增强设置、优化参数和所有阈值。trainer 会持久化解析配置和 split hash；恢复 checkpoint 时会拒绝不兼容的不可变字段。
