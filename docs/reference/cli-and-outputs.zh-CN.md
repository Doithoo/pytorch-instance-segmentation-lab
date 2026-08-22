# CLI 与输出

[English](cli-and-outputs.md) | [文档导航](../README.zh-CN.md)

请从仓库根目录执行命令。`uv run` 会确保使用项目环境。

## 命令速查

| 命令 | 作用 | 主要结果 |
| --- | --- | --- |
| `show-config` | 合并默认值、YAML 和 CLI 覆盖 | stdout 上的解析配置 |
| `init-config` | 列出或复制已安装模板 | 一个 YAML 文件 |
| `doctor` | 检查 CPU/CUDA/MPS | stdout 上的设备报告 |
| `prepare-data` | 准备 Penn-Fudan manifest | `dataset.yaml`、`source.yaml` 和 split CSV |
| `prepare-coco` | 从三个 COCO instance JSON 准备 manifest | 相同的 manifest 契约 |
| `verify-data` | 检查源文件、尺寸和 hash | 校验摘要 |
| `inspect-data` | 汇总一个 prepared split | stdout 上的 YAML 摘要 |
| `list-datasets` | 列出注册的数据 provider | 名称和描述 |
| `list-models` | 列出注册模型和权重策略 | stdout 上的模型清单 |
| `model-info NAME` | 查看模型说明和参数 | 教学元数据 |
| `train --dry-run` | 执行一次真实更新 | stdout 诊断，不创建运行目录 |
| `train` | 训练并验证 | 一个运行目录 |
| `evaluate` | 在一个 split 上评估 checkpoint | `evaluation/` 报告 |
| `predict` | 对一张图片预测实例 | JSON、mask 和 overlay |
| `compare-runs` | 排序兼容运行 | stdout 上的制表符排名 |

可使用 `uv run instance-segment --help` 以及各子命令的 `--help` 查看完整 parser 契约。阈值参数必须是 `0` 到 `1` 之间的有限数值。

## 首次命令

```bash
uv run instance-segment show-config --config configs/learning_minimal.yaml
uv run instance-segment doctor --device auto
uv run instance-segment list-datasets
uv run instance-segment list-models
uv run instance-segment model-info maskrcnn_resnet50_fpn
```

配置命令支持重复传入 `--set KEY VALUE`。优先级为默认值、YAML、CLI。例如：

```bash
uv run instance-segment train --config configs/learning_minimal.yaml \
  --set run.name first-cpu-run --set data.train_limit 2 --device cpu
```

## 运行目录

一次正常训练会写入 `artifacts/<run.name>/`：

```text
config.yaml              解析后的配置
manifest-hashes.yaml     本次运行使用的 hash
environment.json         Python、Torch、设备、git 和 lock provenance
events.jsonl             生命周期和 epoch 事件
metrics.csv              loss、验证指标和运行时数据
best.pt                  按验证 mask_map 选择的 checkpoint
last.pt                  最近完成 epoch 的 checkpoint
evaluation/              evaluate 生成的可选报告
```

`train --dry-run` 会返回图片尺寸、target 数量、分项 loss 和 `dry-run OK`，并且刻意不创建运行目录。正常运行不会覆盖已有目录；恢复训练必须使用合法的 `--resume`。

## 评估输出

评估 `artifacts/run/best.pt` 时，默认在 checkpoint 旁写入：

```text
artifacts/run/evaluation/
  evaluation.json
  per_class.csv
  per_image.csv
  visualizations/          只有传入 --plot 才生成
    worst-*-ground-truth.png
    worst-*-prediction.png
```

JSON 会记录 split、指标后端和协议、全部阈值、类别名称、dataset identity、split hash、数量和指标值。替换已有报告目录必须显式传 `--overwrite`。

```bash
uv run instance-segment evaluate --checkpoint artifacts/first-cpu-run/best.pt \
  --split test --device cpu --output-dir artifacts/first-cpu-run/test-evaluation --plot
```

## 单图预测输出

`predict` 需要图片和输出目录。它会为每个保留实例写一个阈值化灰度 PNG、`instances.json` 索引和 `overlay.png`：

```bash
uv run instance-segment predict --checkpoint artifacts/first-cpu-run/best.pt \
  --image data/raw/PennFudanPed/PNGImages/FudanPed00028.png \
  --output artifacts/first-cpu-run/prediction --device cpu
```

`instances.json` 包含源图片、阈值、类别 ID/名称、置信度、半开区间 `box_xyxy` 和相对 mask 路径。单图预测不会读取或校验数据集 manifest；加载前请自行验证 checkpoint hash。

## 运行比较

使用 `valid_mask_map` 比较训练时按 valid 选模的运行；使用 `mask_map` 比较已有兼容评估报告的运行：

```bash
uv run instance-segment compare-runs artifacts/run-a artifacts/run-b --metric valid_mask_map
uv run instance-segment compare-runs artifacts/run-a artifacts/run-b --metric mask_map --allow-incompatible
```

比较默认拒绝 dataset identity、split hash、类别数、指标协议、score floor 或 mask threshold 不一致的运行。`--allow-incompatible` 只用于诊断，不应出现在公开排名中。
