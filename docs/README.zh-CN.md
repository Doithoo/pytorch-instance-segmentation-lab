# 文档导航

这个仓库适合按阶段完成一次实验。先读[教程](tutorial/README.zh-CN.md)，需要精确查询字段、文件或输出时再查[参考](reference/config-reference.zh-CN.md)。

## 按目标进入

| 目标 | 从这里开始 | 接着阅读 |
| --- | --- | --- |
| 在 CPU 上跑通 Penn-Fudan | [环境](tutorial/01-environment.zh-CN.md) | [数据与实例](tutorial/02-data-and-instances.zh-CN.md)、[训练](tutorial/04-training.zh-CN.md) |
| 理解实例 target | [基础](tutorial/00-basics.zh-CN.md) | [Instance target](concepts/instance-target.zh-CN.md)、[数据格式](reference/dataset-format.zh-CN.md) |
| 配置可复现实验 | [配置流](concepts/configuration-flow.zh-CN.md) | [配置参考](reference/config-reference.zh-CN.md)、[实验管理](guides/experiments.zh-CN.md) |
| 评估或查看预测 | [评估与推理](tutorial/05-evaluation-and-inference.zh-CN.md) | [指标](reference/metrics.zh-CN.md)、[CLI 与输出](reference/cli-and-outputs.zh-CN.md) |
| 使用 COCO JSON 或自定义数据 | [使用自己的数据](guides/using-your-data.zh-CN.md) | [数据格式](reference/dataset-format.zh-CN.md) |
| 选择或添加模型 | [模型选择](guides/choosing-models.zh-CN.md) | [模型清单](reference/model-zoo.zh-CN.md)、[添加模型](guides/adding-models.zh-CN.md) |
| 复现已发布的 GPU 运行 | [Kaggle 指南](guides/kaggle.zh-CN.md) | [训练记录](recorded-run/README.zh-CN.md)、[ADR 0002](architecture/0002-evaluation-and-splits.zh-CN.md) |

中文页面使用同名的 `.zh-CN.md` 后缀。仓库根目录还包含 `mkdocs.yml`，可将这些页面发布成可浏览的网站。

## 完整流程

```text
download -> prepare -> verify -> inspect -> dry-run -> train -> evaluate -> compare/predict
```

每个阶段的职责不同：

1. 只下载源压缩包并校验 checksum。
2. 生成确定性的 train/valid/test manifest。
3. 在昂贵训练前校验源文件和 manifest hash。
4. 检查 split 并生成数据预览。
5. 在 CPU 上执行真实的 forward、loss、backward 和 optimizer update。
6. 训练时只用 valid 的 `mask_map` 选择 `best.pt`，训练器不会读取 test。
7. 对选出的 checkpoint 只评估一次，也可以比较兼容运行或预测单张图片。

## 稳定契约

- 图片是 `[0, 1]` 范围内的 float32 `CHW` tensor。
- target 包含长度对齐的 `boxes`、`labels`、`masks`、`image_id`、`area` 和 `iscrowd`。
- label `0` 是 background，前景类别 ID 连续且从 `1` 开始。
- manifest 及其 SHA-256 identity 属于 checkpoint 和评估契约。
- AP 保留模型的置信度排序；`evaluation_score_floor` 与展示用 `score_threshold` 相互独立。
- `.pt` checkpoint 属于可信代码输入。请校验 SHA-256，只加载可信文件。

## 仓库目录

```text
configs/        可运行 YAML 模板和字段说明
examples/       可执行的小型示例
scripts/        下载、预览、绘图和 Kaggle 构建任务
src/            按数据流组织的包实现
tests/          synthetic 与集成测试
docs/           教程、概念、指南、参考、ADR 和运行证据
```

[训练记录](recorded-run/README.zh-CN.md)只证明一个有明确边界的 Penn-Fudan 协议，不是通用 benchmark。[模型卡](recorded-run/MODEL_CARD.md)说明其用途和限制。
