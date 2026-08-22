# 配置参考

`run.name`、`seed` 和 `output_dir` 标识 artifact 目录。`data.provider` 可选 `pennfudan` 或 `coco`；可信插件仍可使用 `data.factory=module:callable`。数据根目录、manifest、resize、loader、增强和 split limit 都是显式配置。

`model.name`、权重策略、类别数和构造器 `params` 定义模型。向 torchvision 传参数前可运行 `instance-segment model-info NAME`。

训练配置包括 optimizer/scheduler、AMP、梯度裁剪、评估频率和三个不同阈值：

- `evaluation_score_floor`：进入置信度排序 AP 前的可选下限，通常为 `0.0`。
- `score_threshold`：推理展示和逐图错误阈值。
- `mask_threshold`：mask 概率二值化阈值。

`training.best_metric` 固定为 `mask_map`。即使最后一轮不满足 `evaluate_every`，也一定执行评估。`device` 可选 `auto`、`cpu`、`cuda` 或 `mps`。

YAML 只能包含已知字段，`--set section.field VALUE` 优先级最高。`show-config` 展示解析值及来源，`init-config` 可从源码或安装后的 wheel 复制模板。
