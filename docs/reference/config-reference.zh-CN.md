# 配置参考

`run.name` 标识 artifact 目录。`data.factory` 可选，格式为 `module:callable`；否则使用 `data.provider=pennfudan`。`model.params` 转发支持的 torchvision 构造参数。`training.best_metric` 固定为 `mask_map`。`device` 可为 `auto`、`cpu`、`cuda`、`mps`。
