# 添加模型

[English](adding-models.md) | [文档导航](../README.zh-CN.md)

模型可以加入 registry，也可以通过可信的外部 factory 提供。两种方式都必须遵守同一个 torchvision detection 契约。

## 契约

factory 接收 `num_classes`、`weights` 和 `params`。训练调用 `model(images, targets)`，要求返回有限的标量 loss 字典；评估调用 `model(images)`，要求每张图片返回一个字典，且 `boxes`、`labels`、`scores`、`masks` 的第一维对齐。mask 必须保持为独立实例，不能把相互接触的对象合并成 semantic mask。

## 内置模型清单

1. 在 `src/instance_segmenter/models/` 实现 factory，并保持构造参数显式。
2. 在 `models/registry.py` 加入 `ModelSpec`，填写稳定名称、描述、支持的权重策略、输入说明和参数说明。
3. 当 `num_classes` 不同于上游默认值时，同时替换 box 和 mask predictor。
4. 添加权重支持/拒绝、预测头 shape、空预测、真实训练 forward 和 checkpoint 恢复测试。
5. 添加或更新可运行配置，并同步两种语言的模型清单。
6. 运行 CPU smoke；如果模型声明支持 GPU，再执行真实的一 batch CUDA smoke。

## 外部 factory

在项目环境可 import 的模块中创建 factory，并配置：

```yaml
model:
  factory: my_project.models:build_segmenter
  name: external-segmenter
  weights: none
  num_classes: 2
  params:
    min_size: 128
```

可调用对象应接收 keyword 参数并返回 `torch.nn.Module`。外部 factory 会在训练、评估和推理过程中执行 Python 代码，因此属于可信代码输入。请把 factory 与测试放在一起，并记录权重下载和许可证信息。

不要只添加模型名称而不说明 predictor head、权重策略、图片尺寸假设和支持的 torchvision 版本。仅添加依赖和 registry 条目还不能形成可用的模型契约。
