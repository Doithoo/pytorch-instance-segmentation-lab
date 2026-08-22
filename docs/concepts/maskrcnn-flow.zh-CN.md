# Mask R-CNN 流程

[English](maskrcnn-flow.md) | [文档导航](../README.zh-CN.md)

运行时是面向 list 的 detection pipeline：

```text
list[float32 CHW images]
        |
        v
Torchvision transform 与 backbone
        |
        v
ResNet-50 或 MobileNetV3 特征
        |
        v
FPN（按配置） -> RPN proposals
        |
        v
ROI box head：类别与 box 修正
        |
        v
ROI mask head：每个保留实例一张 mask 概率图
```

RPN 与 ROI head 作用于 proposals，因此输出实例数不会由 batch 固定。训练时，target 中的 boxes、labels、masks、area 和 crowd flags 监督对应分支；评估时只传入图片 list，后处理返回字段对齐的结果。

项目会根据 `num_classes` 更换最终的 box 和 mask predictor。这也是 COCO 预训练 backbone/head 不能原样用于二类 Penn-Fudan 任务的原因。权重策略会明确哪些上游组件被初始化，详见[模型清单](../reference/model-zoo.zh-CN.md)。

resize 策略一部分属于 torchvision 模型构造参数（`min_size`/`max_size`），另一部分属于数据配置（当输入 target 本身需要 resize 时）。比较运行时应固定这些选择。推理输出会再将 mask 概率阈值化，并为每个实例写独立产物，而不是生成单张 semantic label 图。
