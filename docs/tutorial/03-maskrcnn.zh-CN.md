# Mask R-CNN

[English](03-maskrcnn.md) | [文档导航](../README.zh-CN.md)

项目使用 torchvision detection API，不重复实现 RPN、ROIAlign、NMS 或 COCO AP。学习重点放在数据契约、模型接线、实验协议和可检查的输出上。

## 两条 forward 路径

训练时，一个 batch 是 float32 `CHW` 图片 list 和对应的 target list。torchvision Mask R-CNN 返回包含 classifier、box regression、mask、objectness 和 RPN box loss 的字典。trainer 汇总有限的分项 loss 并执行 optimizer update。

评估时，模型只接收图片，并为每张图片返回一个字典：

```text
boxes:  [N, 4] float 坐标
labels: [N]    类别 ID
scores: [N]    置信度排序
masks:  [N, 1, H, W] mask 概率
```

推理层会为展示应用 score 过滤、将 mask 概率阈值化，并保存独立实例文件。指标路径可以保留全部预测，从而保持置信度排序 AP 的意义。

## 模型构造

使用 `list-models` 查看可用名称和权重策略。当 `num_classes` 改变时，项目会同时替换 box predictor 和 mask predictor，使输出 head 与持久化的 label schema 一致。`model.params` 会转发模型对应的构造参数，例如 `min_size` 和 `max_size`。

模型选择见[模型选择](../guides/choosing-models.zh-CN.md)，权重行为见[模型清单](../reference/model-zoo.zh-CN.md)。[Mask R-CNN 流程](../concepts/maskrcnn-flow.zh-CN.md)会跟踪 backbone、RPN、ROI heads 和 mask head 的 tensor 流向。
