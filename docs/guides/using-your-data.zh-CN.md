# 使用自己的数据

条件允许时优先使用内置 COCO 路径。把图片和 train/valid/test instance JSON 放在同一个数据根目录下，运行 `instance-segment prepare-coco`；格式见[数据参考](../reference/dataset-format.zh-CN.md)。它支持 polygon、RLE、多类别、crowd 和无实例图片。

若使用索引 PNG mask，0 保留给背景，每个对象使用不同正整数。Penn-Fudan 是内置实现示例。

只有两种格式都不适用时才设置 `data.factory=module:callable`。按 `examples/extensions/my_dataset.py` 返回 float32 CHW 图片和精确 dtype 的 `InstanceTarget`。factory 属于可信 Python 代码。manifest 目录仍需提供 `dataset.yaml` 和带 hash 的 train/valid/test CSV，才能保持 label 所有权、checkpoint 校验与实验对比可复现。

长训练前应校验 source hash、检查每个 split、覆盖空样本和多类别样本、强制执行 flip/resize，并完成真实单 batch 参数更新。
