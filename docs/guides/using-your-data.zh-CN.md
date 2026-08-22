# 使用自己的数据

[English](using-your-data.md) | [文档导航](../README.zh-CN.md)

如果标注可以表达为实例 polygon 或 RLE，优先使用内置 COCO 路径。它会保留标准元数据，也能避免在尚未理解数据契约前就编写 provider。

## COCO 路径

把图片和三个 split 的 instance JSON 放在同一个数据根目录下。使用显式 annotation 路径运行 `prepare-coco`，随后校验并检查每个 split。三个 JSON 的类别定义必须一致，图片和标注路径必须位于数据根目录内。

COCO 准备流程支持 polygon 列表、压缩/非压缩 RLE、多前景类别、`iscrowd` 和无 annotation 的图片。源 category ID 会映射成连续模型 ID，并持久化在 `dataset.yaml`。将 `model.num_classes` 设置为最终 schema 数量，其中包括 background。

## Penn-Fudan 风格 mask

索引 PNG 布局使用 `0` 表示背景，并为每个对象使用不同的正整数。不要让所有前景像素共用一个值，否则实例边界会丢失。每个正区域都应有非零面积，且不能超出图片尺寸。

## 自定义 provider

只有 COCO 和 Penn-Fudan 都不适用时才使用 `data.factory=module:callable`。callable 会接收：

```text
manifest_dir, split, data_dir, training,
horizontal_flip, image_size, limit
```

它必须返回 `torch.utils.data.Dataset`，产出 float32 `CHW` 图片和精确 dtype 的 `InstanceTarget`。可以从 [GitHub 的 `my_dataset.py` 扩展示例](https://github.com/Doithoo/pytorch-instance-segmentation-lab/blob/main/examples/extensions/my_dataset.py) 开始。自定义数据集仍需要包含 `dataset.yaml` 和带 hash 的 split CSV 的 manifest 目录，因为 checkpoint 校验和运行比较依赖这些 identity。

## 长时间训练前

1. 复制或重新生成文件后运行 `verify-data`。
2. 检查 train、valid、test 摘要，并生成代表性 overlay。
3. 覆盖空图片、多类别、接触实例和最大尺寸图片。
4. 在 synthetic 测试中强制水平翻转和 resize，确认 box 与 mask 对齐。
5. 使用 `train --dry-run` 执行真实的一 batch 更新。

factory 属于可信 Python 代码。下载的标注和图片不要提交到版本库，并记录来源、许可证和 hash；不要让 checkpoint 依赖未记录的本地路径。
