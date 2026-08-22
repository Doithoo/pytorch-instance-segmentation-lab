# Instance Target

[English](instance-target.md) | [文档导航](../README.zh-CN.md)

`InstanceTarget` 是所有数据 provider 与 torchvision 模型之间的边界。它是只包含六个字段的 typed dictionary：

```python
{
    "boxes": float32[N, 4],
    "labels": int64[N],
    "masks": bool[N, H, W],
    "image_id": int64[1],
    "area": float32[N],
    "iscrowd": int64[N],
}
```

box 使用半开区间 `xyxy`：`[x1, y1, x2, y2]`，并满足 `x1 < x2`、`y1 < y2`。`area` 是每个二值 mask 的 true 像素数。所有描述实例的字段共享同一个 `N`；几何变换必须同步更新每个对齐字段。

label schema 从 background ID `0` 开始连续编号。provider 的源类别 ID 可以不同，但 prepare 阶段会映射为模型 schema 并持久化该映射。`num_classes` 必须等于 schema 类别总数，其中包括 background。

当 provider 支持空图片时，空 target 也是合法的。它仍包含相同字段和 dtype，只是实例 tensor 长度为零，`image_id` 仍是单元素 tensor。这对模型 smoke test 和包含负样本的 COCO 数据都很重要。

collate 返回 `list[image]` 和 `list[target]`，而不是 padding 后的 batch tensor。这符合 torchvision detection 模型，也不会为不同尺寸或实例数的图片伪造实例。

实现 provider 时可参考 [GitHub examples](https://github.com/Doithoo/pytorch-instance-segmentation-lab/tree/main/examples) 中的 `01_instance_target.py`、`02_mask_to_instances.py` 和 `src/instance_segmenter/data/schema.py`。
