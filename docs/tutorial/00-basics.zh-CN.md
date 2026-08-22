# 实例分割基础

[English](00-basics.md) | [文档导航](../README.zh-CN.md)

目标检测返回数量可变的 box 列表；语义分割为每个像素返回类别。实例分割结合了两者：每个对象都保留为独立记录，包含类别、box、二值 mask 和置信度。

## 为什么需要实例

两个人可以接触或重叠，但训练目标仍然是两个实例。只有一张 semantic foreground mask 无法表达这个区别，除非额外约定 instance ID。本项目从 mask 中读取实例 ID，将每个正数 ID 转换为一个对象，并在变换、batch、训练、指标和可视化中保持对象独立。

## 项目契约

对于高度为 `H`、宽度为 `W` 的图片，provider 返回：

```text
image:  float32 [3, H, W]，取值范围 [0, 1]
target:
  boxes:    float32 [N, 4]，半开区间 xyxy
  labels:   int64   [N]
  masks:    bool    [N, H, W]
  image_id: int64   [1]
  area:     float32 [N]，像素面积
  iscrowd:  int64   [N]
```

所有带 `N` 维度的字段描述同一组实例，必须保持对齐。`labels` 只包含前景 ID；共享 schema 中的 `0` 保留给 background。COCO provider 支持 `N=0` 的空图片。

## 完整数据流

```text
图片 + instance-ID mask
        |
        v
独立的 boxes、labels、bool masks
        |
        v
训练阶段的 Mask R-CNN losses
        |
        v
推理阶段的 boxes、labels、scores、概率 masks
        |
        v
COCO AP、错误表、阈值化 PNG mask、overlay
```

可以先运行 [GitHub examples 目录](https://github.com/Doithoo/pytorch-instance-segmentation-lab/tree/main/examples) 中的可执行示例，再阅读[数据与实例](02-data-and-instances.zh-CN.md)，了解 Penn-Fudan 文件如何变成上述 target。
