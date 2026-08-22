# 数据格式

[English](dataset-format.md) | [文档导航](../README.zh-CN.md)

prepare 命令会创建一个 manifest 目录，其中包含数据元信息和每个 split 一个 CSV。dataset identity 由准备后的元数据和 split hash 得出；训练与评估会用它拒绝被修改的数据。

## Penn-Fudan 目录

```text
<data-dir>/PennFudanPed/PNGImages/<id>.png
<data-dir>/PennFudanPed/PedMasks/<id>_mask.png
```

mask 值 `0` 是背景，每个正整数是一个实例 ID。protocol v2 使用按来源分层的 SHA-256 排序生成 136/17/17 的 train/valid/test。划分和来源见[Penn-Fudan 参考](penn-fudan.zh-CN.md)。

## Manifest 行

准备后的 CSV 包含 provider 所需的路径和完整性数据，应视为生成文件。每行标识一张图片、实例 mask 或 annotation 记录、尺寸、实例数和源 hash。`dataset.yaml` 保存格式版本、provider、label schema、split hash、数量和 dataset identity；`source.yaml` 在适用时保存下载来源。

## COCO Instance JSON

`prepare-coco` 要求每个 JSON 包含 `images`、`annotations` 和 `categories`。三个文件的类别定义必须一致。`segmentation` 支持 polygon 列表、压缩和非压缩 RLE。由 image 记录和 annotation 输入解析出的路径必须位于 `--data-dir` 内。

类别按原始 category ID 排序，并映射为从 `1` 开始的连续模型 ID；`0` 保留给 background。映射会写入 `dataset.yaml`。crowd 值和无 annotation 的图片会保留。准备阶段检查标注/图片尺寸并记录源 hash，后续由 verify 命令校验。

```bash
uv run instance-segment prepare-coco \
  --data-dir data/coco --manifest-dir data/coco-manifests \
  --train-annotations annotations/train.json \
  --valid-annotations annotations/valid.json \
  --test-annotations annotations/test.json
```

## 运行时 target

两个内置 provider 都返回 `[0,1]` 范围内的 float32 图片和精确的 target 契约：

```text
boxes    float32 [N,4]
labels   int64   [N]
masks    bool    [N,H,W]
image_id int64   [1]
area     float32 [N]
iscrowd  int64   [N]
```

所有实例字段共享 `N`；COCO 支持 `N=0`。每次 resize 或水平翻转后，box 和 area 都从变换后的 mask 重新计算。长时间训练前请使用 `validate_instance_target` 和扩展示例校验自定义 provider。
