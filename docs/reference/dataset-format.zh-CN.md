# 数据格式

## Penn-Fudan 实例 ID

内置 Penn-Fudan provider 读取 `PennFudanPed/PNGImages/<id>.png` 和 `PennFudanPed/PedMasks/<id>_mask.png`。mask 值 0 是背景，每个正整数表示一个独立 person。准备阶段会检查尺寸和源码 SHA-256，并写入按来源分层的固定 manifest。

## COCO instance JSON

COCO 准备流程支持 polygon、压缩 RLE 和未压缩 RLE。类别按原始 category ID 排序并映射到从 1 开始的连续模型 label，映射保存在 `dataset.yaml`。没有 annotation 的图片也是合法样本。

```bash
uv run instance-segment prepare-coco \
  --data-dir data/coco \
  --manifest-dir data/coco-manifests \
  --train-annotations annotations/instances_train.json \
  --valid-annotations annotations/instances_valid.json \
  --test-annotations annotations/instances_test.json
```

annotation 和图片路径必须位于 `data-dir` 内，三个文件的类别定义必须一致。准备阶段记录 annotation/image hash 与尺寸，训练前可用 `verify-data` 校验。

## 运行时 target

两个 provider 都返回 `[0,1]` 范围 float32 `image[C,H,W]`，以及 float32 半开区间 `boxes[N,4]`、int64 `labels[N]`、bool `masks[N,H,W]`、int64 `image_id[1]`、float32 mask 像素 `area[N]` 和 int64 `iscrowd[N]`。所有实例字段共享同一个 `N`；COCO 支持 `N=0`。
