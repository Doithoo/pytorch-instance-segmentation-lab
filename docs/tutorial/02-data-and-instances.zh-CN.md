# 数据与实例

[English](02-data-and-instances.md) | [文档导航](../README.zh-CN.md)

默认示例是 Penn-Fudan Pedestrian。下载脚本会在解压前校验官方压缩包；随后由 prepare 命令生成确定性的 manifest，不依赖目录遍历顺序。

## 准备 Penn-Fudan

```bash
uv run python scripts/download_data.py --data-dir data/raw --manifest-dir data/manifests
uv run instance-segment prepare-data --data-dir data/raw --manifest-dir data/manifests
uv run instance-segment verify-data --data-dir data/raw --manifest-dir data/manifests
uv run instance-segment inspect-data --data-dir data/raw --manifest-dir data/manifests --split train
uv run python scripts/preview_dataset.py --output artifacts/dataset-preview.png
```

提交的 protocol v2 manifest 包含 170 张图片，划分为 136 train、17 valid 和 17 test 行。划分在 Fudan 与 Penn 来源之间分层，并记录在 `data/manifests/dataset.yaml`。每行包含尺寸、实例数和图片/mask 的 SHA-256。准备完成后不要手改 CSV；它的 identity 属于 checkpoint 兼容性契约。

## 读取实例 ID mask

Penn-Fudan mask 使用像素值 `0` 表示背景，每个正整数代表一个对象。provider 为每个正数 ID 创建一个 bool mask，计算半开区间 `xyxy` box 和像素面积，并赋予前景 label `1`（`person`）。稀疏 ID 不会产生空对象，相互接触的 ID 仍保持为独立实例。

几何变换会同时作用于图片和 mask；resize 或水平翻转后会重新从 mask 计算 box 和 area，避免旧 box 静默失效。由于图片尺寸和实例数量都可变，collate 会保留 list。

## COCO 替代路径

使用 polygon 或 RLE 标注时，把三个 split 的 COCO instance JSON 放在同一个数据根目录下：

```bash
uv run instance-segment prepare-coco \
  --data-dir data/coco --manifest-dir data/coco-manifests \
  --train-annotations annotations/instances_train.json \
  --valid-annotations annotations/instances_valid.json \
  --test-annotations annotations/instances_test.json
uv run instance-segment verify-data --data-dir data/coco --manifest-dir data/coco-manifests
```

类别 ID 会排序后重新映射为从 `1` 开始的连续模型 label，映射持久化在 `dataset.yaml`。支持 polygon 列表、压缩/非压缩 RLE、多类别对象、crowd 标记和空图片。标注及图片路径必须位于 `--data-dir` 内。

CSV 列、COCO 一致性规则和 target dtype 见[数据格式参考](../reference/dataset-format.zh-CN.md)。
