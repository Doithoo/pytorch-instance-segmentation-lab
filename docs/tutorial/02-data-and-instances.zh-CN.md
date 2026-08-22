# 数据与实例

```bash
uv run python scripts/download_data.py --data-dir data/raw --manifest-dir data/manifests
uv run instance-segment prepare-data
uv run instance-segment verify-data
uv run python scripts/preview_dataset.py --output artifacts/dataset-preview.png
```

提交的 manifest 按确定顺序将 170 个文件分成 `136/17/17`。训练前会校验图片和 indexed mask 的 SHA-256。`PedMasks` 中每个正像素 ID 都变成一个 `[H,W]` 独立 bool mask。
