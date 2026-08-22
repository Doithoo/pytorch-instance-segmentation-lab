# 排错

- `CUDA was requested`：本地使用 CPU dry-run，或 Kaggle 请求 T4/newer。
- manifest hash mismatch：重新下载/准备，不要手改 CSV 行。
- 没有预测：分别检查 score threshold 和 mask threshold。
- mask/box 漂移：检查 mask 的 nearest-neighbor resize 和 box 重算。
- checkpoint schema mismatch：使用同一 label schema 生成的 checkpoint。
