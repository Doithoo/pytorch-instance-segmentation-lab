# 指标

`mask_map` 是 mask IoU 0.50 到 0.95 的 COCO 风格平均 AP，并用于选择 `best.pt`。`mask_map_50`、`mask_map_75`、`mask_mar_100` 提供更多细节；`bbox_*` 输出等价的 box 指标。项目不使用 pixel accuracy，因为背景占比高时它会掩盖实例轮廓失败。
