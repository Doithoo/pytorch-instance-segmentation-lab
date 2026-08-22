# 指标

`mask_map` 是 mask IoU 0.50:0.95 上的 COCO 风格 AP，并用于选择 `best.pt`；`bbox_map` 是对应的 box 指标。AP 依赖预测置信度排序，因此协议 v2 会把不低于 `training.evaluation_score_floor`（默认 `0.0`）的全部输出交给 `torchmetrics.MeanAveragePrecision`。

`training.score_threshold` 与指标分离，只控制推理文件、overlay 和逐图匹配/错误统计，不影响 AP。`training.mask_threshold` 把 mask 概率二值化。

报告包含 AP/AP50/AP75、AR@100、逐类 mask/bbox AP、图片/目标/预测数量和错误数量。evaluation JSON 会记录指标后端、协议、floor、阈值、dataset identity 和 split hash。只有这些字段一致的运行才应比较。

项目刻意不使用 pixel accuracy，因为大量背景像素会掩盖实例失败。测试集较小时，应在点估计之外发布按图片 bootstrap 的置信区间或多 seed 结果。
