# Mask R-CNN

backbone 生成 FPN 特征；RPN 提出候选框；ROI box head 分类并细化候选框；mask head 为保留的 ROI 预测二值 mask。`train()` 返回五项 loss，`eval()` 返回 `boxes`、`labels`、`scores` 和 mask 概率。项目在类别数改变时同时替换 box 与 mask predictor。
