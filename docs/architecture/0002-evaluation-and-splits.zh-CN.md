# ADR 0002：评估协议与按来源分层的数据划分

状态：已接受

## 背景

首个版本在调用 `MeanAveragePrecision` 前删除 score 小于 0.5 的预测，这会改变按置信度排序的 precision-recall 曲线，不能直接与标准 COCO AP 比较。原先按文件名连续切分 Penn-Fudan，也导致全部 Fudan 图片只进入 train，而 valid/test 只包含连续 Penn 图片。

## 决策

协议 v2 分离三类阈值：

- `training.evaluation_score_floor` 默认 `0.0`，只作为可选的指标输入下限。
- `training.score_threshold` 默认 `0.5`，用于推理展示和逐图错误分析。
- `training.mask_threshold` 用于把 mask logit 二值化。

Penn-Fudan manifest 使用 `source-stratified-sha256-v2`、seed 42。Fudan/Penn 来源组分别按稳定 hash 排序并分配到固定 136/17/17 总量。train、valid、test 中 Fudan/Penn 数量分别为 59/77、7/10、8/9。

评估只执行一次模型遍历，同一份 CPU prediction 同时生成 bbox/mask 指标、逐图报告和有限数量的最差样本图。

恢复训练和带数据集的 checkpoint 评估必须匹配当前 manifest hash。除非明确覆盖，实验对比会拒绝 split hash、指标协议、score floor、mask 阈值或类别数不同的运行。

## 影响

0.1.0 的 T4 指标仍作为历史执行记录保留，但标记为 legacy，不能改名为协议 v2 结果。之后 Kaggle kernel version 2 已完成所需替代训练，报告与精确提交 runner 保存在 `docs/recorded-run`。
