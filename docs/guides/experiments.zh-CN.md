# 实验管理

每个独立实验使用新的 `run.name`。运行会记录解析配置、split hash、结构化环境、git commit/dirty 状态、`uv.lock` hash、JSONL 生命周期事件、每轮 loss/验证指标、学习率、耗时、CUDA 峰值显存和 checkpoint。

继续同一训练轨迹时应从 `last.pt` 恢复，而不是 `best.pt`。写入前会检查不可变的数据/模型/优化字段和 metrics 尾部。明确分叉实验时可修改 run/output、总 epoch、device 和 worker，同时保留 lineage。

模型选择指标可运行 `instance-segment compare-runs RUN_A RUN_B --metric valid_mask_map`；每个运行都有 evaluation 报告时可比较 `mask_map`。默认拒绝 split hash、指标协议、score floor、mask 阈值或类别数不同的运行。`--allow-incompatible` 只用于诊断，不应用于公开排名。

调参期间不要反复查看 test。使用 valid 选模，只在结束后评估一次固定 test；小数据集应发布置信区间，所有公开数值都应附 checkpoint/source hash。
