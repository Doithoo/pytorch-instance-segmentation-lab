# Penn-Fudan Pedestrian

官方 archive 包含来自 Fudan 与 Penn 两个来源的 170 张街景图片和行人 instance-ID mask。下载脚本会先验证 archive SHA-256，再安全解压。

协议 v2 按 ID 的非数字来源前缀分组，以 seed 42 对每组做稳定 hash 排序，并填充固定的 136 train、17 valid、17 test：

| Split | Fudan | Penn | 总计 |
|---|---:|---:|---:|
| train | 59 | 77 | 136 |
| valid | 7 | 10 | 17 |
| test | 8 | 9 | 17 |

每条 manifest 保存 image/mask 路径、尺寸、实例数和两个文件 hash。`dataset.yaml` 保存格式版本、策略、seed、split hash、label schema、范围与 dataset identity。训练和 checkpoint 评估都会拒绝被修改的 manifest。
