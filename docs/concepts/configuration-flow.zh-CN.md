# 配置流

dataclass 提供默认值；YAML 只能覆盖已知字段；`--set key value` 覆盖 YAML。解析后的配置写入运行目录和 checkpoint；Kaggle 只覆盖 device、路径、workers 和 AMP，并保存实际解析配置。
