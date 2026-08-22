# 使用自己的数据

使用一一对应的 image 与 instance-ID mask 文件。0 保留给背景，每个对象使用不同正整数。按 `examples/extensions/my_dataset.py` 的关键字参数实现 `module:build_dataset`，返回文档规定 dtype 的 image/target，再在 YAML 设置 `data.factory`。长训练前先测试水平翻转和 resize。
