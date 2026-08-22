# Mask R-CNN 流程

`list[CHW image]` 进入 torchvision transform 和 ResNet-50 FPN；RPN 提出候选区域，ROI heads 分类/细化 box，mask head 为保留实例输出概率图。训练消费 target list 并返回 losses；推理只消费 images 并返回可变数量的 predictions。
