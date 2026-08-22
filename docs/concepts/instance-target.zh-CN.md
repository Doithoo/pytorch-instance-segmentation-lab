# 实例 Target

`target` 的第一维 `N` 对齐：`boxes[N,4]`、前景 `labels[N]`、`masks[N,H,W]`、`area[N]`、`iscrowd[N]` 和一个 `image_id[1]`。每次几何变换后都从 mask 重算 box；空 mask 与所有对齐字段一起过滤。
