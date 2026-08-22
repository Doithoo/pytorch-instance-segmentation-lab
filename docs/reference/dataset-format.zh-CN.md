# 数据格式

内置 provider 读取 `PennFudanPed/PNGImages/<id>.png` 与 `PennFudanPed/PedMasks/<id>_mask.png`。mask 值 0 是背景，每个正整数是一个 person 实例。target 使用 float32 半开 `xyxy` box 和 bool mask。不要把相邻的人共享同一个值的语义类别图当作实例标注。
