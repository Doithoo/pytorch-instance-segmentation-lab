# 故障排查

- `CUDA was requested`：运行 `instance-segment doctor --device auto`，使用 CPU dry-run，或在 Kaggle 请求 T4/newer。
- `manifest hash mismatch`：不要手改 CSV；恢复提交版本或重新执行对应 prepare 命令，并开始新实验。
- `resume configuration changes immutable fields`：只修改 epochs/device/workers 后恢复原轨迹，或者不用 `--resume` 新建运行。
- `metrics schema is incompatible`：保持历史目录不变，恢复到新 run 目录；不要跨协议版本追加。
- `evaluation output already exists`：使用新目录，或明确传 `--overwrite`。
- `unknown config field`：用 `show-config` 检查；YAML 是严格模式。
- COCO 解码失败：检查 polygon/RLE 结构、尺寸、类别一致性，以及所有路径都位于 data root 内。
- 显存不足：减小 batch/min-max image size、使用 MobileNetV3、开启 CUDA AMP 或减少 worker，并记录用于对比的实际配置。
