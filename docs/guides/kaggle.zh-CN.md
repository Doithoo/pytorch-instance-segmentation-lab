# 在 Kaggle 上完整训练

生成的 runner 是协议 v2 的可复现 GPU 路径。它内嵌精确 package 源码、参考配置和按来源分层的 manifests，不需要挂载 Kaggle Dataset。

```bash
uv run python scripts/build_kaggle_runner.py
cd docs/recorded-run/kaggle
kaggle kernels push -p .
kaggle kernels status Doithoo/pytorch-instance-segmentation-lab-penn-fudan-gpu
kaggle kernels output Doithoo/pytorch-instance-segmentation-lab-penn-fudan-gpu -p output
```

只修改 `kernel-metadata.json` 的 `id`。必须开启 Internet，因为任务会下载带 checksum 的 Penn-Fudan archive、COCO 初始化权重和缺失的指标依赖。GPU 选择 T4 或更新的兼容 NVIDIA 型号。

runner 输出 JSON started/running/completed 事件和 60 秒 heartbeat，依次执行 GPU preflight、源码下载校验、预览、真实 dry-run、20 epoch 训练/验证、best checkpoint 选取、一次 test 评估、推理、summary 写入，并清理运行时 data/project 目录，使未来 kernel output 只保留有用 artifact。

kernel version 2 早于最后的清理改进，因此其下载输出仍暴露原始运行目录。只下载有用文件可使用 `kaggle kernels output ... --file-pattern '^artifacts/'`。

协议 v2 只有在记录 20 epoch、当前 dataset identity `64bfbd3d...`、metric score floor `0.0`、有限 best epoch、test 报告和 checkpoint/source hash 后才算完成。将小型报告和代表图复制到 `docs/recorded-run`，大型可信 checkpoint 作为 Release/Kaggle asset 发布并附 SHA-256。不要覆盖归档的 `run_kaggle-v1.py`，也不要把协议 v1 指标改名为新基线。
