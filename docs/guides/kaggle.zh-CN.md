# 在 Kaggle 上完整训练

Kaggle GPU 是强制的参考训练路径。runner 内嵌精确源码归档和提交的 136/17/17 manifests，不需要挂载 Kaggle Dataset，也不需要本地 CUDA。

## 提交

```bash
uv tool install kaggle
kaggle auth login
uv run python scripts/build_kaggle_runner.py --check
kaggle kernels push -p docs/recorded-run/kaggle
kaggle kernels status <username>/pytorch-instance-segmentation-lab-penn-fudan-gpu
```

只修改 `kernel-metadata.json` 中的 `id`。必须开启 Internet，任务会下载官方 Penn-Fudan archive 和 COCO 权重。请求 T4 或更新 GPU；P100 可能与当前 PyTorch CUDA kernel 不兼容。即使 Kaggle 显示两张 GPU，项目只使用 `cuda:0`。

## runner 做什么

runner 输出 JSON `started`、`running`、`completed` 事件，慢阶段每 60 秒发送 heartbeat。流程为：GPU preflight、checksum 下载、manifest 校验、预览、真实 Mask R-CNN dry-run、完整 20 epoch 训练、用 `best.pt` 进行一次 test 评估、一个 test 图片预测和 summary。

未完成的运行不是参考结果。成功运行必须记录 `completed_epochs: 20`、固定 split counts、按验证集选择的 `best_epoch` 和 test 指标。

## 下载结果

```bash
kaggle kernels output <username>/pytorch-instance-segmentation-lab-penn-fudan-gpu \
  --file-pattern 'artifacts/.*' -p kaggle-output
```

查看 `reference-maskrcnn/best.pt`、`last.pt`、`metrics.csv`、`evaluation/`、predictions、`dataset-preview.png` 和 `kaggle-run-summary.json`。失败时查看 `kaggle-run-failure.json`，不要把其中的部分结果称为完整训练记录。
