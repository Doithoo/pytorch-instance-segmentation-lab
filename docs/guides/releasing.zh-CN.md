# 发布指南

## GitHub Release

发布 workflow 会在 GitHub Release 发布时触发，也支持从 Actions 页面手动运行。它构建 wheel 和 sdist，使用 OIDC trusted publishing 发布到 PyPI，并将分发包上传到 GitHub Release。

## PyPI Trusted Publishing

首次发布前，请在 PyPI 为项目 `pytorch-instance-segmentation-lab` 创建 pending publisher：

- Owner：`Doithoo`
- Repository：`pytorch-instance-segmentation-lab`
- Workflow：`.github/workflows/publish.yml`
- Environment：`pypi`

GitHub workflow 已请求 `id-token: write` 并设置 `environment: pypi`。OIDC claims 必须与仓库、workflow 精确路径和 environment 完全一致。若收到 `422 invalid-publisher`，说明 PyPI 侧条目缺失或配置不一致。

保存 pending publisher 后重新运行失败的 workflow：

```bash
gh run rerun 32578729496 --failed
# 或从 GitHub Actions 手动运行 Publish workflow。
```

trusted publishing 可用时不要把 PyPI API token 放入仓库 secrets。

## Release Asset

协议 v2 基线应上传 wheel、sdist、model card、可信 `best.pt` 和 checksum 文件。上传前后都应验证：

```bash
sha256sum -c SHA256SUMS-v0.2.0.txt
```

checkpoint 是基于 pickle 的可信 PyTorch artifact。文件旁应提供 SHA-256，并保留包含用途和限制说明的 model card。
