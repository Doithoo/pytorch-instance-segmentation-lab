# 发布指南

## GitHub Release

`release-assets` workflow 会在 GitHub Release 发布时触发，也支持为已有 release tag 手动运行。它构建 wheel 和 sdist，然后将两个文件上传到 GitHub Release。本项目不发布到 PyPI。

手动运行时，在 GitHub Actions 页面填写已有 release tag，也可以使用：

```bash
gh workflow run release.yml -f tag=v0.2.0
```

发布新版本时，创建 tag、推送 tag 并发布 GitHub Release，published-release 事件会自动提供 tag。该 workflow 只使用 `contents: write`，不会创建 PyPI environment，也不会请求 OIDC 发布权限。

## Release Asset

协议 v2 基线应上传 wheel、sdist、model card、可信 `best.pt` 和 checksum 文件。上传前后都应验证：

```bash
sha256sum -c SHA256SUMS-v0.2.0.txt
```

checkpoint 是基于 pickle 的可信 PyTorch artifact。文件旁应提供 SHA-256，并保留包含用途和限制说明的 model card。

仍然可以在本地构建包，但不会发布到 PyPI：

```bash
uv sync --locked --extra dev
uv build
uv run twine check dist/*
```
