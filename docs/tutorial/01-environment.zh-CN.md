# 环境

使用 Python 3.10-3.12 和 uv：

```bash
uv sync --locked --extra dev
uv run instance-segment --version
uv run python -m pytest
```

CPU 足以运行测试和 dry-run。正式 GPU 完整训练使用生成的 Kaggle runner。
