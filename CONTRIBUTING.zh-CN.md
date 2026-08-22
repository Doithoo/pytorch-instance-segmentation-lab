# 贡献指南

使用 Python 3.10-3.12 和 uv。提交前执行：

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run python -m pytest
uv run python scripts/build_kaggle_runner.py --check
```

保持数据契约、训练记录和生成的 Kaggle runner 同步。
