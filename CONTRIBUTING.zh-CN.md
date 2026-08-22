# 贡献指南

使用 Python 3.10-3.12 和 `uv`。在相关 README 或指南中记录用户可见改动，并保持中英文契约文档一致。

提交 PR 前运行：

```bash
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run python -m pytest --cov=instance_segmenter --cov-report=term-missing
uv run python scripts/build_kaggle_runner.py --check
uv run python -m build
uv run twine check dist/*
```

指标、split 成员、label、checkpoint 或生成运行时输入发生变化时，必须增加针对性回归测试和架构决策。不要改写历史运行，使其看起来与新协议兼容。大型 checkpoint 和原始数据不进入 Git；发布 asset 时必须提供 hash。
