# Contributing

Use Python 3.10-3.12 and uv. Before opening a pull request, run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run python -m pytest
uv run python scripts/build_kaggle_runner.py --check
```

Keep data contracts, recorded-run artifacts, and the generated Kaggle runner in sync.
