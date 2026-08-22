# Environment

Use Python 3.10-3.12 and uv:

```bash
uv sync --locked --extra dev
uv run instance-segment --version
uv run python -m pytest
```

CPU is enough for tests and the dry-run. Use the generated Kaggle runner for the required full GPU training.
