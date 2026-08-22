# Contributing

Use Python 3.10-3.12 and `uv`. Document user-visible behavior in the relevant README or guide, and keep English/Chinese contract documentation aligned.

Before opening a pull request, run:

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

Changes to metrics, split membership, labels, checkpoints, or generated runtime inputs require focused regression tests and an architecture decision. Never rewrite a historical run to appear compatible with a new protocol. Keep large checkpoints and raw datasets outside Git; publish hashes for release assets.
