.PHONY: lint format-check typecheck test build metadata check kaggle-runner

lint:
	uv run ruff check .

format-check:
	uv run ruff format --check .

typecheck:
	uv run mypy

test:
	uv run python -m pytest

build:
	uv run python -m build

metadata:
	uv run twine check dist/*

kaggle-runner:
	uv run python scripts/build_kaggle_runner.py --check

check: lint format-check typecheck test build metadata kaggle-runner
