.PHONY: lock-check lint format-check typecheck test coverage build metadata kaggle-runner check

lock-check:
	uv lock --check

lint:
	uv run ruff check .

format-check:
	uv run ruff format --check .

typecheck:
	uv run mypy

test:
	uv run python -m pytest

coverage:
	uv run python -m pytest --cov=instance_segmenter --cov-report=term-missing

build:
	uv run python -m build

metadata:
	uv run twine check dist/*

kaggle-runner:
	uv run python scripts/build_kaggle_runner.py --check

check: lock-check lint format-check typecheck coverage build metadata kaggle-runner
