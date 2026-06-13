.PHONY: setup test lint format typecheck check refresh dashboard

setup:
	uv sync --dev --extra bayes --extra gbm --extra dashboard

test:
	uv run pytest -m "not live"

test-live:
	uv run pytest -m live

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

typecheck:
	uv run mypy

check: lint typecheck test

refresh:
	uv run wc2026 refresh

dashboard:
	uv run wc2026 dashboard
