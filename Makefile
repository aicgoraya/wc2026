.PHONY: setup test test-live lint format typecheck check refresh dashboard fixture-artifact serve smoke infra-synth

setup:
	uv sync --dev --group infra --extra bayes --extra gbm --extra dashboard --extra serve

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

# --- prediction service -----------------------------------------------------
# Synthetic TEST artifact (invented teams) for trying the API without any data.
fixture-artifact:
	uv run wc2026 export-artifact --synthetic --version synthetic-fixture --out artifacts

# Serve a local bundle: make serve ARTIFACT=artifacts/<version>
ARTIFACT ?= artifacts/synthetic-fixture
serve:
	WC2026_ARTIFACT_SOURCE=local WC2026_ARTIFACT_DIR=$(ARTIFACT) uv run wc2026 serve --port 8080

smoke:
	python3 tools/smoke.py http://127.0.0.1:8080

infra-synth:
	cd infra && uv run --group infra npx --yes aws-cdk@2.1144.0 synth --quiet -c artifact_version=local
