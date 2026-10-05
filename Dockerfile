# syntax=docker/dockerfile:1
# Prediction service image: runtime dependencies only (the `serve` extra) -
# no training stack, no dev tools, no data, no credentials.
# Build for the ECS task architecture:  docker build --platform linux/amd64 -t wc2026-api .
ARG PYTHON_IMAGE=python:3.12-slim-bookworm
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.32

FROM ${UV_IMAGE} AS uv

FROM ${PYTHON_IMAGE} AS build
COPY --from=uv /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app
# dependencies first (cached until the lockfile changes), exactly as locked
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra serve --no-install-project
# then the project itself, installed into the venv (not editable)
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra serve --no-editable

FROM ${PYTHON_IMAGE} AS runtime
# libgomp1: OpenMP runtime LightGBM links against
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --create-home --home-dir /home/app app
COPY --from=build /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    WC2026_ARTIFACT_CACHE_DIR=/tmp/wc2026-artifacts
USER 10001
WORKDIR /home/app
EXPOSE 8080
# Local/Compose convenience only: ECS ignores this and uses the load balancer's /ready check.
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2).status == 200 else 1)"]
# Exec form: uvicorn is PID 1 and receives SIGTERM directly, so it drains
# in-flight requests (up to 20s) before exiting.
CMD ["wc2026", "serve", "--host", "0.0.0.0", "--port", "8080"]
