# Verification record (2026-10-05)

What was actually run for the prediction-service change, and what was not.
Environment: Linux sandbox VM, 2 vCPU, Python 3.12.3, uv 0.11.32.

## Verified

| Check | Command | Result |
|---|---|---|
| Baseline before any edit | `uv run pytest -m "not live"`; ruff; mypy | 266 passed; clean |
| Full suite after | `uv run pytest -m "not live"` | **366 passed** (266 existing + 100 new) |
| Lint / format / types | `uv run ruff check src tests`, `ruff format --check`, `uv run mypy` | clean (62 source files) |
| Real dataset | `wc2026 ingest-history` on the public history + `data` branch | 49,546 finished matches, last result 2026-08-26 |
| Real artifact | `wc2026 export-artifact --version 2026-10-05.1` | written; served predictions equal the in-memory pipeline on probe fixtures (export's parity gate) |
| Runtime-only install | `uv sync --frozen --no-dev --extra serve --no-editable` (the Dockerfile's install step) | installs; no scikit-learn, PyMC, pytest or CDK present |
| Service as non-root | `wc2026 serve --host 0.0.0.0 --port 8080` as uid 10001 with the real artifact | ready in under 1 s, 322 teams, about 245 MB resident |
| Smoke | `tools/smoke.py http://127.0.0.1:8080 --expect-version 2026-10-05.1 --require-real` | passed |
| Graceful stop | SIGTERM | clean shutdown in about 2 s |
| Load (local) | `tools/loadtest.py … --concurrency 8 --duration 30` | 12,319 requests, 0 errors, p50 19.3 ms, p95 25.9 ms, p99 29.4 ms |
| Infrastructure | `cdk synth`; `tests/infra` (14 template assertions) | synthesizes without AWS access; assertions pass |
| Workflows | `actionlint .github/workflows/*.yml` | no findings |

Evaluation stored in the artifact (copy in `results/service_evaluation_2026-10-05.1.json`):
historical walk-forward 2018-01-01 to 2026-06-10, n = 8,116: Elo 0.1712, Dixon-Coles 0.1675,
LightGBM 0.1705 RPS; rolling-weights blend 0.1655 vs Dixon-Coles 0.1665 on n = 6,031. World Cup
2026 vs the de-vigged closing-line proxy, n = 101: market 0.1501, blend 0.1559.

## Not verified

| Item | Why | What will verify it |
|---|---|---|
| `docker build` and the container smoke test | The sandbox cannot reach any container registry | The `container` job on the first pull request |
| CI and deploy workflows executing on GitHub | Not pushed from here | First pull request; first manual `deploy` run |
| Anything in AWS: stack creation, S3 loading on ECS, CloudWatch delivery, rollout, rollback | No AWS account or approval | `docs/deployment.md`, then a green `deploy` run |
| CloudWatch Logs Insights queries | No live log group | Run them after the first deployment |
| Performance on the Fargate task | Not deployed | `tools/loadtest.py` against the deployed URL |
