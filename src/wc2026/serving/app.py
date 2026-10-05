"""The prediction service: a thin HTTP layer over a loaded model artifact.

Boundaries: this module does HTTP only. Feature construction and inference
live in ``serving.inference``; artifact validation in ``serving.artifacts``;
S3 in ``serving.s3``; the evaluation numbers are produced offline by
``pipeline.export`` and served verbatim. Nothing here trains or backtests.

Startup: the artifact is loaded once, before the server accepts connections
(local directory, or one pinned S3 version). The S3 download is bounded by
``WC2026_ARTIFACT_LOAD_TIMEOUT_S`` plus the client's own timeouts/retries. If
loading fails the process stays up so the failure is observable: ``/health``
answers 200 (the process is alive), ``/ready`` and ``/predict`` answer 503
with a reason code, and the cause is in the logs. It does not retry on its own;
on ECS the failing ``/ready`` target check replaces the task.

Concurrency: handlers are plain ``def`` functions, so Starlette runs them in
its worker thread pool and the event loop is never blocked by inference. One
prediction is a few numpy operations plus a single-row LightGBM call.
"""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import anyio.to_thread
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from wc2026.serving.artifacts import ArtifactConfigError, ArtifactError
from wc2026.serving.inference import (
    LoadedArtifact,
    PredictionInputError,
    UnknownTeamError,
    load_artifact,
)
from wc2026.serving.logs import (
    RequestLogMiddleware,
    add_log_fields,
    configure_logging,
    current_request_id,
    log_event,
)
from wc2026.serving.schemas import (
    ComparisonResponse,
    ErrorResponse,
    FixtureEcho,
    HealthResponse,
    ModelInfo,
    NotReadyResponse,
    PredictRequest,
    PredictResponse,
    Probabilities,
    ReadyResponse,
    TeamsResponse,
)
from wc2026.serving.settings import ServingSettings

Loader = Callable[[ServingSettings], LoadedArtifact]

DESCRIPTION = """
1X2 (home win / draw / away win) probabilities for international football
fixtures from a saved Dixon-Coles + LightGBM blend.

* Predictions come from one pinned, pre-trained artifact. The service never
  trains, and it has **no live data feed**: the model knows nothing after its
  `training_cutoff`, and only dates up to `max_prediction_date` are accepted.
* `GET /models/comparison` returns evaluation results that were computed
  offline and stored with the artifact.
* Every error has the shape `{"error": {"code", "message", "details"?}, "request_id"}`.
"""


def load_from_settings(settings: ServingSettings) -> LoadedArtifact:
    """Resolve the configured artifact source and load it (blocking)."""
    if settings.artifact_source == "s3":
        if not settings.artifact_bucket or not settings.artifact_version:
            raise ArtifactConfigError(
                "S3 source needs WC2026_ARTIFACT_BUCKET and WC2026_ARTIFACT_VERSION"
            )
        from wc2026.serving import s3

        bundle_dir = s3.download_bundle(
            s3.make_client(),
            bucket=settings.artifact_bucket,
            prefix=settings.artifact_prefix,
            version=settings.artifact_version,
            cache_root=settings.artifact_cache_dir,
            timeout_s=settings.artifact_load_timeout_s,
        )
    else:
        if settings.artifact_dir is None:
            raise ArtifactConfigError("local source needs WC2026_ARTIFACT_DIR")
        bundle_dir = settings.artifact_dir
    return load_artifact(bundle_dir, expected_version=settings.artifact_version)


class ServiceState:
    """What the running process has loaded. Set once at startup, then read-only."""

    def __init__(self) -> None:
        self.artifact: LoadedArtifact | None = None
        self.not_ready_reason: str = "starting"


def _model_info(artifact: LoadedArtifact) -> ModelInfo:
    m = artifact.manifest
    return ModelInfo(
        artifact_version=m.artifact_version,
        training_cutoff=m.training_cutoff,
        max_prediction_date=m.max_prediction_date,
        feature_schema_version=m.feature_schema_version,
        synthetic=m.synthetic,
    )


def _error(
    status: int, code: str, message: str, details: list[dict[str, Any]] | None = None
) -> JSONResponse:
    add_log_fields(error_code=code)
    body = ErrorResponse.model_validate(
        {
            "error": {"code": code, "message": message, "details": details},
            "request_id": current_request_id(),
        }
    )
    return JSONResponse(body.model_dump(exclude_none=True), status_code=status)


_UNAVAILABLE: dict[int | str, dict[str, Any]] = {
    503: {
        "model": ErrorResponse,
        "description": "No model artifact is loaded (`model_unavailable`).",
    }
}


def create_app(
    settings: ServingSettings | None = None, *, loader: Loader = load_from_settings
) -> FastAPI:
    """Build the service. ``loader`` is injectable for tests."""
    settings = settings or ServingSettings()
    state = ServiceState()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.log_level)
        log_event(
            logging.INFO,
            "artifact_load_start",
            source=settings.artifact_source,
            pinned_version=settings.artifact_version,
        )
        try:
            artifact = await anyio.to_thread.run_sync(loader, settings)
        except ArtifactError as exc:
            state.not_ready_reason = exc.code
            log_event(logging.ERROR, "artifact_load_failed", reason=exc.code, detail=str(exc))
        except Exception:
            state.not_ready_reason = "artifact_load_error"
            logging.getLogger("wc2026.serving").exception(
                "artifact_load_failed", extra={"fields": {"reason": "artifact_load_error"}}
            )
        else:
            state.artifact = artifact
            m = artifact.manifest
            log_event(
                logging.INFO,
                "artifact_loaded",
                artifact_version=m.artifact_version,
                training_cutoff=m.training_cutoff.isoformat(),
                synthetic=m.synthetic,
                teams=len(artifact.predictor.teams),
                evaluation_available=artifact.evaluation is not None,
                built_with=m.runtime,
            )
        yield
        log_event(logging.INFO, "shutdown")

    app = FastAPI(
        title="WC2026 match prediction service",
        version="1",
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    app.state.service = state
    app.add_middleware(RequestLogMiddleware, max_body_bytes=settings.max_body_bytes)

    @app.exception_handler(RequestValidationError)
    async def _invalid(_: Request, exc: RequestValidationError) -> JSONResponse:
        # field + reason only; the offending input is never echoed back
        details = [
            {
                "field": ".".join(str(p) for p in err["loc"] if p != "body") or "body",
                "issue": str(err["msg"]),
            }
            for err in exc.errors()
        ]
        return _error(422, "validation_error", "request failed validation", details)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        codes = {404: "not_found", 405: "method_not_allowed"}
        return _error(exc.status_code, codes.get(exc.status_code, "http_error"), str(exc.detail))

    def _loaded() -> LoadedArtifact | JSONResponse:
        if state.artifact is None:
            return _error(
                503,
                "model_unavailable",
                f"no model artifact is loaded ({state.not_ready_reason})",
            )
        add_log_fields(artifact_version=state.artifact.manifest.artifact_version)
        return state.artifact

    @app.get("/health", response_model=HealthResponse, tags=["ops"], summary="Liveness")
    def health() -> HealthResponse:
        """The process is running and answering HTTP. Says nothing about the model."""
        return HealthResponse()

    @app.get(
        "/ready",
        response_model=ReadyResponse,
        responses={503: {"model": NotReadyResponse, "description": "No model is loaded."}},
        tags=["ops"],
        summary="Readiness",
    )
    def ready() -> Any:
        """200 only when a validated model artifact is in memory.

        Reads in-process state; it never calls S3, so it is safe to poll.
        """
        if state.artifact is None:
            return JSONResponse(
                NotReadyResponse(reason=state.not_ready_reason).model_dump(), status_code=503
            )
        return ReadyResponse(
            status="ready",
            model=_model_info(state.artifact),
            evaluation_available=state.artifact.evaluation is not None,
            teams=len(state.artifact.predictor.teams),
        )

    @app.get("/teams", response_model=TeamsResponse, responses=_UNAVAILABLE, tags=["predictions"])
    def teams() -> Any:
        """Team ids the loaded model can predict."""
        artifact = _loaded()
        if isinstance(artifact, JSONResponse):
            return artifact
        return TeamsResponse(teams=list(artifact.predictor.teams), model=_model_info(artifact))

    @app.post(
        "/predict",
        response_model=PredictResponse,
        responses={
            413: {"model": ErrorResponse, "description": "Body too large (`payload_too_large`)."},
            422: {
                "model": ErrorResponse,
                "description": "`validation_error` (malformed body), `unknown_team` (not in the"
                " loaded model; `details` names them) or `unsupported_date` (outside"
                " `training_cutoff`..`max_prediction_date`).",
            },
            **_UNAVAILABLE,
        },
        tags=["predictions"],
        summary="Predict one fixture",
    )
    def predict(body: PredictRequest) -> Any:
        """Blended home/draw/away probabilities for one fixture.

        Team strength and form are frozen at the artifact's `training_cutoff`;
        only `date` (rest days), `neutral` and `competition` vary per request.
        """
        artifact = _loaded()
        if isinstance(artifact, JSONResponse):
            return artifact
        try:
            result = artifact.predictor.predict(
                body.home_team,
                body.away_team,
                body.date,
                neutral=body.neutral,
                tournament=body.competition,
            )
        except UnknownTeamError as exc:
            return _error(
                422,
                exc.code,
                "team is not known to the loaded model; see GET /teams",
                [{"field": "team", "issue": f"unknown team id: {t}"} for t in exc.teams],
            )
        except PredictionInputError as exc:
            return _error(422, exc.code, str(exc))

        def as_probs(p: Any) -> Probabilities:
            return Probabilities(home_win=p.home, draw=p.draw, away_win=p.away)

        return PredictResponse(
            fixture=FixtureEcho(**body.model_dump()),
            probabilities=as_probs(result.blend),
            components={
                "dixon_coles": as_probs(result.dixon_coles),
                "gbm": as_probs(result.gbm),
            },
            blend_weights=artifact.predictor.blend_weights,
            model=_model_info(artifact),
            request_id=current_request_id() or "",
        )

    @app.get(
        "/models/comparison",
        response_model=ComparisonResponse,
        responses={
            503: {
                "model": ErrorResponse,
                "description": "`model_unavailable`, or `evaluation_unavailable` when the loaded"
                " artifact was exported without evaluation results.",
            }
        },
        tags=["evaluation"],
        summary="Stored model-vs-model and model-vs-market results",
    )
    def comparison() -> Any:
        """The evaluation stored with the loaded artifact, verbatim.

        Two separate tracks: `historical` (walk-forward model results; no market
        baseline exists for it) and `market` (models vs the de-vigged closing-odds
        baseline on the matches that have stored odds). Check each track's `n`.
        """
        artifact = _loaded()
        if isinstance(artifact, JSONResponse):
            return artifact
        if artifact.evaluation is None:
            return _error(
                503,
                "evaluation_unavailable",
                "the loaded artifact has no stored evaluation results",
            )
        return ComparisonResponse(model=_model_info(artifact), evaluation=artifact.evaluation)

    return app
