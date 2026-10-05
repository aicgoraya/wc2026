"""Typed request/response models for the prediction API."""

import datetime as dt
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from wc2026.data.names import canonical_slug

_TEAM_DESC = (
    "Team id (canonical slug such as `brazil` or `united_states`); a plain English name"
    " like `United States` is normalised to its slug. `GET /teams` lists what the loaded"
    " model supports."
)


class PredictRequest(BaseModel):
    """One fixture to predict."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "home_team": "brazil",
                    "away_team": "morocco",
                    "date": "2026-10-10",
                    "neutral": True,
                    "competition": "friendly",
                }
            ]
        },
    )

    home_team: str = Field(min_length=1, max_length=64, description=_TEAM_DESC)
    away_team: str = Field(min_length=1, max_length=64, description=_TEAM_DESC)
    date: dt.date = Field(
        description="Match date (ISO). Must be on/after the model's training cutoff and no"
        " later than its `max_prediction_date` (both reported by `GET /ready`)."
    )
    neutral: bool = Field(
        default=True,
        description="True for a neutral venue; False when `home_team` plays at home.",
    )
    competition: str = Field(
        default="friendly",
        pattern=r"^[a-z0-9_]{1,64}$",
        description="Competition slug used for the match-importance feature, e.g. `friendly`,"
        " `fifa_world_cup`, `uefa_euro`, `fifa_world_cup_qualification`. Unrecognised slugs"
        " get the default weight for 'other competitive match'.",
    )

    @field_validator("home_team", "away_team")
    @classmethod
    def _slug(cls, value: str) -> str:
        return canonical_slug(value)

    @model_validator(mode="after")
    def _distinct(self) -> "PredictRequest":
        if self.home_team == self.away_team:
            raise ValueError("home_team and away_team must be different teams")
        return self


class Probabilities(BaseModel):
    """A 1X2 distribution; the three values sum to 1."""

    home_win: float = Field(ge=0.0, le=1.0)
    draw: float = Field(ge=0.0, le=1.0)
    away_win: float = Field(ge=0.0, le=1.0)


class ModelInfo(BaseModel):
    """Which artifact produced a response."""

    artifact_version: str
    training_cutoff: dt.date
    max_prediction_date: dt.date
    feature_schema_version: str
    synthetic: bool = Field(
        description="True when the loaded artifact is a synthetic test fixture, not a model"
        " trained on real match data. Never treat such output as a real forecast."
    )


class FixtureEcho(BaseModel):
    """The fixture as the service interpreted it (team ids normalised)."""

    home_team: str
    away_team: str
    date: dt.date
    neutral: bool
    competition: str


class PredictResponse(BaseModel):
    """Blended 1X2 probabilities for the fixture, with component models."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "fixture": {
                        "home_team": "brazil",
                        "away_team": "morocco",
                        "date": "2026-10-10",
                        "neutral": True,
                        "competition": "friendly",
                    },
                    "probabilities": {"home_win": 0.5, "draw": 0.27, "away_win": 0.23},
                    "components": {
                        "dixon_coles": {"home_win": 0.51, "draw": 0.27, "away_win": 0.22},
                        "gbm": {"home_win": 0.48, "draw": 0.27, "away_win": 0.25},
                    },
                    "blend_weights": {"dixon_coles": 0.67, "gbm": 0.33},
                    "model": {
                        "artifact_version": "example",
                        "training_cutoff": "2026-10-01",
                        "max_prediction_date": "2027-03-30",
                        "feature_schema_version": "1",
                        "synthetic": False,
                    },
                    "request_id": "0f8fad5bd9cb469fa16570867728950e",
                }
            ]
        }
    )

    fixture: FixtureEcho
    probabilities: Probabilities = Field(description="The production blend.")
    components: dict[str, Probabilities] = Field(
        description="Each base model's own probabilities, before blending."
    )
    blend_weights: dict[str, float]
    model: ModelInfo
    request_id: str


class ErrorBody(BaseModel):
    """Machine-readable error."""

    code: str = Field(description="Stable error identifier, e.g. `unknown_team`.")
    message: str
    details: list[dict[str, Any]] | None = None


class ErrorResponse(BaseModel):
    """Every non-2xx response has this shape."""

    error: ErrorBody
    request_id: str | None = None


class HealthResponse(BaseModel):
    """Liveness: the process is up and answering."""

    status: str = "ok"


class ReadyResponse(BaseModel):
    """Readiness: the model artifact is loaded and predictions can be served."""

    status: str
    model: ModelInfo
    evaluation_available: bool
    teams: int


class NotReadyResponse(BaseModel):
    """Returned with 503 while (or because) no model is loaded."""

    status: str = "not_ready"
    reason: str = Field(description="Stable reason code, e.g. `artifact_not_found`.")


class TeamsResponse(BaseModel):
    """Team ids the loaded model can predict."""

    teams: list[str]
    model: ModelInfo


class ComparisonResponse(BaseModel):
    """Stored offline evaluation results for the loaded artifact."""

    model: ModelInfo
    evaluation: dict[str, Any] = Field(
        description="The artifact's `evaluation.json`, verbatim: walk-forward model results"
        " (`historical`) and, separately, the comparison against the de-vigged closing-odds"
        " baseline (`market`). Computed offline by `wc2026 export-artifact`; the service"
        " never computes or alters these numbers."
    )
