"""Online inference from a loaded artifact: no fitting, no data access.

``Predictor`` holds the fitted Dixon-Coles parameters, the LightGBM booster and
the per-team feature state as of the training cutoff, and reproduces the
production blend (``models.blend``) for one fixture. Features come from the
same ``features.build.fixture_features`` the training matrix uses.
"""

import dataclasses
import datetime as dt
import threading
from pathlib import Path
from typing import Any

import numpy as np

from wc2026.features.build import FEATURE_COLUMNS, TeamState, fixture_features
from wc2026.models.base import Fixture, OutcomeProbs
from wc2026.models.dixon_coles import DixonColesForecaster
from wc2026.serving.artifacts import (
    ArtifactCorruptError,
    Manifest,
    RawBundle,
    read_bundle,
)


class PredictionInputError(ValueError):
    """The fixture is well-formed but this artifact cannot predict it."""

    code = "unsupported_input"


class UnknownTeamError(PredictionInputError):
    """One or both teams are not in the loaded artifact."""

    code = "unknown_team"

    def __init__(self, teams: list[str]) -> None:
        super().__init__(f"unknown team id(s): {', '.join(teams)}")
        self.teams = teams


class UnsupportedDateError(PredictionInputError):
    """The fixture date is outside the window this artifact supports."""

    code = "unsupported_date"


@dataclasses.dataclass(frozen=True)
class Prediction:
    """Blended 1X2 probabilities plus the two component models' probabilities."""

    blend: OutcomeProbs
    dixon_coles: OutcomeProbs
    gbm: OutcomeProbs


class Predictor:
    """Thread-safe single-fixture inference over one loaded artifact."""

    def __init__(self, bundle: RawBundle) -> None:
        import lightgbm

        self.manifest: Manifest = bundle.manifest
        self._dc = DixonColesForecaster.from_params(
            bundle.dc_params, max_goals=int(bundle.manifest.dixon_coles.get("max_goals", 10))
        )
        try:
            self._booster: Any = lightgbm.Booster(model_str=bundle.gbm_model_string)
        except lightgbm.basic.LightGBMError as exc:
            raise ArtifactCorruptError("gbm.txt is not a loadable LightGBM model") from exc
        if (
            self._booster.num_feature() != len(FEATURE_COLUMNS)
            or self._booster.num_model_per_iteration() != 3
        ):
            raise ArtifactCorruptError("gbm.txt does not match the feature schema / 3 classes")
        # LightGBM documents Booster.predict as thread-safe, but a single-row
        # predict takes well under a millisecond, so serialising it costs nothing
        # and removes any dependence on that guarantee.
        self._gbm_lock = threading.Lock()
        self._states: dict[str, TeamState] = bundle.team_states
        total = sum(bundle.manifest.blend_weights.values())
        self._weights = {k: v / total for k, v in bundle.manifest.blend_weights.items()}
        # a team is predictable only if BOTH models know it
        self.teams: tuple[str, ...] = tuple(sorted(set(bundle.dc_params.teams) & set(self._states)))
        self._team_set = frozenset(self.teams)
        if len(self.teams) < 2:
            raise ArtifactCorruptError("artifact has fewer than two predictable teams")

    @property
    def blend_weights(self) -> dict[str, float]:
        """The normalised blend weights in use."""
        return dict(self._weights)

    def predict(
        self, home_id: str, away_id: str, date: dt.date, *, neutral: bool, tournament: str
    ) -> Prediction:
        """Blend the two models for one fixture; raises ``PredictionInputError``."""
        unknown = [t for t in (home_id, away_id) if t not in self._team_set]
        if unknown:
            raise UnknownTeamError(unknown)
        m = self.manifest
        if not (m.training_cutoff <= date <= m.max_prediction_date):
            raise UnsupportedDateError(
                f"date must be between {m.training_cutoff} (the training cutoff) and"
                f" {m.max_prediction_date}; this model has no information after its cutoff"
                " and is not validated for longer horizons"
            )

        dc = self._dc.predict(Fixture(home_id, away_id, date, neutral=neutral))
        feats = fixture_features(
            self._states[home_id],
            self._states[away_id],
            date,
            neutral=neutral,
            tournament=tournament,
        )
        x = np.array([[feats[c] for c in FEATURE_COLUMNS]], dtype=np.float64)
        with self._gbm_lock:
            raw = self._booster.predict(x)[0]
        gbm = OutcomeProbs(float(raw[0]), float(raw[1]), float(raw[2])).validated()

        w_dc, w_gbm = self._weights["dixon_coles"], self._weights["gbm"]
        blend = OutcomeProbs(
            w_dc * dc.home + w_gbm * gbm.home,
            w_dc * dc.draw + w_gbm * gbm.draw,
            w_dc * dc.away + w_gbm * gbm.away,
        ).validated()
        return Prediction(blend=blend, dixon_coles=dc, gbm=gbm)


@dataclasses.dataclass(frozen=True)
class LoadedArtifact:
    """Everything the service needs from one artifact."""

    predictor: Predictor
    evaluation: dict[str, Any] | None

    @property
    def manifest(self) -> Manifest:
        """The artifact's manifest."""
        return self.predictor.manifest


def load_artifact(bundle_dir: Path, *, expected_version: str | None = None) -> LoadedArtifact:
    """Validate, parse and build the predictor for a local bundle directory."""
    bundle = read_bundle(bundle_dir, expected_version=expected_version)
    return LoadedArtifact(predictor=Predictor(bundle), evaluation=bundle.evaluation)
