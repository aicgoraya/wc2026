"""Phase 5: LightGBM multiclass — the discriminative contrast to the ladder.

Unlike the generative models (Elo/DC/Bayes), the GBM consumes engineered
features the others structurally cannot — recent form, rest, momentum,
competition importance (``features.build``). It is a fundamentally different
model class, included as an honest contrast, not rigged to win or lose.

Leak-freedom is by construction: the feature matrix is built once and every
row is computed strictly from prior matches; ``fit`` trains only on rows whose
date is before the as-of cutoff, so no post-cutoff label or feature reaches
the model. ``predict`` looks up the fixture's pre-computed (as-of-its-date)
features.

Implements ``Forecaster`` (1X2 only) — it produces no scoreline grid, so it is
deliberately not a ``ScorelineForecaster`` and does not feed the tournament
simulator; its role is the eval scoreboard and the ensemble.

Requires the ``gbm`` extra (``uv sync --extra gbm``).
"""

import datetime as dt
import warnings
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from wc2026.features.build import FEATURE_COLUMNS, build_feature_matrix
from wc2026.models.base import Fixture, OutcomeProbs

FloatArray = npt.NDArray[np.float64]

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "multiclass",
    "num_class": 3,
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "verbose": -1,
}


class GbmForecaster:
    """LightGBM 1X2 classifier on engineered features (implements ``Forecaster``)."""

    name = "gbm"

    def __init__(self, matches: pd.DataFrame, params: dict[str, Any] | None = None) -> None:
        self._params = params or DEFAULT_PARAMS
        self._matrix = build_feature_matrix(matches)
        self._lookup: dict[tuple[pd.Timestamp, str, str], FloatArray] = {}
        for row in self._matrix.itertuples(index=False):
            key = (cast(pd.Timestamp, row.date), str(row.home_id), str(row.away_id))
            self._lookup[key] = np.array(
                [getattr(row, c) for c in FEATURE_COLUMNS], dtype=np.float64
            )
        self._model: Any = None

    def fit(self, history: pd.DataFrame, as_of: dt.date) -> None:
        """Train on feature rows strictly before ``as_of`` (finished matches only)."""
        from lightgbm import LGBMClassifier

        train = self._matrix[
            (self._matrix["date"] < pd.Timestamp(as_of)) & (self._matrix["label"] >= 0)
        ]
        if len(train) < 500:
            raise ValueError(f"need at least 500 labelled rows to fit, got {len(train)}")
        # fit on a bare ndarray (no feature names) so predict on ndarrays is silent;
        # feature_importances_ stay positional and map back to FEATURE_COLUMNS by index
        x = np.ascontiguousarray(train[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64))
        y = train["label"].to_numpy(dtype=np.int64)
        self._model = LGBMClassifier(**self._params)
        self._model.fit(x, y)

    def _features(self, fixture: Fixture) -> FloatArray:
        key = (pd.Timestamp(fixture.date), fixture.home_id, fixture.away_id)
        feats = self._lookup.get(key)
        if feats is None:
            raise KeyError(
                f"no precomputed features for {fixture.home_id} v {fixture.away_id}"
                f" on {fixture.date}; build the matrix from a frame containing this fixture"
            )
        return feats

    def predict(self, fixture: Fixture) -> OutcomeProbs:
        """Predicted class probabilities (home, draw, away) for the fixture."""
        if self._model is None:
            raise RuntimeError("call fit() before predict()")
        with warnings.catch_warnings():
            # LightGBM's sklearn wrapper records default feature names internally;
            # predicting on a bare ndarray is correct but warns. Harmless, silence it.
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            proba = self._model.predict_proba(self._features(fixture).reshape(1, -1))[0]
        return OutcomeProbs(float(proba[0]), float(proba[1]), float(proba[2])).validated()

    def feature_importances(self) -> dict[str, float]:
        """Gain-based feature importances from the last fit."""
        if self._model is None:
            raise RuntimeError("call fit() before reading importances")
        return dict(zip(FEATURE_COLUMNS, self._model.feature_importances_, strict=True))
