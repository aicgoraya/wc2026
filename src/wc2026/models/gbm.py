"""Phase 5: LightGBM multiclass comparison model on engineered features.

The discriminative contrast point to the generative ladder. Compared honestly
on the same walk-forward scoreboard — not rigged to win or lose.

Requires the ``gbm`` extra (``uv sync --extra gbm``).
"""

import datetime as dt

import pandas as pd

from wc2026.models.base import Fixture, OutcomeProbs


class GbmForecaster:
    """LightGBM 1X2 classifier with calibration check (implements ``Forecaster``)."""

    name = "gbm"

    def fit(self, history: pd.DataFrame, as_of: dt.date) -> None:
        """Train on features built strictly from pre-``as_of`` information."""
        raise NotImplementedError("ships in Phase 5")

    def predict(self, fixture: Fixture) -> OutcomeProbs:
        """Predicted class probabilities for the fixture."""
        raise NotImplementedError("ships in Phase 5")
