"""Phase 2: Dixon-Coles bivariate Poisson with time decay.

Latent attack/defence per team, the tau low-score dependence correction, and
exponential time-decay weights (half-life selected on out-of-sample RPS).
Produces full scoreline grids; 1X2 derives from them.
"""

import datetime as dt

import pandas as pd

from wc2026.models.base import Fixture, OutcomeProbs, ScorelineDist


class DixonColesForecaster:
    """Dixon-Coles MLE model (implements ``ScorelineForecaster``)."""

    name = "dixon_coles"

    def __init__(self, half_life_days: float, max_goals: int = 10) -> None:
        self._half_life_days = half_life_days
        self._max_goals = max_goals

    def fit(self, history: pd.DataFrame, as_of: dt.date) -> None:
        """Weighted MLE over finished matches strictly before ``as_of``."""
        raise NotImplementedError("ships in Phase 2")

    def predict(self, fixture: Fixture) -> OutcomeProbs:
        """1X2 derived from the scoreline grid."""
        raise NotImplementedError("ships in Phase 2")

    def predict_scoreline(self, fixture: Fixture) -> ScorelineDist:
        """Joint goals grid with tau correction applied to the low-score cells."""
        raise NotImplementedError("ships in Phase 2")
