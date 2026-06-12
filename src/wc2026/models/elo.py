"""Phase 1 baseline: Elo rating difference → ordinal logistic 1X2 probabilities.

The floor every later model must beat. Deliberately simple: rating diff plus a
neutral-venue adjustment, draw probability from an ordered-logit cut.
"""

import datetime as dt

import pandas as pd

from wc2026.models.base import Fixture, OutcomeProbs


class EloForecaster:
    """Elo + ordinal logistic baseline (implements ``Forecaster``)."""

    name = "elo_baseline"

    def fit(self, history: pd.DataFrame, as_of: dt.date) -> None:
        """Replay history to current ratings; fit the ordinal link on past matches."""
        raise NotImplementedError("ships in Phase 1c")

    def predict(self, fixture: Fixture) -> OutcomeProbs:
        """1X2 from the fitted ratings and link."""
        raise NotImplementedError("ships in Phase 1c")
