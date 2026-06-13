"""The production blend: a fixed-weight convex combination of base forecasters.

Phase 5/6 established that no single model beats Dixon-Coles, but a convex blend
of DC and the GBM does, walk-forward and significantly. The optimal weights were
remarkably stable across 13 six-monthly refits (DC ~0.67, GBM ~0.31, Bayes 0.00,
Elo ~0), so the production blend freezes a representative set rather than refit
live. ``BLEND_WEIGHTS`` records them; change only with fresh ``model-compare``
evidence.

This wraps base ``Forecaster`` objects and blends their 1X2 probabilities, so it
is itself a ``Forecaster`` (no scoreline grid — it does not feed the simulator).
"""

import datetime as dt

import pandas as pd

from wc2026.models.base import Fixture, Forecaster, OutcomeProbs

# Frozen from the walk-forward ensemble (rolling weights were stable at ~these
# values across 2020-2026; see results/model_comparison.md).
BLEND_WEIGHTS: dict[str, float] = {"dixon_coles": 0.67, "gbm": 0.33}


class BlendForecaster:
    """Fixed-weight convex blend of base forecasters (implements ``Forecaster``)."""

    name = "blend"

    def __init__(
        self,
        forecasters: dict[str, Forecaster],
        weights: dict[str, float] = BLEND_WEIGHTS,
    ) -> None:
        if set(weights) - set(forecasters):
            raise ValueError(f"weights reference unknown models: {set(weights) - set(forecasters)}")
        total = sum(weights.values())
        self._forecasters = forecasters
        self._weights = {name: w / total for name, w in weights.items()}

    def fit(self, history: pd.DataFrame, as_of: dt.date) -> None:
        """Fit every base forecaster on the same history/cutoff."""
        for forecaster in self._forecasters.values():
            forecaster.fit(history, as_of)

    def predict(self, fixture: Fixture) -> OutcomeProbs:
        """Weighted convex blend of the base forecasters' 1X2 probabilities."""
        home = draw = away = 0.0
        for name, weight in self._weights.items():
            probs = self._forecasters[name].predict(fixture)
            home += weight * probs.home
            draw += weight * probs.draw
            away += weight * probs.away
        return OutcomeProbs(home, draw, away).validated()
