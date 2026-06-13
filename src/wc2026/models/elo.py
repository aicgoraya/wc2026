"""Phase 1 baseline: Elo rating difference → ordinal logistic 1X2 probabilities.

The floor every later model must beat. Two leak-free pieces, both fit only on
matches strictly before the ``as_of`` cutoff:

1. Elo ratings from the chronological replay (``data.sources.elo_own``).
2. A proportional-odds (ordinal logistic) link from pre-match rating
   difference to (home, draw, away) probabilities — the draw probability
   peaks where ratings are level and home advantage enters through the
   rating difference, not a separate draw model.

Outcome ordering along the latent scale: away < draw < home as the home-side
rating advantage grows.
"""

import dataclasses
import datetime as dt

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize

from wc2026.data.sources.elo_own import INITIAL_RATING, replay
from wc2026.models.base import Fixture, OutcomeProbs

FloatArray = npt.NDArray[np.float64]

_ELO_NATURAL_SLOPE = np.log(10.0) / 400.0  # logistic slope implied by the Elo update rule


@dataclasses.dataclass(frozen=True)
class OrdinalParams:
    """Proportional-odds parameters: thresholds ``theta_lo < theta_hi``, slope beta."""

    beta: float
    theta_lo: float
    theta_hi: float


def _sigmoid(z: FloatArray) -> FloatArray:
    result: FloatArray = 1.0 / (1.0 + np.exp(-z))
    return result


def predict_ordinal(params: OrdinalParams, xdiff: FloatArray) -> FloatArray:
    """(n,) rating diffs → (n, 3) probabilities ordered (home, draw, away)."""
    x = np.asarray(xdiff, dtype=np.float64)
    p_away = _sigmoid(params.theta_lo - params.beta * x)
    p_home = _sigmoid(params.beta * x - params.theta_hi)
    p_draw = 1.0 - p_away - p_home
    return np.stack([p_home, p_draw, p_away], axis=-1)


def fit_ordinal_logistic(xdiff: FloatArray, outcomes: FloatArray) -> OrdinalParams:
    """MLE for the proportional-odds model on (rating diff, outcome code) pairs.

    Parameterized with ``theta_hi = theta_lo + exp(log_gap)`` so the threshold
    order constraint can never be violated during optimization.
    """
    x = np.asarray(xdiff, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.int64)
    if len(x) < 100:
        raise ValueError(f"need at least 100 samples to fit the link, got {len(x)}")

    def nll(params: FloatArray) -> float:
        beta, theta_lo, log_gap = params
        probs = predict_ordinal(
            OrdinalParams(beta=beta, theta_lo=theta_lo, theta_hi=theta_lo + np.exp(log_gap)), x
        )
        picked = probs[np.arange(len(y)), y]
        return float(-np.log(np.clip(picked, 1e-12, 1.0)).sum())

    start = np.array([_ELO_NATURAL_SLOPE, -0.45, np.log(0.9)])
    result = minimize(nll, start, method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-6})
    if not result.success:
        raise RuntimeError(f"ordinal logistic fit did not converge: {result.message}")
    beta, theta_lo, log_gap = result.x
    return OrdinalParams(
        beta=float(beta), theta_lo=float(theta_lo), theta_hi=float(theta_lo + np.exp(log_gap))
    )


class EloForecaster:
    """Elo + ordinal logistic baseline (implements ``Forecaster``)."""

    name = "elo_baseline"

    def __init__(self, home_advantage: float = 100.0, link_window_years: int = 30) -> None:
        self._home_advantage = home_advantage
        self._link_window_years = link_window_years
        self._ratings: dict[str, float] | None = None
        self._params: OrdinalParams | None = None

    def fit(self, history: pd.DataFrame, as_of: dt.date) -> None:
        """Replay ratings and fit the link, both strictly before ``as_of``."""
        rep = replay(history, as_of=as_of, home_advantage=self._home_advantage)
        window_start = pd.Timestamp(as_of) - pd.DateOffset(years=self._link_window_years)
        samples = rep.samples[rep.samples["date"] >= window_start]
        self._params = fit_ordinal_logistic(
            samples["xdiff"].to_numpy(), samples["outcome"].to_numpy()
        )
        self._ratings = rep.ratings

    def predict(self, fixture: Fixture) -> OutcomeProbs:
        """1X2 from the fitted ratings and link; unseen teams enter at 1500."""
        if self._ratings is None or self._params is None:
            raise RuntimeError("call fit() before predict()")
        home = self._ratings.get(fixture.home_id, INITIAL_RATING)
        away = self._ratings.get(fixture.away_id, INITIAL_RATING)
        advantage = 0.0 if fixture.neutral else self._home_advantage
        xdiff = np.array([home + advantage - away])
        p_home, p_draw, p_away = predict_ordinal(self._params, xdiff)[0]
        return OutcomeProbs(float(p_home), float(p_draw), float(p_away)).validated()
