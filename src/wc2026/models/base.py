"""Model interfaces and probability containers.

The contracts every forecaster implements, plus ``ScorelineDist`` — the joint
distribution over (home goals, away goals) that the tournament simulator and
the 1X2 derivation both consume.
"""

import dataclasses
import datetime as dt
import math
from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd

from wc2026.data.schema import Stage

PROB_ATOL = 1e-6
"""Tolerance for 'probabilities sum to 1' checks."""


class OutcomeProbs(NamedTuple):
    """A 1X2 probability distribution (home win, draw, away win)."""

    home: float
    draw: float
    away: float

    def validated(self) -> "OutcomeProbs":
        """Return self if a valid distribution, else raise ``ValueError``."""
        vals = (self.home, self.draw, self.away)
        if any(not math.isfinite(v) or v < -PROB_ATOL or v > 1.0 + PROB_ATOL for v in vals):
            raise ValueError(f"probabilities must be in [0, 1]: {self}")
        if abs(sum(vals) - 1.0) > PROB_ATOL:
            raise ValueError(f"probabilities must sum to 1: {self}")
        return self

    def as_array(self) -> npt.NDArray[np.float64]:
        """Return as a (3,) float array ordered home, draw, away."""
        return np.array(self, dtype=np.float64)


@dataclasses.dataclass(frozen=True)
class Fixture:
    """A match to predict, identified by canonical team slugs."""

    home_id: str
    away_id: str
    date: dt.date
    neutral: bool = True
    stage: Stage | None = None


@dataclasses.dataclass(frozen=True)
class ScorelineDist:
    """Joint distribution over goals: ``matrix[i, j] = P(home=i, away=j)``.

    The matrix is square with shape (K+1, K+1); the final index folds the tail
    mass P(goals >= K), so the matrix always sums to 1 exactly as a distribution.
    """

    matrix: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        m = self.matrix
        if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] < 2:
            raise ValueError(f"matrix must be square (K+1, K+1) with K >= 1, got {m.shape}")
        if not np.isfinite(m).all() or (m < 0).any():
            raise ValueError("matrix entries must be finite and non-negative")
        if abs(float(m.sum()) - 1.0) > PROB_ATOL:
            raise ValueError(f"matrix must sum to 1, got {m.sum():.8f}")
        m.setflags(write=False)

    @property
    def max_goals(self) -> int:
        """K, the tail-folding goal cap."""
        return int(self.matrix.shape[0]) - 1

    def prob(self, home: int, away: int) -> float:
        """P(home goals = home, away goals = away), with the cap K meaning '>= K'."""
        k = self.max_goals
        if not (0 <= home <= k and 0 <= away <= k):
            raise ValueError(f"scoreline ({home}, {away}) outside [0, {k}]")
        return float(self.matrix[home, away])

    def outcome_probs(self) -> OutcomeProbs:
        """Derive the 1X2 distribution: home win below the diagonal, draws on it."""
        home = float(np.tril(self.matrix, k=-1).sum())
        draw = float(np.trace(self.matrix))
        away = float(np.triu(self.matrix, k=1).sum())
        return OutcomeProbs(home, draw, away).validated()

    def top_scorelines(self, n: int) -> list[tuple[int, int, float]]:
        """The n most likely (home, away, probability) scorelines, descending."""
        flat = self.matrix.ravel()
        order = np.argsort(flat)[::-1][:n]
        k = self.matrix.shape[0]
        return [(int(i // k), int(i % k), float(flat[i])) for i in order]

    def sample(self, rng: np.random.Generator, n: int) -> npt.NDArray[np.int64]:
        """Draw n scorelines; returns an (n, 2) array of (home, away) goals."""
        flat = self.matrix.ravel()
        idx = rng.choice(flat.size, size=n, p=flat / flat.sum())
        k = self.matrix.shape[0]
        return np.column_stack(np.divmod(idx, k)).astype(np.int64)


@runtime_checkable
class Forecaster(Protocol):
    """Anything that can be fitted on history and produce 1X2 probabilities.

    ``fit`` must only use matches strictly before ``as_of`` — the walk-forward
    harness enforces this by construction, models must not re-fetch data.
    """

    name: str

    def fit(self, history: pd.DataFrame, as_of: dt.date) -> None:
        """Fit on the canonical matches frame (rows strictly before ``as_of``)."""
        ...

    def predict(self, fixture: Fixture) -> OutcomeProbs:
        """Predict the 1X2 distribution for a fixture."""
        ...


@runtime_checkable
class ScorelineForecaster(Forecaster, Protocol):
    """A forecaster that produces full scoreline distributions (DC, Bayes).

    Required by the tournament simulator; 1X2 derives from the scoreline grid.
    """

    def predict_scoreline(self, fixture: Fixture) -> ScorelineDist:
        """Predict the joint goals distribution for a fixture."""
        ...


@runtime_checkable
class PosteriorForecaster(ScorelineForecaster, Protocol):
    """A Bayesian forecaster exposing per-draw posterior outcome probabilities."""

    def predict_posterior(self, fixture: Fixture, n_draws: int) -> npt.NDArray[np.float64]:
        """Return an (n_draws, 3) array of 1X2 probabilities across posterior draws."""
        ...
