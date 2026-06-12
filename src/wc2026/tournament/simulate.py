"""Vectorized Monte-Carlo tournament simulation (50k+ runs).

Samples scorelines from a ``ScorelineForecaster`` for every unplayed fixture,
applies the exact group ranking, Annex C allocation and bracket chain, and an
explicit extra-time/penalties policy for knockouts. Re-run after every
completed match to refresh advancement probabilities.
"""

import dataclasses
from typing import Literal

import numpy as np
import pandas as pd

from wc2026.models.base import ScorelineForecaster


@dataclasses.dataclass(frozen=True)
class KnockoutPolicy:
    """How drawn knockout matches are resolved in simulation.

    Extra-time goals are sampled as Poisson with the 90-minute rates scaled by
    ``et_rate_scale`` (30 minutes of play); unresolved ties go to penalties.
    """

    et_rate_scale: float = 1.0 / 3.0
    pens: Literal["coinflip", "rating_logit"] = "coinflip"


@dataclasses.dataclass(frozen=True)
class TournamentState:
    """Completed results plus remaining fixtures, as canonical frames."""

    matches: pd.DataFrame
    fifa_ranks: dict[str, int]


def simulate_tournament(
    state: TournamentState,
    model: ScorelineForecaster,
    n_sims: int,
    rng: np.random.Generator,
    policy: KnockoutPolicy,
) -> pd.DataFrame:
    """Per-team probabilities of reaching R32/R16/QF/SF/final/champion."""
    raise NotImplementedError("ships in Phase 3")
