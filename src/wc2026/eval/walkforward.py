"""Leak-proof walk-forward evaluation: refit on the past, predict the next matches.

The harness — not the models — controls what history each fit sees, via
``Store.matches_as_of``. Refit cadence is configurable because MCMC refits are
expensive; predictions always use the most recent fit strictly before kickoff.
"""

import dataclasses
import datetime as dt
from collections.abc import Callable

import pandas as pd

from wc2026.data.store import Store
from wc2026.models.base import Forecaster


@dataclasses.dataclass(frozen=True)
class RefitSchedule:
    """Refit every ``every_days`` days across the evaluation window."""

    every_days: int = 1


def walk_forward(
    make_model: Callable[[], Forecaster],
    eval_window: tuple[dt.date, dt.date],
    schedule: RefitSchedule,
    store: Store,
) -> pd.DataFrame:
    """One row per evaluated match: probabilities, outcome, and per-match scores."""
    raise NotImplementedError("ships in Phase 1c")
