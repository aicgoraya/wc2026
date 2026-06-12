"""Leak-free Elo ratings computed by replaying the historical results in date order.

Published rating tables already contain the match being predicted, so the
walk-forward harness uses these self-computed ratings; eloratings.net serves
only as a cross-check.
"""

import datetime as dt

import pandas as pd


def compute_elo(
    matches: pd.DataFrame,
    *,
    k_base: float,
    home_advantage: float,
    as_of: dt.date,
) -> pd.DataFrame:
    """Replay finished matches before ``as_of``; returns team_id → rating."""
    raise NotImplementedError("ships in Phase 1b")
