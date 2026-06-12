"""Match-level features: rating diffs, recent form, rest days, competition weight.

No player/lineup features by design — they don't reliably exist for
international football.
"""

import datetime as dt

import pandas as pd


def build_features(history: pd.DataFrame, fixtures: pd.DataFrame, as_of: dt.date) -> pd.DataFrame:
    """One row per fixture; uses only information available strictly before ``as_of``."""
    raise NotImplementedError("ships in Phase 5")
