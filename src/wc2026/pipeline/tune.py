"""Leak-free hyperparameter selection for Dixon-Coles.

THE RULE: hyperparameters are chosen using only data that precedes the
reported test period. The decay half-life is selected by walk-forward RPS on
an inner validation window (default 2004-2009); every candidate's fits see
only pre-cutoff matches, and the 2010+ Track-A test window is never touched
by the selection. The winning value is frozen into
``models.dixon_coles.DEFAULT_HALF_LIFE_DAYS`` and documented in RESULTS.md.

Re-running the selection (``wc2026 tune-dc``) reproduces the table.
"""

import datetime as dt

import pandas as pd

from wc2026.eval.walkforward import RefitSchedule, walk_forward
from wc2026.models.dixon_coles import DixonColesForecaster

HALF_LIFE_CANDIDATES = (365.0, 730.0, 1460.0, 2920.0)
VALIDATION_WINDOW = (dt.date(2004, 1, 1), dt.date(2009, 12, 31))


def select_half_life(
    matches: pd.DataFrame,
    candidates: tuple[float, ...] = HALF_LIFE_CANDIDATES,
    validation_window: tuple[dt.date, dt.date] = VALIDATION_WINDOW,
    schedule: RefitSchedule | None = None,
) -> tuple[float, pd.DataFrame]:
    """Walk-forward RPS per candidate on the validation window; best first.

    Returns (best half-life, full selection table).
    """
    schedule = schedule or RefitSchedule(every_days=30)
    rows = []
    for half_life in candidates:
        model = DixonColesForecaster(half_life_days=half_life)

        def make_model(m: DixonColesForecaster = model) -> DixonColesForecaster:
            return m  # same instance every refit: warm starts carry over

        result = walk_forward(make_model, matches, validation_window, schedule)
        rows.append(
            {
                "half_life_days": half_life,
                "n": len(result),
                "rps": float(result["rps"].mean()),
                "log_loss": float(result["log_loss"].mean()),
                "brier": float(result["brier"].mean()),
            }
        )
    table = pd.DataFrame(rows).sort_values("rps", ignore_index=True)
    return float(table.iloc[0]["half_life_days"]), table
