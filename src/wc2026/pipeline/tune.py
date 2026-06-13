"""Leak-free hyperparameter selection for Dixon-Coles.

THE RULE: hyperparameters are chosen using only data that precedes the
reported test period. The decay half-life AND the L2 shrinkage strength are
selected jointly by walk-forward RPS on an inner validation window (default
2004-2009); every candidate's fits see only pre-cutoff matches, and the
2010+ Track-A test window is never touched by the selection. The winning
pair is frozen into ``models.dixon_coles`` defaults, the full table is
written to ``results/dc_tuning.md``, and the freeze is committed BEFORE any
test-window numbers for the chosen values (verifiable from git history).

The half-life grid extends to "no decay" (``inf``) so the winner can be an
interior optimum rather than a grid boundary. Re-run with ``wc2026 tune-dc``.
"""

import datetime as dt
import itertools

import pandas as pd

from wc2026.eval.walkforward import RefitSchedule, walk_forward
from wc2026.models.dixon_coles import DixonColesForecaster

HALF_LIFE_CANDIDATES = (1460.0, 2920.0, 5840.0, float("inf"))
L2_CANDIDATES = (0.25, 0.5, 1.0, 2.0, 5.0)
VALIDATION_WINDOW = (dt.date(2004, 1, 1), dt.date(2009, 12, 31))


def select_hyperparams(
    matches: pd.DataFrame,
    half_lives: tuple[float, ...] = HALF_LIFE_CANDIDATES,
    l2s: tuple[float, ...] = L2_CANDIDATES,
    validation_window: tuple[dt.date, dt.date] = VALIDATION_WINDOW,
    schedule: RefitSchedule | None = None,
) -> tuple[tuple[float, float], pd.DataFrame]:
    """Joint walk-forward selection of (half_life_days, l2) on the inner window.

    Returns the winning pair (by RPS) and the full selection table sorted by
    RPS ascending.
    """
    schedule = schedule or RefitSchedule(every_days=30)
    rows = []
    for half_life, l2 in itertools.product(half_lives, l2s):
        model = DixonColesForecaster(half_life_days=half_life, l2=l2)

        def make_model(m: DixonColesForecaster = model) -> DixonColesForecaster:
            return m  # same instance every refit: warm starts carry over

        result = walk_forward(make_model, matches, validation_window, schedule)
        rows.append(
            {
                "half_life_days": half_life,
                "l2": l2,
                "n": len(result),
                "rps": float(result["rps"].mean()),
                "log_loss": float(result["log_loss"].mean()),
                "brier": float(result["brier"].mean()),
            }
        )
    table = pd.DataFrame(rows).sort_values("rps", ignore_index=True)
    best = table.iloc[0]
    return (float(best["half_life_days"]), float(best["l2"])), table
