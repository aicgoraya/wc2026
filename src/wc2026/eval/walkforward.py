"""Leak-proof walk-forward evaluation: refit on the past, predict the future.

THE guarantee: every prediction comes from a model fit at a cutoff at or
before that match's date, and the fit only ever receives matches STRICTLY
before the cutoff — the harness slices the frame itself (defense in depth:
models also receive ``as_of`` and are expected to enforce the same rule).

Refit cadence is configurable because some models are expensive to refit;
a coarser cadence only makes models slightly stale, never leaky.
"""

import dataclasses
import datetime as dt
from collections.abc import Callable

import numpy as np
import pandas as pd

from wc2026.data.schema import Stage
from wc2026.eval import scoring
from wc2026.models.base import Fixture, Forecaster


@dataclasses.dataclass(frozen=True)
class RefitSchedule:
    """Refit every ``every_days`` days across the evaluation window."""

    every_days: int = 30


def walk_forward(
    make_model: Callable[[], Forecaster],
    matches: pd.DataFrame,
    eval_window: tuple[dt.date, dt.date],
    schedule: RefitSchedule | None = None,
) -> pd.DataFrame:
    """Evaluate a model over finished matches in ``eval_window``.

    Returns one row per evaluated match: identifiers, the predicted
    (p_home, p_draw, p_away), the outcome code, and per-match rps /
    log_loss / brier.
    """
    schedule = schedule or RefitSchedule()
    start, end = eval_window
    finished = matches[matches["status"] == "finished"].sort_values(["date", "match_id"])
    targets = finished[
        (finished["date"] >= pd.Timestamp(start)) & (finished["date"] <= pd.Timestamp(end))
    ]

    rows: list[dict[str, object]] = []
    cutoff = pd.Timestamp(start)
    step = pd.Timedelta(days=schedule.every_days)
    model: Forecaster | None = None

    target_rows = zip(
        targets["match_id"].astype(str),
        targets["date"],
        targets["home_id"].astype(str),
        targets["away_id"].astype(str),
        targets["neutral"].astype(bool),
        targets["tournament"].astype(str),
        targets["stage"],
        targets["home_goals"].astype(int),
        targets["away_goals"].astype(int),
        strict=True,
    )
    for match_id, date, home_id, away_id, neutral, tournament, stage, hg, ag in target_rows:
        if model is None or date >= cutoff + step:
            while model is not None and date >= cutoff + step:
                cutoff = cutoff + step
            model = make_model()
            history = finished[finished["date"] < cutoff]  # the harness-level leak guard
            model.fit(history, as_of=cutoff.date())
        fixture = Fixture(
            home_id=home_id,
            away_id=away_id,
            date=date.date(),
            neutral=neutral,
            stage=Stage(str(stage)) if stage else None,
        )
        probs = model.predict(fixture).validated()
        rows.append(
            {
                "match_id": match_id,
                "date": date,
                "home_id": home_id,
                "away_id": away_id,
                "tournament": tournament,
                "neutral": neutral,
                "model": model.name,
                "fit_cutoff": cutoff,
                "p_home": probs.home,
                "p_draw": probs.draw,
                "p_away": probs.away,
                "home_goals": int(hg),
                "away_goals": int(ag),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["outcome"] = scoring.outcome_codes(
        frame["home_goals"].to_numpy(), frame["away_goals"].to_numpy()
    )
    probs_arr = frame[["p_home", "p_draw", "p_away"]].to_numpy(dtype=np.float64)
    outcomes_arr = frame["outcome"].to_numpy(dtype=np.int64)
    frame["rps"] = scoring.rps(probs_arr, outcomes_arr)
    frame["log_loss"] = scoring.log_loss(probs_arr, outcomes_arr)
    frame["brier"] = scoring.brier(probs_arr, outcomes_arr)
    return frame
