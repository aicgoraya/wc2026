"""Match-level features for the GBM, built leak-free in one chronological pass.

The whole point of the GBM (Phase 5) is to use information the generative
models cannot: recent form, rest, momentum, competition importance. Every
feature for a match is computed STRICTLY from matches before that match's
date — the builder walks matches in date order and, for each one, reads the
running per-team state BEFORE folding that match's result in. A future match
therefore cannot influence a past row (property-tested).

No player/lineup features by design — they don't reliably exist for
internationals. Team-strength + form + schedule is the right level.
"""

import datetime as dt
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from wc2026.data.sources.elo_own import (
    CONTINENTAL_FINALS,
    INITIAL_RATING,
    goal_diff_multiplier,
    tournament_k,
)

FEATURE_COLUMNS: tuple[str, ...] = (
    "elo_diff",
    "elo_momentum_diff",
    "form_diff",
    "gf_diff",
    "ga_diff",
    "rest_diff",
    "log_experience_diff",
    "comp_weight",
    "is_home",
)

_FORM_WINDOW = 5
_MOMENTUM_WINDOW = 5
_HOME_ADVANTAGE = 100.0
_REST_CAP_DAYS = 60.0


def competition_weight(tournament: str) -> float:
    """Match-importance weight from the canonical tournament slug."""
    if tournament == "fifa_world_cup":
        return 1.0
    if tournament in CONTINENTAL_FINALS:
        return 0.85
    if tournament.endswith("_qualification") or "nations_league" in tournament:
        return 0.70
    if tournament == "friendly":
        return 0.30
    return 0.50


class _TeamState:
    """Running, leak-free state for one team."""

    __slots__ = ("last_played", "n_played", "rating", "rating_history", "results")

    def __init__(self) -> None:
        self.rating = INITIAL_RATING
        self.rating_history: deque[float] = deque(maxlen=_MOMENTUM_WINDOW + 1)
        self.rating_history.append(INITIAL_RATING)
        self.results: deque[tuple[int, int, int]] = deque(maxlen=_FORM_WINDOW)  # points, gf, ga
        self.last_played: dt.date | None = None
        self.n_played = 0

    def form_ppg(self) -> float:
        return float(np.mean([r[0] for r in self.results])) if self.results else 1.0

    def avg_gf(self) -> float:
        return float(np.mean([r[1] for r in self.results])) if self.results else 1.2

    def avg_ga(self) -> float:
        return float(np.mean([r[2] for r in self.results])) if self.results else 1.2

    def momentum(self) -> float:
        """Rating change over the recent window (0 if not enough history)."""
        if len(self.rating_history) < 2:
            return 0.0
        return self.rating_history[-1] - self.rating_history[0]

    def rest_days(self, date: dt.date) -> float:
        if self.last_played is None:
            return _REST_CAP_DAYS
        return min(float((date - self.last_played).days), _REST_CAP_DAYS)


def build_feature_matrix(matches: pd.DataFrame) -> pd.DataFrame:
    """One feature row per match, computed as-of each match's date (leak-free).

    Features are computed for EVERY match (finished or scheduled); the running
    team state is updated only from FINISHED matches. Returns a frame keyed by
    ``match_id`` with ``home_id``/``away_id``/``date``/``label`` plus
    ``FEATURE_COLUMNS``. ``label`` is the outcome code (0 home / 1 draw / 2
    away) for finished matches, else ``-1``.
    """
    ordered = matches.sort_values(["date", "match_id"])
    state: dict[str, _TeamState] = defaultdict(_TeamState)
    rows: list[dict[str, object]] = []

    cols = zip(
        ordered["match_id"].astype(str),
        ordered["date"],
        ordered["home_id"].astype(str),
        ordered["away_id"].astype(str),
        ordered["neutral"].astype(bool),
        ordered["tournament"].astype(str),
        ordered["status"].astype(str),
        ordered["home_goals"],
        ordered["away_goals"],
        strict=True,
    )
    for match_id, ts, home, away, neutral, tournament, status, hg, ag in cols:
        date = ts.date()
        h, a = state[home], state[away]
        advantage = 0.0 if neutral else _HOME_ADVANTAGE
        label = -1
        if status == "finished":
            hg_i, ag_i = int(hg), int(ag)
            label = 0 if hg_i > ag_i else (1 if hg_i == ag_i else 2)
        rows.append(
            {
                "match_id": match_id,
                "date": ts,
                "home_id": home,
                "away_id": away,
                "label": label,
                "elo_diff": (h.rating + advantage) - a.rating,
                "elo_momentum_diff": h.momentum() - a.momentum(),
                "form_diff": h.form_ppg() - a.form_ppg(),
                "gf_diff": h.avg_gf() - a.avg_gf(),
                "ga_diff": h.avg_ga() - a.avg_ga(),
                "rest_diff": h.rest_days(date) - a.rest_days(date),
                "log_experience_diff": np.log1p(h.n_played) - np.log1p(a.n_played),
                "comp_weight": competition_weight(tournament),
                "is_home": 0.0 if neutral else 1.0,
            }
        )
        if status == "finished":
            _update(h, a, int(hg), int(ag), neutral, tournament, date)

    frame = pd.DataFrame(rows)
    return frame.set_index("match_id")


def _update(
    h: _TeamState,
    a: _TeamState,
    hg: int,
    ag: int,
    neutral: bool,
    tournament: str,
    date: dt.date,
) -> None:
    """Fold a finished match into both teams' running state (Elo + form + rest)."""
    advantage = 0.0 if neutral else _HOME_ADVANTAGE
    expected_home = 1.0 / (1.0 + 10.0 ** (-((h.rating + advantage) - a.rating) / 400.0))
    result_home = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
    delta = (
        tournament_k(tournament)
        * goal_diff_multiplier(abs(hg - ag))
        * (result_home - expected_home)
    )
    h.rating += delta
    a.rating -= delta
    h.rating_history.append(h.rating)
    a.rating_history.append(a.rating)

    h_points = 3 if hg > ag else (1 if hg == ag else 0)
    a_points = 3 if ag > hg else (1 if hg == ag else 0)
    h.results.append((h_points, hg, ag))
    a.results.append((a_points, ag, hg))
    h.last_played = a.last_played = date
    h.n_played += 1
    a.n_played += 1
