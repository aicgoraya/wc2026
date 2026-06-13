"""Leak-free Elo ratings computed by replaying the historical results in date order.

Published rating tables already contain the match being predicted, so the
walk-forward harness uses these self-computed ratings; eloratings.net serves
only as an external sanity cross-check.

Conventions follow World Football Elo (eloratings.net methodology):

- K by tournament importance: World Cup finals 60; continental finals and the
  Confederations Cup 50; qualifiers, Nations Leagues and other FIFA-calendar
  competitive matches 40/30; friendlies 20.
- Goal-difference multiplier: x1.5 for a 2-goal margin, x1.75 for 3,
  1.75 + (n-3)/8 for n >= 4.
- Home advantage added to the home side's rating only when ``neutral`` is
  False (host matches; the canonical frame encodes hosts as true home games).
- Shootout matches enter as the draws they were after play; extra-time goals
  are part of the recorded score where the source folds them in.

Absolute levels differ from eloratings.net (different seeding era and minor
rule details); rank ORDER is the meaningful comparison.
"""

import dataclasses
import datetime as dt
import math

import pandas as pd

CONTINENTAL_FINALS = frozenset(
    {
        "copa_america",
        "uefa_euro",
        "african_cup_of_nations",
        "afc_asian_cup",
        "gold_cup",
        "concacaf_championship",
        "oceania_nations_cup",
        "confederations_cup",
        "fifa_confederations_cup",
    }
)

INITIAL_RATING = 1500.0


def tournament_k(tournament: str) -> float:
    """World-Football-Elo style K factor from the canonical tournament slug."""
    if tournament == "fifa_world_cup":
        return 60.0
    if tournament in CONTINENTAL_FINALS:
        return 50.0
    if tournament.endswith("_qualification") or tournament in (
        "uefa_nations_league",
        "concacaf_nations_league",
    ):
        return 40.0
    if tournament == "friendly":
        return 20.0
    return 30.0


def goal_diff_multiplier(margin: int) -> float:
    """x1 for <=1 goal, x1.5 for 2, x1.75 for 3, then +1/8 per extra goal."""
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return 1.75 + (margin - 3) / 8.0


@dataclasses.dataclass(frozen=True)
class EloReplay:
    """Result of one chronological replay up to a cutoff.

    ``samples`` holds one row per replayed match with the PRE-MATCH rating
    difference (home perspective, home advantage included) and the outcome
    code (0 home / 1 draw / 2 away) — by construction every row only contains
    information available before that match's kickoff, so a link function fit
    on ``samples`` is walk-forward clean.
    """

    ratings: dict[str, float]
    n_matches: dict[str, int]
    last_played: dict[str, pd.Timestamp]
    samples: pd.DataFrame  # columns: date, xdiff, outcome


def replay(
    matches: pd.DataFrame,
    *,
    as_of: dt.date,
    home_advantage: float = 100.0,
) -> EloReplay:
    """Replay finished matches STRICTLY BEFORE ``as_of`` in date order.

    The strict cutoff is the leak guard: a rating "as of" a date never
    contains that date's results, so predicting a match with its same-day
    rating is walk-forward clean.
    """
    finished = matches["status"] == "finished"
    before = matches["date"] < pd.Timestamp(as_of)
    window = matches.loc[finished & before].sort_values(["date", "match_id"])

    ratings: dict[str, float] = {}
    counts: dict[str, int] = {}
    last_played: dict[str, pd.Timestamp] = {}
    sample_dates: list[pd.Timestamp] = []
    sample_xdiff: list[float] = []
    sample_outcome: list[int] = []

    rows = zip(
        window["home_id"],
        window["away_id"],
        window["home_goals"].astype(int),
        window["away_goals"].astype(int),
        window["neutral"],
        window["tournament"],
        window["date"],
        strict=True,
    )
    for home_id, away_id, home_goals, away_goals, neutral, tournament, date in rows:
        home = ratings.get(home_id, INITIAL_RATING)
        away = ratings.get(away_id, INITIAL_RATING)
        advantage = 0.0 if neutral else home_advantage
        xdiff = (home + advantage) - away
        expected_home = 1.0 / (1.0 + math.pow(10.0, -xdiff / 400.0))
        if home_goals > away_goals:
            result_home, outcome = 1.0, 0
        elif home_goals < away_goals:
            result_home, outcome = 0.0, 2
        else:
            result_home, outcome = 0.5, 1
        sample_dates.append(date)
        sample_xdiff.append(xdiff)
        sample_outcome.append(outcome)
        delta = (
            tournament_k(tournament)
            * goal_diff_multiplier(abs(home_goals - away_goals))
            * (result_home - expected_home)
        )
        ratings[home_id] = home + delta
        ratings[away_id] = away - delta
        for team in (home_id, away_id):
            counts[team] = counts.get(team, 0) + 1
            last_played[team] = date

    samples = pd.DataFrame({"date": sample_dates, "xdiff": sample_xdiff, "outcome": sample_outcome})
    return EloReplay(ratings=ratings, n_matches=counts, last_played=last_played, samples=samples)


def compute_elo(
    matches: pd.DataFrame,
    *,
    as_of: dt.date,
    home_advantage: float = 100.0,
) -> pd.DataFrame:
    """Rating table from a replay: (team_id, rating, n_matches, last_played), best first."""
    rep = replay(matches, as_of=as_of, home_advantage=home_advantage)
    frame = pd.DataFrame(
        {
            "team_id": list(rep.ratings),
            "rating": [rep.ratings[t] for t in rep.ratings],
            "n_matches": [rep.n_matches[t] for t in rep.ratings],
            "last_played": [rep.last_played[t] for t in rep.ratings],
        }
    )
    return frame.sort_values("rating", ascending=False, ignore_index=True)
