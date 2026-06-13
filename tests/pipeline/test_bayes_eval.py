"""Tests for the Phase 4 comparison helpers that don't require MCMC."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from wc2026.data.schema import Match, MatchStatus, matches_to_frame
from wc2026.pipeline.bayes_eval import _sparse_split, _team_match_counts


def _frame(rows: list[tuple[str, str, dt.date]]) -> pd.DataFrame:
    matches = [
        Match(
            match_id=f"m{i}",
            date=d,
            home_id=h,
            away_id=a,
            home_goals=1,
            away_goals=0,
            neutral=True,
            tournament="friendly",
            status=MatchStatus.FINISHED,
        )
        for i, (h, a, d) in enumerate(rows)
    ]
    return matches_to_frame(matches)


class TestTeamMatchCounts:
    def test_counts_only_before_cutoff(self) -> None:
        frame = _frame(
            [
                ("a", "b", dt.date(2020, 1, 1)),
                ("a", "c", dt.date(2020, 6, 1)),
                ("a", "b", dt.date(2025, 1, 1)),  # after cutoff -> excluded
            ]
        )
        counts = _team_match_counts(frame, dt.date(2021, 1, 1), half_life_days=1e9)
        # huge half-life -> weights ~ 1; a played twice before the cutoff
        assert counts["a"] == pytest.approx(2.0, abs=1e-6)
        assert counts["b"] == pytest.approx(1.0, abs=1e-6)
        assert counts["c"] == pytest.approx(1.0, abs=1e-6)

    def test_decay_downweights_old_matches(self) -> None:
        frame = _frame(
            [
                ("a", "b", dt.date(2019, 1, 1)),  # ~2 years before cutoff
                ("a", "c", dt.date(2021, 1, 1)),  # at cutoff-ish (just before)
            ]
        )
        hl = 365.0
        counts = _team_match_counts(frame, dt.date(2021, 1, 2), half_life_days=hl)
        # the recent match weighs ~1, the ~2y-old one ~0.5**2 = 0.25
        assert counts["a"] == pytest.approx(1.0 + 0.25, abs=0.05)
        assert counts["b"] < counts["c"]  # b only in the old game


class TestSparseSplit:
    def test_terciles_and_delta(self) -> None:
        # bayes beats dc more on the sparse side
        n = 90
        dates = [dt.date(2024, 1, 1)] * n
        teams = [(f"rich{i % 3}", f"sparse{i}") for i in range(n)]
        bayes_rows = pd.DataFrame(
            {
                "date": pd.to_datetime(dates),
                "home_id": [t[0] for t in teams],
                "away_id": [t[1] for t in teams],
                "rps": np.full(n, 0.17),
            }
        )
        dc_rows = bayes_rows.assign(rps=0.18)  # dc uniformly worse here
        # history giving 'rich*' many games, 'sparse*' few
        hist_rows = []
        for i in range(3):
            for j in range(40):
                hist_rows.append((f"rich{i}", f"opp{j}", dt.date(2023, 1, 1)))
        for i in range(n):
            hist_rows.append((f"sparse{i}", "opp0", dt.date(2023, 6, 1)))
        history = _frame(hist_rows)

        out = _sparse_split(history, bayes_rows, dc_rows, None)
        assert set(out["tercile"]) <= {"sparse", "mid", "rich"}
        assert out["count"].sum() == n
        # uniform -0.01 delta everywhere in this toy case
        assert (out["mean"] < 0).all()
