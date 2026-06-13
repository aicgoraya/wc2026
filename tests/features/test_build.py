import datetime as dt

import numpy as np
import pytest

from wc2026.data.schema import Match, MatchStatus, matches_to_frame
from wc2026.features.build import FEATURE_COLUMNS, build_feature_matrix, competition_weight


def match(
    mid: str,
    date: dt.date,
    home: str,
    away: str,
    goals: tuple[int, int] | None,
    *,
    neutral: bool = True,
    tournament: str = "friendly",
) -> Match:
    finished = goals is not None
    return Match(
        match_id=mid,
        date=date,
        home_id=home,
        away_id=away,
        home_goals=goals[0] if finished else None,
        away_goals=goals[1] if finished else None,
        neutral=neutral,
        tournament=tournament,
        status=MatchStatus.FINISHED if finished else MatchStatus.SCHEDULED,
    )


def test_competition_weight_ordering() -> None:
    assert competition_weight("fifa_world_cup") == 1.0
    assert competition_weight("copa_america") == 0.85
    assert competition_weight("fifa_world_cup_qualification") == 0.70
    assert competition_weight("friendly") == 0.30


def test_columns_and_label() -> None:
    frame = build_feature_matrix(
        matches_to_frame([match("m1", dt.date(2020, 1, 1), "a", "b", (2, 0))])
    )
    assert set(FEATURE_COLUMNS) <= set(frame.columns)
    assert frame.loc["m1", "label"] == 0  # home win
    assert frame.loc["m1", "elo_diff"] == 0.0  # both teams start equal, neutral


def test_label_encoding() -> None:
    frame = build_feature_matrix(
        matches_to_frame(
            [
                match("h", dt.date(2020, 1, 1), "a", "b", (2, 0)),
                match("d", dt.date(2020, 1, 2), "c", "e", (1, 1)),
                match("aw", dt.date(2020, 1, 3), "f", "g", (0, 3)),
            ]
        )
    )
    assert frame.loc["h", "label"] == 0
    assert frame.loc["d", "label"] == 1
    assert frame.loc["aw", "label"] == 2


def test_scheduled_match_has_no_label_but_has_features() -> None:
    frame = build_feature_matrix(
        matches_to_frame(
            [
                match("past", dt.date(2020, 1, 1), "a", "b", (3, 0)),
                match("future", dt.date(2020, 6, 1), "a", "b", None),  # scheduled
            ]
        )
    )
    assert frame.loc["future", "label"] == -1
    # 'a' beat 'b' in the past, so a's elo > b's -> elo_diff > 0 for the future game
    assert frame.loc["future", "elo_diff"] > 0


class TestLeakFreedom:
    def test_future_match_cannot_change_a_past_row(self) -> None:
        base = [
            match("m1", dt.date(2020, 1, 1), "a", "b", (1, 0)),
            match("m2", dt.date(2020, 2, 1), "a", "c", (2, 0)),
        ]
        extra = match("m3", dt.date(2020, 3, 1), "a", "b", (5, 0))  # later match
        f_without = build_feature_matrix(matches_to_frame(base))
        f_with = build_feature_matrix(matches_to_frame([*base, extra]))
        # m1 and m2 rows must be byte-identical whether or not m3 exists
        for mid in ("m1", "m2"):
            for col in FEATURE_COLUMNS:
                assert f_without.loc[mid, col] == pytest.approx(f_with.loc[mid, col])

    def test_features_use_only_prior_results(self) -> None:
        # 'a' wins three in a row; the rest-days and form must reflect only
        # matches strictly before each date.
        frame = build_feature_matrix(
            matches_to_frame(
                [
                    match("m1", dt.date(2020, 1, 1), "a", "x", (1, 0)),
                    match("m2", dt.date(2020, 1, 11), "a", "y", (1, 0)),
                ]
            )
        )
        # m1: a has no prior games -> form default 1.0, rest = cap
        assert frame.loc["m1", "form_diff"] == pytest.approx(0.0)  # both default
        # m2: a played m1 ten days earlier; rest_diff = a_rest(10) - y_rest(cap 60)
        assert frame.loc["m2", "rest_diff"] == pytest.approx(10.0 - 60.0)
        # a has a win in form now -> form_diff positive vs fresh y
        assert frame.loc["m2", "form_diff"] > 0


def test_momentum_reflects_rating_change() -> None:
    # a string of wins for 'a' should give positive elo momentum by the last game
    rng = matches_to_frame(
        [
            match(f"m{i}", dt.date(2020, 1, 1) + dt.timedelta(days=i), "a", f"opp{i}", (3, 0))
            for i in range(6)
        ]
    )
    frame = build_feature_matrix(rng)
    assert frame.loc["m5", "elo_momentum_diff"] > 0  # a's rating has been climbing
    assert np.isfinite(frame[list(FEATURE_COLUMNS)].to_numpy()).all()
