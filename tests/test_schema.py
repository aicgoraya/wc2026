import datetime as dt

import pandas as pd
import pytest
from pydantic import ValidationError

from wc2026.data.schema import (
    MATCH_COLUMNS,
    Match,
    MatchStatus,
    OddsQuote,
    Stage,
    Team,
    matches_to_frame,
)

UTC = dt.UTC


def make_match(**overrides: object) -> Match:
    base: dict[str, object] = {
        "match_id": "m1",
        "date": dt.date(2026, 6, 11),
        "home_id": "mexico",
        "away_id": "south_africa",
        "home_goals": 2,
        "away_goals": 1,
        "neutral": False,
        "tournament": "fifa_world_cup",
        "stage": Stage.GROUP,
        "group": "A",
        "status": MatchStatus.FINISHED,
    }
    base.update(overrides)
    return Match(**base)  # type: ignore[arg-type]


class TestTeam:
    def test_valid(self) -> None:
        team = Team(team_id="cote_divoire", name="Côte d'Ivoire", fifa_code="CIV")
        assert team.team_id == "cote_divoire"

    def test_frozen(self) -> None:
        team = Team(team_id="brazil", name="Brazil")
        with pytest.raises(ValidationError):
            team.name = "Brasil"  # type: ignore[misc]

    @pytest.mark.parametrize("bad", ["", "Brazil", "são paulo", "a-b"])
    def test_bad_slug_rejected(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            Team(team_id=bad, name="x")


class TestMatch:
    def test_valid_finished(self) -> None:
        assert make_match().home_goals == 2

    def test_valid_scheduled(self) -> None:
        m = make_match(home_goals=None, away_goals=None, status=MatchStatus.SCHEDULED)
        assert m.status is MatchStatus.SCHEDULED

    def test_team_vs_itself_rejected(self) -> None:
        with pytest.raises(ValidationError, match="cannot play itself"):
            make_match(away_id="mexico")

    def test_finished_without_score_rejected(self) -> None:
        with pytest.raises(ValidationError, match="finished match"):
            make_match(home_goals=None, away_goals=None)

    def test_scheduled_with_score_rejected(self) -> None:
        with pytest.raises(ValidationError, match="scheduled match"):
            make_match(status=MatchStatus.SCHEDULED)

    def test_half_score_rejected(self) -> None:
        with pytest.raises(ValidationError, match="both"):
            make_match(away_goals=None)

    def test_negative_goals_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_match(home_goals=-1)

    def test_shootout_consistency(self) -> None:
        with pytest.raises(ValidationError, match="shootout"):
            make_match(went_to_shootout=True)
        with pytest.raises(ValidationError, match="shootout"):
            make_match(shootout_winner_id="mexico")
        with pytest.raises(ValidationError, match="one of the two"):
            make_match(
                home_goals=1, away_goals=1, went_to_shootout=True, shootout_winner_id="brazil"
            )
        ok = make_match(
            home_goals=1, away_goals=1, went_to_shootout=True, shootout_winner_id="mexico"
        )
        assert ok.shootout_winner_id == "mexico"

    def test_bad_group_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_match(group="M")


class TestOddsQuote:
    def test_valid(self) -> None:
        quote = OddsQuote(
            match_id="m1",
            bookmaker="pinnacle",
            fetched_at_utc=dt.datetime(2026, 6, 12, 10, 0, tzinfo=UTC),
            home=1.85,
            draw=3.6,
            away=4.2,
        )
        assert quote.home == 1.85

    @pytest.mark.parametrize("odds", [1.0, 0.9, 0.0, -2.0])
    def test_odds_at_most_one_rejected(self, odds: float) -> None:
        with pytest.raises(ValidationError):
            OddsQuote(
                match_id="m1",
                bookmaker="b",
                fetched_at_utc=dt.datetime(2026, 6, 12, tzinfo=UTC),
                home=odds,
                draw=3.0,
                away=3.0,
            )

    def test_naive_timestamp_rejected(self) -> None:
        with pytest.raises(ValidationError, match="UTC"):
            OddsQuote(
                match_id="m1",
                bookmaker="b",
                fetched_at_utc=dt.datetime(2026, 6, 12),  # intentionally naive
                home=2.0,
                draw=3.0,
                away=3.0,
            )


class TestMatchesToFrame:
    def test_columns_and_dtypes(self) -> None:
        frame = matches_to_frame([make_match()])
        assert tuple(frame.columns) == MATCH_COLUMNS
        assert pd.api.types.is_datetime64_any_dtype(frame["date"])
        assert frame["home_goals"].dtype == "Int64"
        assert frame.loc[0, "stage"] == "group"
        assert frame.loc[0, "status"] == "finished"

    def test_sorted_by_date_then_id(self) -> None:
        later = make_match(match_id="m2", date=dt.date(2026, 6, 13))
        frame = matches_to_frame([later, make_match()])
        assert list(frame["match_id"]) == ["m1", "m2"]

    def test_none_fields_preserved(self) -> None:
        m = make_match(
            home_goals=None, away_goals=None, status=MatchStatus.SCHEDULED, stage=None, group=None
        )
        frame = matches_to_frame([m])
        assert frame.loc[0, "home_goals"] is pd.NA
        assert frame.loc[0, "stage"] is None
