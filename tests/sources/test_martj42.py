import datetime as dt

import pytest

from wc2026.data.sources import martj42
from wc2026.data.sources.base import RawPayload

RESULTS_HEADER = "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral"
SHOOTOUTS_HEADER = "date,home_team,away_team,winner,first_shooter"


def payload(header: str, *rows: str) -> RawPayload:
    return RawPayload(
        source="martj42",
        fetched_at_utc=dt.datetime(2026, 6, 12, tzinfo=dt.UTC),
        content="\n".join([header, *rows]).encode(),
    )


def parse(results: list[str], shootouts: list[str] | None = None):  # type: ignore[no-untyped-def]
    return martj42.parse(
        payload(RESULTS_HEADER, *results), payload(SHOOTOUTS_HEADER, *(shootouts or []))
    )


class TestParse:
    def test_basic_row(self) -> None:
        frame = parse(["1998-07-12,France,Brazil,3,0,FIFA World Cup,Paris,France,FALSE"])
        row = frame.iloc[0]
        assert row["match_id"] == "mj_19980712_france_brazil"
        assert row["home_goals"] == 3 and row["away_goals"] == 0
        assert row["tournament"] == "fifa_world_cup"
        assert bool(row["neutral"]) is False
        assert row["status"] == "finished"

    def test_future_na_rows_dropped(self) -> None:
        frame = parse(
            [
                "2026-06-27,Panama,England,NA,NA,FIFA World Cup,East Rutherford,United States,TRUE",
                "1998-07-12,France,Brazil,3,0,FIFA World Cup,Paris,France,FALSE",
            ]
        )
        assert list(frame["match_id"]) == ["mj_19980712_france_brazil"]

    def test_shootout_draw_tagged(self) -> None:
        frame = parse(
            ["2022-12-18,Argentina,France,3,3,FIFA World Cup,Lusail,Qatar,TRUE"],
            ["2022-12-18,Argentina,France,Argentina,France"],
        )
        row = frame.iloc[0]
        assert bool(row["went_to_shootout"]) is True
        assert row["shootout_winner_id"] == "argentina"

    def test_shootout_unknown_winner_untagged(self) -> None:
        frame = parse(
            ["1971-11-14,South Korea,Vietnam Republic,1,1,Friendly,Seoul,South Korea,FALSE"],
            ["1971-11-14,South Korea,Vietnam Republic,NA,NA"],
        )
        assert bool(frame.iloc[0]["went_to_shootout"]) is False

    def test_aggregate_tie_shootout_untagged(self) -> None:
        # two-legged tie decided on penalties: match score is decisive, stands untagged
        row_1973 = (
            "1973-04-21,Senegal,Ghana,1,0,African Cup of Nations qualification,Dakar,Senegal,FALSE"
        )
        frame = parse([row_1973], ["1973-04-21,Senegal,Ghana,Ghana,"])
        row = frame.iloc[0]
        assert bool(row["went_to_shootout"]) is False
        assert row["home_goals"] == 1

    def test_orphan_shootout_raises(self) -> None:
        with pytest.raises(ValueError, match="no matching result row"):
            parse(
                ["1998-07-12,France,Brazil,3,0,FIFA World Cup,Paris,France,FALSE"],
                ["1999-01-01,France,Brazil,France,"],
            )

    def test_known_orphan_skipped(self) -> None:
        frame = parse(
            ["1998-07-12,France,Brazil,3,0,FIFA World Cup,Paris,France,FALSE"],
            ["2011-06-29,Saare County,Åland Islands,Åland Islands,"],
        )
        assert len(frame) == 1

    def test_exact_duplicates_collapse(self) -> None:
        frame = parse(
            [
                "2026-06-06,Gibraltar,Cayman Islands,4,1,Friendly,Gibraltar,Gibraltar,FALSE",
                "2026-06-06,Gibraltar,Cayman Islands,4,1,Friendly,Europa Point,Gibraltar,FALSE",
            ]
        )
        assert len(frame) == 1

    def test_orientation_flipped_duplicate_collapses(self) -> None:
        # real corpus case: one match entered twice with sides flipped
        frame = parse(
            [
                "1925-05-20,China,Japan,2,0,Friendly,Manila,Philippines,TRUE",
                "1925-05-20,Japan,China,0,2,Far Eastern Championship Games,Manila,Philippines,TRUE",
            ]
        )
        assert len(frame) == 1
        assert frame.iloc[0]["home_id"] == "china"

    def test_double_header_gets_suffixed_ids(self) -> None:
        frame = parse(
            [
                "1974-02-17,Tahiti,New Caledonia,2,1,Friendly,Papeete,Tahiti,FALSE",
                "1974-02-17,Tahiti,New Caledonia,1,2,Friendly,Papeete,Tahiti,FALSE",
            ]
        )
        assert sorted(frame["match_id"]) == [
            "mj_19740217_tahiti_new_caledonia",
            "mj_19740217_tahiti_new_caledonia_2",
        ]


def test_team_universe() -> None:
    frame = parse(["1998-07-12,France,Brazil,3,0,FIFA World Cup,Paris,France,FALSE"])
    assert list(martj42.team_universe(frame)["team_id"]) == ["brazil", "france"]
