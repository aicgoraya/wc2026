"""Parse tests against the OFFICIALLY DOCUMENTED v4 response shape.

Fixture JSON mirrors https://docs.football-data.org/general/v4/match.html and
.../overtime.html (score decomposition). The live schema is additionally
asserted on the first real pull (Phase 1a live verification).
"""

import datetime as dt
import json

import pandas as pd
import pytest

from wc2026.data.sources.base import RawPayload
from wc2026.data.sources.football_data import FootballDataSource


def _match_node(**overrides: object) -> dict[str, object]:
    node: dict[str, object] = {
        "id": 500001,
        "utcDate": "2026-06-11T20:00:00Z",
        "status": "FINISHED",
        "stage": "GROUP_STAGE",
        "group": "GROUP_A",
        "homeTeam": {"id": 769, "name": "Mexico", "tla": "MEX"},
        "awayTeam": {"id": 776, "name": "South Africa", "tla": "RSA"},
        "score": {
            "winner": "HOME_TEAM",
            "duration": "REGULAR",
            "fullTime": {"home": 2, "away": 1},
            "halfTime": {"home": 1, "away": 0},
        },
    }
    node.update(overrides)
    return node


def payload(*nodes: dict[str, object]) -> RawPayload:
    content = json.dumps({"count": len(nodes), "matches": list(nodes)}).encode()
    return RawPayload(
        source="football_data",
        fetched_at_utc=dt.datetime(2026, 6, 12, tzinfo=dt.UTC),
        content=content,
    )


@pytest.fixture
def source() -> FootballDataSource:
    return FootballDataSource(token="dummy")


class TestParse:
    def test_finished_regular(self, source: FootballDataSource) -> None:
        frame = source.parse(payload(_match_node()))
        row = frame.iloc[0]
        assert row["match_id"] == "fd_500001"
        assert row["home_id"] == "mexico"
        assert row["away_id"] == "south_africa"
        assert row["home_goals"] == 2 and row["away_goals"] == 1
        assert pd.isna(row["et_home_goals"])
        assert row["stage"] == "group"
        assert row["group"] == "A"
        assert row["status"] == "finished"
        assert bool(row["neutral"]) is False  # Mexico is a 2026 host: true home game

    def test_scheduled_timed(self, source: FootballDataSource) -> None:
        node = _match_node(
            status="TIMED",
            score={"winner": None, "duration": "REGULAR", "fullTime": {"home": None, "away": None}},
        )
        frame = source.parse(payload(node))
        row = frame.iloc[0]
        assert row["status"] == "scheduled"
        assert pd.isna(row["home_goals"])

    def test_in_play_has_no_score_yet(self, source: FootballDataSource) -> None:
        node = _match_node(
            status="IN_PLAY",
            score={"winner": None, "duration": "REGULAR", "fullTime": {"home": 1, "away": 0}},
        )
        row = source.parse(payload(node)).iloc[0]
        # running scores are not final scores; canonical frame keeps them unset
        assert row["status"] == "in_play"

    def test_penalty_shootout_decomposition(self, source: FootballDataSource) -> None:
        # docs overtime.html: fullTime INCLUDES extra-time and shootout goals
        node = _match_node(
            status="FINISHED",
            stage="LAST_16",
            group=None,
            score={
                "winner": "HOME_TEAM",
                "duration": "PENALTY_SHOOTOUT",
                "fullTime": {"home": 7, "away": 6},
                "halfTime": {"home": 1, "away": 1},
                "regularTime": {"home": 1, "away": 1},
                "extraTime": {"home": 0, "away": 0},
                "penalties": {"home": 6, "away": 5},
            },
        )
        row = source.parse(payload(node)).iloc[0]
        assert row["home_goals"] == 1 and row["away_goals"] == 1  # NOT 7-6
        assert row["et_home_goals"] == 0 and row["et_away_goals"] == 0
        assert bool(row["went_to_shootout"]) is True
        assert row["shootout_winner_id"] == "mexico"
        assert row["stage"] == "r16"

    def test_extra_time_win_decomposition(self, source: FootballDataSource) -> None:
        node = _match_node(
            status="FINISHED",
            stage="QUARTER_FINALS",
            group=None,
            score={
                "winner": "AWAY_TEAM",
                "duration": "EXTRA_TIME",
                "fullTime": {"home": 1, "away": 2},
                "halfTime": {"home": 0, "away": 0},
                "regularTime": {"home": 1, "away": 1},
                "extraTime": {"home": 0, "away": 1},
            },
        )
        row = source.parse(payload(node)).iloc[0]
        assert row["home_goals"] == 1 and row["away_goals"] == 1
        assert row["et_home_goals"] == 0 and row["et_away_goals"] == 1
        assert bool(row["went_to_shootout"]) is False

    def test_undecided_knockout_placeholder_dropped(self, source: FootballDataSource) -> None:
        node = _match_node(homeTeam={"id": None, "name": None}, awayTeam={"id": None, "name": None})
        assert source.parse(payload(node)).empty

    def test_unknown_status_raises(self, source: FootballDataSource) -> None:
        node = _match_node(status="SOMETHING_NEW")
        with pytest.raises(KeyError):
            source.parse(payload(node))

    def test_non_host_match_is_neutral(self, source: FootballDataSource) -> None:
        node = _match_node(
            homeTeam={"id": 1, "name": "South Korea"}, awayTeam={"id": 2, "name": "Czechia"}
        )
        row = source.parse(payload(node)).iloc[0]
        assert bool(row["neutral"]) is True
        assert row["away_id"] == "czech_republic"  # override applied

    def test_host_listed_away_swaps_sides(self, source: FootballDataSource) -> None:
        node = _match_node(
            homeTeam={"id": 1, "name": "Paraguay"},
            awayTeam={"id": 2, "name": "United States"},
            score={
                "winner": "AWAY_TEAM",
                "duration": "REGULAR",
                "fullTime": {"home": 1, "away": 3},
                "halfTime": {"home": 0, "away": 1},
            },
        )
        row = source.parse(payload(node)).iloc[0]
        assert row["home_id"] == "united_states"  # host carries the home advantage
        assert row["away_id"] == "paraguay"
        assert row["home_goals"] == 3 and row["away_goals"] == 1  # goals swapped too
        assert bool(row["neutral"]) is False

    def test_unresolvable_name_raises(self, source: FootballDataSource) -> None:
        from wc2026.data.names import UnresolvedTeamNameError

        node = _match_node(homeTeam={"id": 1, "name": "Atlantis"})
        with pytest.raises(UnresolvedTeamNameError, match="Atlantis"):
            source.parse(payload(node))
