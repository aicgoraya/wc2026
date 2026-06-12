"""Parse tests against the OFFICIALLY DOCUMENTED the-odds-api v4 response shape.

Fixture JSON mirrors https://the-odds-api.com/liveapi/guides/v4/ (odds endpoint
and sports list). The sport key is discovered live, never assumed.
"""

import datetime as dt
import json
from typing import Any

import pandas as pd
import pytest

import wc2026.data.sources.odds_api as odds_api_module
from wc2026.data.sources.base import RawPayload, SourceUnavailableError
from wc2026.data.sources.odds_api import OddsApiSource, discover_sport_key


def raw(content: object) -> RawPayload:
    return RawPayload(
        source="odds_api",
        fetched_at_utc=dt.datetime(2026, 6, 12, tzinfo=dt.UTC),
        content=json.dumps(content).encode(),
    )


def event(**overrides: object) -> dict[str, object]:
    node: dict[str, object] = {
        "id": "e4cb60c1cd96813bbf67450007cb2a10",
        "sport_key": "soccer_fifa_world_cup",
        "commence_time": "2026-06-13T16:00:00Z",
        "home_team": "Netherlands",
        "away_team": "Japan",
        "bookmakers": [
            {
                "key": "pinnacle",
                "title": "Pinnacle",
                "last_update": "2026-06-12T20:00:00Z",
                "markets": [
                    {
                        "key": "h2h",
                        "last_update": "2026-06-12T20:00:00Z",
                        "outcomes": [
                            {"name": "Netherlands", "price": 1.75},
                            {"name": "Japan", "price": 5.2},
                            {"name": "Draw", "price": 3.6},
                        ],
                    }
                ],
            },
            {
                "key": "unibet_eu",
                "title": "Unibet",
                "last_update": "2026-06-12T20:00:00Z",
                "markets": [{"key": "totals", "last_update": None, "outcomes": []}],
            },
        ],
    }
    node.update(overrides)
    return node


@pytest.fixture
def source() -> OddsApiSource:
    return OddsApiSource(api_key="dummy", sport_key="soccer_fifa_world_cup")


class TestParse:
    def test_one_row_per_event_bookmaker_with_h2h(self, source: OddsApiSource) -> None:
        frame = source.parse(raw([event()]))
        assert len(frame) == 1  # the totals-only bookmaker is skipped
        row = frame.iloc[0]
        assert row["home_name"] == "Netherlands"
        assert row["home"] == 1.75 and row["draw"] == 3.6 and row["away"] == 5.2
        assert row["bookmaker"] == "pinnacle"
        assert row["commence_time"] == pd.Timestamp("2026-06-13T16:00:00Z")

    def test_unexpected_outcomes_raise(self, source: OddsApiSource) -> None:
        bad = event()
        bookmakers: Any = bad["bookmakers"]
        bookmakers[0]["markets"][0]["outcomes"][0]["name"] = "Holland"  # name mismatch
        with pytest.raises(ValueError, match="unexpected h2h outcomes"):
            source.parse(raw([bad]))

    def test_empty_response(self, source: OddsApiSource) -> None:
        frame = source.parse(raw([]))
        assert frame.empty
        assert "event_id" in frame.columns


SPORTS = [
    {"key": "soccer_fifa_world_cup", "group": "Soccer", "title": "FIFA World Cup", "active": True},
    {"key": "soccer_epl", "group": "Soccer", "title": "EPL", "active": True},
    {
        "key": "soccer_fifa_world_cup_winner",
        "group": "Soccer",
        "title": "FIFA World Cup Winner",
        "active": True,
    },
]


class TestDiscoverSportKey:
    def _patch(self, monkeypatch: pytest.MonkeyPatch, sports: list[dict[str, object]]) -> None:
        monkeypatch.setattr(odds_api_module, "http_get", lambda *args, **kwargs: raw(sports))

    def test_exactly_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch(monkeypatch, SPORTS)
        assert discover_sport_key("k") == "soccer_fifa_world_cup"

    def test_zero_candidates_raises_with_soccer_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch(monkeypatch, [s for s in SPORTS if s["key"] == "soccer_epl"])
        with pytest.raises(SourceUnavailableError, match="soccer_epl"):
            discover_sport_key("k")

    def test_ambiguous_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        extra = {
            "key": "soccer_fifa_world_cup_other",
            "group": "Soccer",
            "title": "FIFA World Cup Special",
            "active": True,
        }
        self._patch(monkeypatch, [*SPORTS, extra])
        with pytest.raises(SourceUnavailableError, match="expected exactly one"):
            discover_sport_key("k")
