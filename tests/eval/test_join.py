import datetime as dt

import pandas as pd
import pytest

from wc2026.data.names import NameResolver
from wc2026.data.schema import Match, MatchStatus, Stage, matches_to_frame
from wc2026.data.store import Store
from wc2026.eval.join import (
    OddsJoinError,
    join_coverage,
    join_events_to_fixtures,
    load_all_quotes,
    unique_events,
)
from wc2026.pipeline.collect import ODDS_DATASET

RESOLVER = NameResolver(
    frozenset({"united_states", "paraguay", "mexico", "south_africa"}),
    overrides={"odds_api": {"usa": "united_states"}},
)


def fixture_row(match_id: str, date: dt.date, home: str, away: str) -> Match:
    return Match(
        match_id=match_id,
        date=date,
        home_id=home,
        away_id=away,
        neutral=False,
        tournament="fifa_world_cup",
        stage=Stage.GROUP,
        status=MatchStatus.SCHEDULED,
    )


def event(event_id: str, kickoff: str, home: str, away: str) -> dict[str, object]:
    return {
        "event_id": event_id,
        "commence_time": pd.Timestamp(kickoff),
        "home_name": home,
        "away_name": away,
    }


FIXTURES = matches_to_frame(
    [
        # late-UTC kickoff: odds say June 13 01:00Z, fixture date June 13
        fixture_row("fd_1", dt.date(2026, 6, 13), "united_states", "paraguay"),
        fixture_row("fd_2", dt.date(2026, 6, 11), "mexico", "south_africa"),
    ]
)


class TestJoin:
    def test_exact_join_with_orientation(self) -> None:
        events = pd.DataFrame(
            [
                event("e1", "2026-06-13T01:00:00Z", "USA", "Paraguay"),
                event("e2", "2026-06-11T20:00:00Z", "South Africa", "Mexico"),  # listed flipped
            ]
        )
        joined = join_events_to_fixtures(events, FIXTURES, RESOLVER)
        by_event = joined.set_index("event_id")
        assert by_event.loc["e1", "match_id"] == "fd_1"
        assert bool(by_event.loc["e1", "flipped"]) is False
        assert by_event.loc["e2", "match_id"] == "fd_2"
        assert bool(by_event.loc["e2", "flipped"]) is True

    def test_one_day_tolerance(self) -> None:
        # martj42-style local date one day earlier than the UTC timestamp
        events = pd.DataFrame([event("e1", "2026-06-14T01:00:00Z", "USA", "Paraguay")])
        joined = join_events_to_fixtures(events, FIXTURES, RESOLVER)
        assert joined.iloc[0]["match_id"] == "fd_1"

    def test_unjoinable_event_raises(self) -> None:
        events = pd.DataFrame([event("e9", "2026-07-01T01:00:00Z", "USA", "Paraguay")])
        with pytest.raises(OddsJoinError, match="0 candidate fixtures"):
            join_events_to_fixtures(events, FIXTURES, RESOLVER)

    def test_unresolvable_name_raises(self) -> None:
        events = pd.DataFrame([event("e9", "2026-06-13T01:00:00Z", "Atlantis", "Paraguay")])
        with pytest.raises(OddsJoinError, match="Atlantis"):
            join_events_to_fixtures(events, FIXTURES, RESOLVER)

    def test_ambiguous_event_raises(self) -> None:
        double = matches_to_frame(
            [
                fixture_row("fd_1", dt.date(2026, 6, 13), "united_states", "paraguay"),
                fixture_row("fd_dup", dt.date(2026, 6, 14), "united_states", "paraguay"),
            ]
        )
        events = pd.DataFrame([event("e1", "2026-06-13T12:00:00Z", "USA", "Paraguay")])
        with pytest.raises(OddsJoinError, match="2 candidate fixtures"):
            join_events_to_fixtures(events, double, RESOLVER)

    def test_coverage_collects_instead_of_raising(self) -> None:
        events = pd.DataFrame(
            [
                event("ok", "2026-06-13T01:00:00Z", "USA", "Paraguay"),
                event("bad", "2026-07-01T01:00:00Z", "USA", "Paraguay"),
            ]
        )
        report = join_coverage(events, FIXTURES, RESOLVER)
        assert report.coverage == 0.5
        assert len(report.failures) == 1


class TestLoadAllQuotes:
    def test_backfills_missing_fetched_at(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        store = Store(tmp_path)
        legacy = pd.DataFrame({"event_id": ["e1"], "home": [2.0], "draw": [3.0], "away": [4.0]})
        store.write_snapshot(
            ODDS_DATASET,
            {"quotes": legacy},
            meta={"fetched_at_utc": "2026-06-12T21:47:18+00:00"},
            now=dt.datetime(2026, 6, 12, 21, 47, tzinfo=dt.UTC),
        )
        modern = legacy.assign(fetched_at_utc=pd.Timestamp("2026-06-13T03:00:00Z"), event_id="e2")
        store.write_snapshot(
            ODDS_DATASET,
            {"quotes": modern},
            meta={},
            now=dt.datetime(2026, 6, 13, 3, 0, tzinfo=dt.UTC),
        )
        quotes = load_all_quotes(store)
        assert len(quotes) == 2
        assert quotes["fetched_at_utc"].notna().all()
        legacy_row = quotes[quotes["event_id"] == "e1"].iloc[0]
        assert legacy_row["fetched_at_utc"] == pd.Timestamp("2026-06-12T21:47:18Z")

    def test_empty_store_raises(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(FileNotFoundError):
            load_all_quotes(Store(tmp_path))


def test_unique_events() -> None:
    quotes = pd.DataFrame(
        [
            {**event("e1", "2026-06-13T01:00:00Z", "USA", "Paraguay"), "bookmaker": "a"},
            {**event("e1", "2026-06-13T01:00:00Z", "USA", "Paraguay"), "bookmaker": "b"},
        ]
    )
    events = unique_events(quotes)
    assert len(events) == 1
    assert list(events.columns) == ["event_id", "commence_time", "home_name", "away_name"]
