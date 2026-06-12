import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from wc2026.data.schema import Match, MatchStatus, matches_to_frame
from wc2026.data.store import MATCHES_DATASET, SnapshotNotFoundError, Store

UTC = dt.UTC


def make_match(match_id: str, date: dt.date, status: MatchStatus) -> Match:
    finished = status is MatchStatus.FINISHED
    return Match(
        match_id=match_id,
        date=date,
        home_id="mexico",
        away_id="canada",
        home_goals=1 if finished else None,
        away_goals=0 if finished else None,
        neutral=True,
        tournament="fifa_world_cup",
        status=status,
    )


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path)


class TestSnapshots:
    def test_roundtrip(self, store: Store) -> None:
        frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        snap = store.write_snapshot("odds", {"quotes": frame}, {"source": "test"})
        out = store.read("odds", "quotes", snap)
        pd.testing.assert_frame_equal(out, frame)

    def test_latest_picks_most_recent(self, store: Store) -> None:
        t0 = dt.datetime(2026, 6, 12, 10, 0, tzinfo=UTC)
        store.write_snapshot("odds", {"q": pd.DataFrame({"v": [1]})}, {}, now=t0)
        store.write_snapshot(
            "odds", {"q": pd.DataFrame({"v": [2]})}, {}, now=t0 + dt.timedelta(hours=1)
        )
        assert store.read("odds", "q")["v"].item() == 2
        assert len(store.snapshots("odds")) == 2

    def test_missing_dataset_raises(self, store: Store) -> None:
        with pytest.raises(SnapshotNotFoundError):
            store.latest("nope")
        with pytest.raises(SnapshotNotFoundError):
            store.read("nope", "q")

    def test_missing_frame_raises(self, store: Store) -> None:
        store.write_snapshot("odds", {"q": pd.DataFrame({"v": [1]})}, {})
        with pytest.raises(SnapshotNotFoundError):
            store.read("odds", "other")

    def test_empty_snapshot_rejected(self, store: Store) -> None:
        with pytest.raises(ValueError, match="at least one frame"):
            store.write_snapshot("odds", {}, {})

    def test_manifest(self, store: Store) -> None:
        snap = store.write_snapshot("odds", {"q": pd.DataFrame({"v": [1, 2]})}, {"k": "v"})
        manifest = store.read_manifest("odds", snap)
        assert manifest["meta"] == {"k": "v"}
        assert manifest["frames"] == {"q": 2}


class TestMatchesAsOf:
    def test_leak_guard(self, store: Store) -> None:
        matches = [
            make_match("m_past", dt.date(2026, 6, 10), MatchStatus.FINISHED),
            make_match("m_cutoff_day", dt.date(2026, 6, 12), MatchStatus.FINISHED),
            make_match("m_future", dt.date(2026, 6, 20), MatchStatus.FINISHED),
            make_match("m_unplayed", dt.date(2026, 6, 11), MatchStatus.SCHEDULED),
        ]
        store.write_snapshot(MATCHES_DATASET, {"matches": matches_to_frame(matches)}, {})

        visible = store.matches_as_of(dt.date(2026, 6, 12))

        # strictly before the cutoff: the cutoff-day match itself is excluded,
        # as are future and unfinished matches
        assert list(visible["match_id"]) == ["m_past"]

    def test_empty_when_all_future(self, store: Store) -> None:
        matches = [make_match("m1", dt.date(2026, 7, 1), MatchStatus.FINISHED)]
        store.write_snapshot(MATCHES_DATASET, {"matches": matches_to_frame(matches)}, {})
        assert store.matches_as_of(dt.date(2026, 6, 1)).empty
