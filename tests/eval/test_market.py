import datetime as dt

import numpy as np
import pandas as pd
import pytest

from wc2026.eval.market import (
    BenchmarkPolicy,
    benchmark_probs,
    closing_quotes,
    devig_proportional,
    devig_shin,
    overround,
)

UTC = dt.UTC


class TestDevig:
    def test_proportional_hand_computed(self) -> None:
        # odds (1.8, 3.6, 4.5): q = (.5556, .2778, .2222), booksum 1.0556
        probs = devig_proportional(np.array([[1.8, 3.6, 4.5]]))
        np.testing.assert_allclose(probs, [[0.52632, 0.26316, 0.21053]], atol=1e-5)

    def test_vig_free_odds_unchanged(self) -> None:
        probs = devig_proportional(np.array([[2.0, 4.0, 4.0]]))
        np.testing.assert_allclose(probs, [[0.5, 0.25, 0.25]])

    def test_shin_sums_to_one(self) -> None:
        probs = devig_shin(np.array([[1.5, 4.2, 7.0], [2.1, 3.3, 3.6]]))
        np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0], atol=1e-9)

    def test_shin_shrinks_longshots_more_than_proportional(self) -> None:
        odds = np.array([[1.2, 6.0, 15.0]])  # heavy favourite + longshot, real vig
        shin = devig_shin(odds)[0]
        prop = devig_proportional(odds)[0]
        assert shin[2] < prop[2]  # longshot shrunk harder
        assert shin[0] > prop[0]  # favourite keeps more probability

    def test_shin_equals_proportional_when_vig_free(self) -> None:
        odds = np.array([[2.0, 4.0, 4.0]])
        np.testing.assert_allclose(devig_shin(odds), devig_proportional(odds), atol=1e-9)

    def test_overround(self) -> None:
        assert overround(np.array([[2.0, 4.0, 4.0]])) == pytest.approx(0.0)
        assert overround(np.array([[1.8, 3.6, 4.5]]))[0] == pytest.approx(0.05556, abs=1e-4)


def quote_row(
    bookmaker: str,
    odds: tuple[float, float, float],
    *,
    event_id: str = "e1",
    fetched: dt.datetime = dt.datetime(2026, 6, 13, 12, 0, tzinfo=UTC),
    kickoff: dt.datetime = dt.datetime(2026, 6, 13, 16, 0, tzinfo=UTC),
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "commence_time": pd.Timestamp(kickoff),
        "home_name": "Netherlands",
        "away_name": "Japan",
        "bookmaker": bookmaker,
        "fetched_at_utc": pd.Timestamp(fetched),
        "market_last_update": pd.Timestamp(fetched),
        "home": odds[0],
        "draw": odds[1],
        "away": odds[2],
    }


class TestClosingQuotes:
    def test_latest_pre_kickoff_snapshot_wins(self) -> None:
        early = quote_row("pinnacle", (2.0, 3.0, 4.0), fetched=dt.datetime(2026, 6, 12, tzinfo=UTC))
        late = quote_row(
            "pinnacle", (1.9, 3.2, 4.4), fetched=dt.datetime(2026, 6, 13, 15, 30, tzinfo=UTC)
        )
        in_play = quote_row(
            "pinnacle", (1.1, 8.0, 20.0), fetched=dt.datetime(2026, 6, 13, 16, 30, tzinfo=UTC)
        )
        closing = closing_quotes(pd.DataFrame([early, late, in_play]), BenchmarkPolicy())
        assert len(closing) == 1
        assert closing.iloc[0]["home"] == 1.9  # latest strictly-before-kickoff quote

    def test_min_lead_pushes_selection_back(self) -> None:
        early = quote_row("pinnacle", (2.0, 3.0, 4.0), fetched=dt.datetime(2026, 6, 12, tzinfo=UTC))
        late = quote_row(
            "pinnacle", (1.9, 3.2, 4.4), fetched=dt.datetime(2026, 6, 13, 15, 30, tzinfo=UTC)
        )
        policy = BenchmarkPolicy(min_lead=dt.timedelta(hours=1))
        closing = closing_quotes(pd.DataFrame([early, late]), policy)
        assert closing.iloc[0]["home"] == 2.0

    def test_per_bookmaker_independence(self) -> None:
        rows = [
            quote_row("pinnacle", (2.0, 3.0, 4.0)),
            quote_row("smarkets", (2.1, 3.1, 3.9)),
        ]
        closing = closing_quotes(pd.DataFrame(rows), BenchmarkPolicy())
        assert set(closing["bookmaker"]) == {"pinnacle", "smarkets"}


class TestBenchmarkProbs:
    def test_pinnacle_preferred_when_present(self) -> None:
        rows = [
            quote_row("pinnacle", (2.0, 4.0, 4.0)),
            quote_row("recreational_book", (1.5, 3.0, 3.0)),
        ]
        bench = benchmark_probs(pd.DataFrame(rows), BenchmarkPolicy())
        row = bench.iloc[0]
        assert row["benchmark_source"] == "pinnacle"
        assert row["p_home"] == pytest.approx(0.5)  # pinnacle line, de-vigged; other book ignored

    def test_sharp_consensus_fallback(self) -> None:
        rows = [
            quote_row("smarkets", (2.0, 4.0, 4.0)),
            quote_row("matchbook", (2.2, 3.8, 3.8)),
            quote_row("recreational_book", (1.5, 3.0, 3.0)),
        ]
        bench = benchmark_probs(pd.DataFrame(rows), BenchmarkPolicy())
        row = bench.iloc[0]
        assert row["benchmark_source"] == "sharp_consensus(n=2)"
        expected = (0.5 + 1 / 2.2 / (1 / 2.2 + 2 / 3.8)) / 2
        assert row["p_home"] == pytest.approx(expected, abs=1e-9)

    def test_low_vig_consensus_last_resort(self) -> None:
        rows = [
            quote_row("book_sharp_vig", (2.0, 4.0, 4.0)),  # overround 0
            quote_row("book_mid_vig", (1.9, 3.8, 3.8)),
            quote_row("book_heavy_vig", (1.5, 3.0, 3.0)),
        ]
        bench = benchmark_probs(pd.DataFrame(rows), BenchmarkPolicy(consensus_size=2))
        row = bench.iloc[0]
        assert row["benchmark_source"] == "low_vig_consensus(n=2)"
        assert row["n_books"] == 3

    def test_probs_sum_to_one(self) -> None:
        rows = [quote_row("pinnacle", (1.85, 3.6, 4.2))]
        row = benchmark_probs(pd.DataFrame(rows), BenchmarkPolicy()).iloc[0]
        assert row["p_home"] + row["p_draw"] + row["p_away"] == pytest.approx(1.0)
