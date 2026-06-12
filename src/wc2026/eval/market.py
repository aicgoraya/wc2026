"""The betting market as a benchmark forecaster: de-vigged odds, scored like a model.

CLOSING-LINE PROXY (locked methodology): historical closing odds are paywalled
on the free tier, so the benchmark uses the LAST stored snapshot strictly
before each kickoff as its closing-line proxy. Collection runs every 6 hours,
so the proxy trails the true closing line by at most ~6h; that staleness is a
disclosed limitation, applied uniformly to every match and every model being
compared. See ``closing_quotes``.

BENCHMARK LINE (locked methodology): the benchmark probability is NOT a
median over all ~40 books — recreational books shade prices for marketing and
would dilute the benchmark. Each book is de-vigged separately, then:

1. the preferred sharp book's line is used when present (default: Pinnacle,
   the conventional sharp benchmark in the literature);
2. else the mean de-vigged line across the sharp set present for that match
   (default: the betting exchanges, whose books are near vig-free);
3. else the mean de-vigged line of the ``consensus_size`` lowest-overround
   books quoting the match.
"""

import dataclasses
import datetime as dt
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import brentq

FloatArray = npt.NDArray[np.float64]


def overround(odds: FloatArray) -> FloatArray:
    """Booksum minus 1 for (n, 3) decimal odds; the bookmaker's margin."""
    result: FloatArray = (1.0 / np.asarray(odds, dtype=np.float64)).sum(axis=-1) - 1.0
    return result


def devig_proportional(odds: FloatArray) -> FloatArray:
    """Normalize inverse odds to sum to 1; (n, 3) odds → (n, 3) probabilities."""
    q = 1.0 / np.asarray(odds, dtype=np.float64)
    result: FloatArray = q / q.sum(axis=-1, keepdims=True)
    return result


def devig_shin(odds: FloatArray) -> FloatArray:
    """Shin-method de-vig; shrinks longshots harder than proportional.

    Models a fraction z of insider money (Shin 1992): implied probabilities
    are ``p_i = (sqrt(z^2 + 4(1-z) q_i^2 / Q) - z) / (2(1-z))`` with z chosen
    per match so the probabilities sum to 1.
    """
    q = np.atleast_2d(np.asarray(odds, dtype=np.float64))
    q = 1.0 / q
    out = np.empty_like(q)
    for i, row in enumerate(q):
        total = row.sum()
        if total <= 1.0:  # no margin: proportional is exact and z=0
            out[i] = row / total
            continue

        def prob_sum(z: float, row: FloatArray = row, total: float = total) -> float:
            p = (np.sqrt(z * z + 4.0 * (1.0 - z) * row * row / total) - z) / (2.0 * (1.0 - z))
            return float(p.sum() - 1.0)

        z_star = brentq(prob_sum, 0.0, 0.5)
        p = (np.sqrt(z_star**2 + 4.0 * (1.0 - z_star) * row * row / total) - z_star) / (
            2.0 * (1.0 - z_star)
        )
        out[i] = p / p.sum()
    return out.reshape(np.shape(odds))


@dataclasses.dataclass(frozen=True)
class BenchmarkPolicy:
    """Which line is THE market benchmark (see module docstring; configurable)."""

    preferred_books: tuple[str, ...] = ("pinnacle",)
    sharp_books: tuple[str, ...] = ("matchbook", "betfair_ex_eu", "betfair_ex_uk", "smarkets")
    consensus_size: int = 5
    devig: Literal["proportional", "shin"] = "proportional"
    min_lead: dt.timedelta = dt.timedelta(0)

    def devig_fn(self, odds: FloatArray) -> FloatArray:
        """The configured de-vig applied to (n, 3) odds."""
        return devig_proportional(odds) if self.devig == "proportional" else devig_shin(odds)


def closing_quotes(quotes: pd.DataFrame, policy: BenchmarkPolicy) -> pd.DataFrame:
    """Select the closing-line proxy quotes from the collected time series.

    Per (event, bookmaker): the latest quote fetched strictly before kickoff
    minus ``policy.min_lead``. Input may span many snapshots.
    """
    eligible = quotes[quotes["fetched_at_utc"] < quotes["commence_time"] - policy.min_lead]
    return (
        eligible.sort_values("fetched_at_utc")
        .groupby(["event_id", "bookmaker"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


def benchmark_probs(closing: pd.DataFrame, policy: BenchmarkPolicy) -> pd.DataFrame:
    """One benchmark probability row per event from closing quotes.

    Returns columns: event_id, commence_time, home_name, away_name,
    p_home, p_draw, p_away, benchmark_source, n_books.
    """
    rows = []
    for event_id, event_quotes in closing.groupby("event_id"):
        odds = event_quotes[["home", "draw", "away"]].to_numpy(dtype=np.float64)
        probs = policy.devig_fn(odds)
        books = event_quotes["bookmaker"].to_numpy()

        preferred = np.isin(books, policy.preferred_books)
        sharp = np.isin(books, policy.sharp_books)
        if preferred.any():
            chosen = probs[preferred][:1]
            source = str(books[preferred][0])
        elif sharp.any():
            chosen = probs[sharp]
            source = f"sharp_consensus(n={int(sharp.sum())})"
        else:
            margin = overround(odds)
            take = np.argsort(margin)[: policy.consensus_size]
            chosen = probs[take]
            source = f"low_vig_consensus(n={len(take)})"

        p = chosen.mean(axis=0)
        first = event_quotes.iloc[0]
        rows.append(
            {
                "event_id": event_id,
                "commence_time": first["commence_time"],
                "home_name": first["home_name"],
                "away_name": first["away_name"],
                "p_home": p[0],
                "p_draw": p[1],
                "p_away": p[2],
                "benchmark_source": source,
                "n_books": len(event_quotes),
            }
        )
    return pd.DataFrame(rows)


class MarketForecaster:
    """The benchmark line as a ``Forecaster`` on the eval scoreboard.

    Wiring to canonical ``match_id`` (odds-event ↔ fixture join via the name
    resolver) lands with the eval harness in Phase 1c.
    """

    name = "market"

    def __init__(self, policy: BenchmarkPolicy | None = None) -> None:
        self._policy = policy or BenchmarkPolicy()
