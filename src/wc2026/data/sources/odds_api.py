"""Pre-match 1X2 odds snapshots from the-odds-api.com (v4, free tier).

Implemented against the official v4 documentation
(https://the-odds-api.com/liveapi/guides/v4/): ``apiKey`` query param,
``/v4/sports`` to list sport keys (quota-free), ``/v4/sports/{key}/odds`` with
``markets=h2h&oddsFormat=decimal`` (cost = markets x regions credits).

The World Cup sport key is DISCOVERED from ``/v4/sports`` and asserted — never
hardcoded on faith. Historical odds are paid-only, so this collector must run
on a schedule from the moment a key exists; market-benchmark coverage starts
at the first stored snapshot.

Odds rows are stored under the odds-api's own event identity (event id, team
names, kickoff). Joining to our canonical ``match_id`` happens in Phase 1b
alongside the tested name-resolution map — collection must not block on it.
"""

import json
from typing import Any

import pandas as pd

from wc2026.data.sources.base import RawPayload, SourceUnavailableError, http_get

BASE_URL = "https://api.the-odds-api.com/v4"

QUOTE_COLUMNS: tuple[str, ...] = (
    "event_id",
    "commence_time",
    "home_name",
    "away_name",
    "bookmaker",
    "market_last_update",
    "home",
    "draw",
    "away",
)


def discover_sport_key(api_key: str) -> str:
    """Find the FIFA World Cup match-odds sport key via ``/v4/sports``.

    Raises ``SourceUnavailableError`` with the candidate list if zero or
    several plausible keys match — a human decides, we don't guess.
    """
    raw = http_get(
        "odds_api",
        f"{BASE_URL}/sports",
        params={"apiKey": api_key, "all": "true"},
    )
    sports: list[dict[str, Any]] = json.loads(raw.content)
    candidates = [
        str(s["key"])
        for s in sports
        if s.get("group") == "Soccer"
        and "world cup" in s.get("title", "").lower()
        and "winner" not in s.get("title", "").lower()  # outrights market, not match odds
        and "qualifier" not in s.get("title", "").lower()
        and "women" not in s.get("title", "").lower()
    ]
    if len(candidates) != 1:
        soccer = sorted(s["key"] for s in sports if s.get("group") == "Soccer")
        raise SourceUnavailableError(
            f"odds_api: expected exactly one World Cup sport key, got {candidates};"
            f" all soccer keys: {soccer}"
        )
    return candidates[0]


class OddsApiSource:
    """Adapter for the-odds-api v4 ``h2h`` (1X2) market on the World Cup."""

    name = "odds_api"

    def __init__(
        self, api_key: str, sport_key: str, regions: tuple[str, ...] = ("eu", "uk")
    ) -> None:
        self._api_key = api_key
        self._sport_key = sport_key
        self._regions = regions

    def fetch(self) -> RawPayload:
        """GET current h2h odds for all upcoming WC events (one call, all bookmakers)."""
        # the key travels in params, which http_get never records in meta
        return http_get(
            self.name,
            f"{BASE_URL}/sports/{self._sport_key}/odds",
            params={
                "apiKey": self._api_key,
                "regions": ",".join(self._regions),
                "markets": "h2h",
                "oddsFormat": "decimal",
            },
            keep_headers=("x-requests-remaining", "x-requests-used", "x-requests-last"),
        )

    def parse(self, raw: RawPayload) -> pd.DataFrame:
        """One row per (event, bookmaker) with decimal 1X2 prices."""
        events: list[dict[str, Any]] = json.loads(raw.content)
        rows: list[dict[str, Any]] = []
        for event in events:
            home_name, away_name = event["home_team"], event["away_team"]
            for bookmaker in event.get("bookmakers", []):
                h2h = next((m for m in bookmaker["markets"] if m["key"] == "h2h"), None)
                if h2h is None:
                    continue
                prices = {o["name"]: float(o["price"]) for o in h2h["outcomes"]}
                if set(prices) != {home_name, away_name, "Draw"}:
                    raise ValueError(
                        f"odds_api: unexpected h2h outcomes {sorted(prices)} for"
                        f" {home_name} vs {away_name}"
                    )
                rows.append(
                    {
                        "event_id": event["id"],
                        "commence_time": event["commence_time"],
                        "home_name": home_name,
                        "away_name": away_name,
                        "bookmaker": bookmaker["key"],
                        "market_last_update": h2h.get("last_update"),
                        "home": prices[home_name],
                        "draw": prices["Draw"],
                        "away": prices[away_name],
                    }
                )
        frame = pd.DataFrame(rows, columns=list(QUOTE_COLUMNS))
        for col in ("commence_time", "market_last_update"):
            frame[col] = pd.to_datetime(frame[col], utc=True, format="ISO8601")
        return frame.sort_values(["commence_time", "event_id", "bookmaker"], ignore_index=True)
