"""Pre-match 1X2 odds snapshots from the-odds-api.com (v4, free tier).

The WC sport key is discovered via ``/v4/sports`` and asserted, not hardcoded
on faith. Historical odds are paid-only, so this adapter is run on a schedule
from the moment keys exist — market coverage starts at the first snapshot.
"""

import pandas as pd

from wc2026.data.sources.base import RawPayload


class OddsApiSource:
    """Adapter for the-odds-api v4 ``h2h`` (1X2) market."""

    name = "odds_api"

    def __init__(self, api_key: str, regions: tuple[str, ...] = ("eu", "uk")) -> None:
        self._api_key = api_key
        self._regions = regions

    def fetch(self) -> RawPayload:
        """GET current h2h odds for all WC fixtures (one call, all events)."""
        raise NotImplementedError("ships in Phase 1a")

    def parse(self, raw: RawPayload) -> pd.DataFrame:
        """Parse to one OddsQuote row per (match, bookmaker)."""
        raise NotImplementedError("ships in Phase 1a")
