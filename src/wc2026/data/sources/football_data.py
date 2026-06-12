"""WC 2026 fixtures + live results from football-data.org (v4, free tier).

Competition code ``WC``; auth via ``X-Auth-Token`` header; 10 calls/min on the
free tier. Response schema is asserted against the live API on first pull, not
assumed from documentation.
"""

import pandas as pd

from wc2026.data.sources.base import RawPayload


class FootballDataSource:
    """Adapter for the football-data.org v4 World Cup endpoints."""

    name = "football_data"

    def __init__(self, token: str) -> None:
        self._token = token

    def fetch(self) -> RawPayload:
        """GET the WC matches endpoint (fixtures + results + group labels)."""
        raise NotImplementedError("ships in Phase 1a")

    def parse(self, raw: RawPayload) -> pd.DataFrame:
        """Parse to the canonical matches frame (stage, group, status mapped)."""
        raise NotImplementedError("ships in Phase 1a")
