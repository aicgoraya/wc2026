"""Historical international results, 1872→present (martj42 GitHub dataset).

results.csv: date, home_team, away_team, home_score, away_score, tournament,
city, country, neutral; shootouts.csv: date, home_team, away_team, winner,
first_shooter. Score semantics (extra-time inclusion) get verified at ingest.
"""

import pandas as pd

from wc2026.data.sources.base import RawPayload


class Martj42Source:
    """Adapter for the martj42/international_results raw CSVs."""

    name = "martj42"

    def fetch(self) -> RawPayload:
        """Download results.csv + shootouts.csv from GitHub raw (no auth)."""
        raise NotImplementedError("ships in Phase 1b")

    def parse(self, raw: RawPayload) -> pd.DataFrame:
        """Parse to the canonical matches frame, resolving team names."""
        raise NotImplementedError("ships in Phase 1b")
