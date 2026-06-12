"""FIFA world rankings — a model feature and the final 2026 tiebreaker."""

import pandas as pd

from wc2026.data.sources.base import RawPayload


class FifaRankingsSource:
    """Adapter for the published FIFA ranking table."""

    name = "fifa_rankings"

    def fetch(self) -> RawPayload:
        """Pull the current ranking table."""
        raise NotImplementedError("ships in Phase 1b")

    def parse(self, raw: RawPayload) -> pd.DataFrame:
        """Parse to team_id → (rank, points)."""
        raise NotImplementedError("ships in Phase 1b")
