"""Merging history and the live WC feed into THE canonical matches dataset.

Ownership rule: everything from WC2026 kickoff onward in the ``FIFA World
Cup`` tournament belongs to the football-data feed (it has the score
decomposition and updates faster); martj42 rows in that window are excluded
wholesale rather than deduplicated row-by-row, because martj42 records local
dates while football-data records UTC dates — exact-date dedup would silently
double-count late-UTC kickoffs.
"""

import datetime as dt
from typing import Any

import pandas as pd

WC2026_START = dt.date(2026, 6, 11)


def merge_canonical(history: pd.DataFrame, wc_fixtures: pd.DataFrame) -> pd.DataFrame:
    """History (martj42) + live WC feed → one canonical matches frame.

    Raises if the merge contains duplicate match ids or an exact duplicate of
    (date, team pair, score) — double-counted matches would poison every model
    downstream. Same-day rematches with different scores are legitimate (the
    corpus contains real double-headers) and pass.
    """
    in_wc_window = (history["tournament"] == "fifa_world_cup") & (
        history["date"] >= pd.Timestamp(WC2026_START)
    )
    merged = pd.concat([history.loc[~in_wc_window], wc_fixtures], ignore_index=True)

    if merged["match_id"].duplicated().any():
        dupes = sorted(merged["match_id"][merged["match_id"].duplicated()])[:5]
        raise ValueError(f"merge produced duplicate match ids: {dupes}")

    def oriented_key(r: "pd.Series[Any]") -> tuple[Any, ...]:
        # orientation-normalized so A 1-0 B equals B 0-1 A recorded elsewhere
        if r["home_id"] <= r["away_id"]:
            return (r["home_id"], r["away_id"], r["home_goals"], r["away_goals"])
        return (r["away_id"], r["home_id"], r["away_goals"], r["home_goals"])

    duplicated = merged.assign(key=merged.apply(oriented_key, axis=1)).duplicated(
        subset=["date", "key"]
    )
    if duplicated.any():
        sample = merged.loc[duplicated, ["date", "home_id", "away_id"]].head(5)
        raise ValueError(f"merge produced duplicate matches:\n{sample}")
    return merged.sort_values(["date", "match_id"], ignore_index=True)
