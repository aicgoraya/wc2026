"""Historical international results, 1872→present (martj42 GitHub dataset).

Verified at ingest (2026-06-12): ``results.csv`` columns are date, home_team,
away_team, home_score, away_score, tournament, city, country, neutral; rows
with NA scores are future fixtures (dropped here). Dates are LOCAL dates,
unlike football-data's UTC dates — WC2026 rows are therefore excluded by
tournament+date window at merge time, never by exact-date dedup. Scores fold
extra time in (no decomposition available, ``et_*`` stay None); shootouts live
in ``shootouts.csv`` and shootout matches read as draws, which the canonical
``Match`` validator enforces.

This source defines the canonical team-id universe: ids are the slugs of its
team names (it has no resolver to fail against — every other source resolves
INTO this namespace).
"""

import datetime as dt
import io

import pandas as pd

from wc2026.data.names import canonical_slug
from wc2026.data.schema import Match, MatchStatus, matches_to_frame
from wc2026.data.sources.base import RawPayload, http_get

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
SHOOTOUTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv"
)

KNOWN_ORPHAN_SHOOTOUTS = frozenset(
    {
        # upstream defect verified 2026-06-12: this Island Games match has a
        # shootouts.csv row but no results.csv row; any OTHER orphan still raises
        ("2011-06-29", "Saare County", "Åland Islands"),
    }
)


def fetch_results() -> RawPayload:
    """Download results.csv from GitHub raw (no auth)."""
    return http_get("martj42", RESULTS_URL)


def fetch_shootouts() -> RawPayload:
    """Download shootouts.csv from GitHub raw (no auth)."""
    return http_get("martj42", SHOOTOUTS_URL)


def parse(results_raw: RawPayload, shootouts_raw: RawPayload) -> pd.DataFrame:
    """Parse both CSVs into the canonical matches frame.

    Raises on: shootout rows that match no result row, duplicate match ids,
    or any row the canonical ``Match`` validation rejects (e.g. a recorded
    shootout after a non-draw) — bad data is surfaced, never dropped silently.
    """
    results = pd.read_csv(io.BytesIO(results_raw.content))
    shootouts = pd.read_csv(io.BytesIO(shootouts_raw.content))

    results = results.dropna(subset=["home_score", "away_score"])  # future fixtures
    # Upstream double-entries collapse: same (date, teams, result), including
    # orientation-flipped entries (e.g. 1925 China 2-0 Japan recorded again as
    # Japan 0-2 China under another tournament label). Same-day rematches with
    # different results are real and get suffixed ids instead.
    flip = results["home_team"] > results["away_team"]
    oriented = pd.DataFrame(
        {
            "date": results["date"],
            "t1": results["home_team"].where(~flip, results["away_team"]),
            "t2": results["away_team"].where(~flip, results["home_team"]),
            "g1": results["home_score"].where(~flip, results["away_score"]),
            "g2": results["away_score"].where(~flip, results["home_score"]),
        }
    )
    results = results.loc[~oriented.duplicated()]

    def text_cols(frame: pd.DataFrame, *cols: str) -> list[tuple[str, ...]]:
        return list(zip(*(frame[c].astype(str) for c in cols), strict=True))

    result_keys = set(text_cols(results, "date", "home_team", "away_team"))
    shootout_by_key: dict[tuple[str, ...], str | None] = {}
    for key, winner in zip(
        text_cols(shootouts, "date", "home_team", "away_team"),
        shootouts["winner"],
        strict=True,
    ):
        if key not in result_keys:
            if key in KNOWN_ORPHAN_SHOOTOUTS:
                continue
            raise ValueError(f"martj42: shootout row {key} has no matching result row")
        shootout_by_key[key] = None if pd.isna(winner) else str(winner)

    matches = []
    seen_ids: dict[str, int] = {}
    rows = zip(
        text_cols(results, "date", "home_team", "away_team", "tournament"),
        results["home_score"].astype(int),
        results["away_score"].astype(int),
        results["neutral"].astype(bool),
        strict=True,
    )
    for (date_str, home_name, away_name, tournament), home_score, away_score, neutral in rows:
        home_id = canonical_slug(home_name)
        away_id = canonical_slug(away_name)
        date = dt.date.fromisoformat(date_str)
        base_id = f"mj_{date:%Y%m%d}_{home_id}_{away_id}"
        seen_ids[base_id] = seen_ids.get(base_id, 0) + 1
        match_id = base_id if seen_ids[base_id] == 1 else f"{base_id}_{seen_ids[base_id]}"
        winner_name = shootout_by_key.get((date_str, home_name, away_name))
        if winner_name is not None and home_score != away_score:
            # two-legged tie decided on penalties after a level AGGREGATE (37
            # such rows, verified 2026-06-12): the shootout decided the tie,
            # not this match — the decisive match score stands untagged
            winner_name = None
        # a shootout row with unknown winner also stays an untagged (correct) draw
        matches.append(
            Match(
                match_id=match_id,
                date=date,
                home_id=home_id,
                away_id=away_id,
                home_goals=int(home_score),
                away_goals=int(away_score),
                neutral=bool(neutral),
                tournament=canonical_slug(tournament),
                status=MatchStatus.FINISHED,
                went_to_shootout=winner_name is not None,
                shootout_winner_id=canonical_slug(winner_name) if winner_name else None,
            )
        )

    frame = matches_to_frame(matches)
    duplicate_ids = frame["match_id"][frame["match_id"].duplicated()]
    if not duplicate_ids.empty:
        raise ValueError(f"martj42: duplicate match ids {sorted(duplicate_ids)[:5]}")
    return frame


def team_universe(frame: pd.DataFrame) -> pd.DataFrame:
    """The canonical (team_id) universe defined by an ingested martj42 frame."""
    ids = sorted(set(frame["home_id"]) | set(frame["away_id"]))
    return pd.DataFrame({"team_id": ids})
