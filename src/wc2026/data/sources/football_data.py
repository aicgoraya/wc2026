"""WC 2026 fixtures + live results from football-data.org (v4, free tier).

Implemented against the official v4 documentation
(https://docs.football-data.org/general/v4/match.html and .../overtime.html):

- ``GET /v4/competitions/WC/matches`` with the ``X-Auth-Token`` header.
- ``score.fullTime`` INCLUDES extra-time and even penalty-shootout goals when
  ``score.duration != "REGULAR"``; the 90-minute score is ``score.regularTime``
  and the extra-time-only goals are ``score.extraTime``. The parser decomposes
  accordingly so the canonical frame never mixes shootout goals into scores.
- Free tier: 10 calls/min; remaining quota is echoed in ``X-RequestsAvailable``.
"""

import datetime as dt
import json
from typing import Any

import pandas as pd

from wc2026.data.names import canonical_slug
from wc2026.data.schema import Match, MatchStatus, Stage, matches_to_frame
from wc2026.data.sources.base import RawPayload, http_get

BASE_URL = "https://api.football-data.org/v4"

_STATUS_MAP = {
    # docs lookup_tables.html: Match.status enum
    "SCHEDULED": MatchStatus.SCHEDULED,
    "TIMED": MatchStatus.SCHEDULED,
    "POSTPONED": MatchStatus.SCHEDULED,
    "SUSPENDED": MatchStatus.SCHEDULED,
    "CANCELLED": MatchStatus.SCHEDULED,
    "IN_PLAY": MatchStatus.IN_PLAY,
    "PAUSED": MatchStatus.IN_PLAY,
    "EXTRA_TIME": MatchStatus.IN_PLAY,
    "PENALTY_SHOOTOUT": MatchStatus.IN_PLAY,
    "FINISHED": MatchStatus.FINISHED,
    "AWARDED": MatchStatus.FINISHED,
}

_STAGE_MAP = {
    "GROUP_STAGE": Stage.GROUP,
    "LAST_32": Stage.R32,
    "LAST_16": Stage.R16,
    "QUARTER_FINALS": Stage.QF,
    "SEMI_FINALS": Stage.SF,
    "THIRD_PLACE": Stage.THIRD,
    "FINAL": Stage.FINAL,
}


class FootballDataSource:
    """Adapter for the football-data.org v4 World Cup matches endpoint."""

    name = "football_data"

    def __init__(self, token: str) -> None:
        self._token = token

    def fetch(self) -> RawPayload:
        """GET all WC matches (fixtures + results); one call covers the tournament."""
        return http_get(
            self.name,
            f"{BASE_URL}/competitions/WC/matches",
            headers={"X-Auth-Token": self._token},
            # header names observed live 2026-06-12 (docs table shows older names)
            keep_headers=("X-Requests-Available-Minute", "X-RequestCounter-Reset", "X-API-Version"),
        )

    def parse(self, raw: RawPayload) -> pd.DataFrame:
        """Parse to the canonical matches frame; placeholder fixtures are kept.

        Knockout fixtures whose teams are not yet decided come through with
        null team objects; those rows are dropped here (they re-appear once
        decided) — the bracket itself is owned by ``tournament.structure``.
        """
        payload = json.loads(raw.content)
        matches = [self._parse_match(m) for m in payload["matches"]]
        return matches_to_frame(m for m in matches if m is not None)

    def _parse_match(self, node: dict[str, Any]) -> Match | None:
        home_name = (node["homeTeam"] or {}).get("name")
        away_name = (node["awayTeam"] or {}).get("name")
        if not home_name or not away_name:
            return None  # undecided knockout placeholder
        status = _STATUS_MAP[node["status"]]
        goals = self._decompose_score(node["score"], status)
        group_raw: str | None = node.get("group")
        return Match(
            match_id=f"fd_{node['id']}",
            date=dt.datetime.fromisoformat(node["utcDate"]).date(),
            home_id=canonical_slug(home_name),
            away_id=canonical_slug(away_name),
            home_goals=goals["home"],
            away_goals=goals["away"],
            et_home_goals=goals["et_home"],
            et_away_goals=goals["et_away"],
            neutral=True,  # 2026 host advantage is modeled as a feature, not via this flag
            tournament="fifa_world_cup",
            stage=_STAGE_MAP[node["stage"]],
            group=group_raw.removeprefix("GROUP_") if group_raw else None,
            status=status,
            went_to_shootout=goals["shootout_winner"] is not None,
            shootout_winner_id={
                None: None,
                "HOME_TEAM": canonical_slug(home_name),
                "AWAY_TEAM": canonical_slug(away_name),
            }[goals["shootout_winner"]],
        )

    @staticmethod
    def _decompose_score(score: dict[str, Any], status: MatchStatus) -> dict[str, Any]:
        """Split the v4 score node into 90-minute / extra-time / shootout parts."""
        no_score: dict[str, Any] = {
            "home": None,
            "away": None,
            "et_home": None,
            "et_away": None,
            "shootout_winner": None,
        }
        if status is not MatchStatus.FINISHED:
            return no_score
        duration = score["duration"]
        full = score["fullTime"]
        if duration == "REGULAR":
            return {**no_score, "home": full["home"], "away": full["away"]}
        regular = score["regularTime"]
        extra = score["extraTime"]
        winner = score["winner"] if duration == "PENALTY_SHOOTOUT" else None
        if winner not in (None, "HOME_TEAM", "AWAY_TEAM"):
            raise ValueError(f"unexpected shootout winner {winner!r}")
        return {
            "home": regular["home"],
            "away": regular["away"],
            "et_home": extra["home"],
            "et_away": extra["away"],
            "shootout_winner": winner,
        }
