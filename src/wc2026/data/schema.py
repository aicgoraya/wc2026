"""Canonical entities and the canonical matches-frame layout.

Every source adapter parses into these types; everything downstream (features,
models, eval, tournament) consumes only the canonical form.
"""

import datetime as dt
from collections.abc import Iterable
from enum import StrEnum

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Stage(StrEnum):
    """Tournament stage of a World Cup 2026 fixture."""

    GROUP = "group"
    R32 = "r32"
    R16 = "r16"
    QF = "qf"
    SF = "sf"
    THIRD = "third"
    FINAL = "final"


class MatchStatus(StrEnum):
    """Lifecycle status of a match."""

    SCHEDULED = "scheduled"
    IN_PLAY = "in_play"
    FINISHED = "finished"


class Team(BaseModel):
    """A national team, identified everywhere by its canonical slug."""

    model_config = ConfigDict(frozen=True)

    team_id: str = Field(min_length=1, pattern=r"^[a-z0-9_]+$")
    name: str = Field(min_length=1)
    fifa_code: str | None = None


class Match(BaseModel):
    """One match — historical result, live result, or future fixture.

    ``home_id`` is the first-listed team; on neutral ground (``neutral=True``,
    the norm at a World Cup) it carries no venue advantage. Scores from the
    historical dataset may include extra time; shootout outcomes are separate
    fields so a shootout match still reads as a draw in regulation terms.
    """

    model_config = ConfigDict(frozen=True)

    match_id: str = Field(min_length=1)
    date: dt.date
    home_id: str = Field(min_length=1)
    away_id: str = Field(min_length=1)
    home_goals: int | None = Field(default=None, ge=0)
    away_goals: int | None = Field(default=None, ge=0)
    neutral: bool
    tournament: str = Field(min_length=1)
    stage: Stage | None = None
    group: str | None = Field(default=None, pattern=r"^[A-L]$")
    status: MatchStatus
    went_to_shootout: bool = False
    shootout_winner_id: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> "Match":
        if self.home_id == self.away_id:
            raise ValueError("a team cannot play itself")
        if (self.home_goals is None) != (self.away_goals is None):
            raise ValueError("goals must be both set or both unset")
        if self.status is MatchStatus.FINISHED and self.home_goals is None:
            raise ValueError("a finished match must have a score")
        if self.status is MatchStatus.SCHEDULED and self.home_goals is not None:
            raise ValueError("a scheduled match cannot have a score")
        if self.went_to_shootout != (self.shootout_winner_id is not None):
            raise ValueError("shootout flag and winner must be set together")
        if self.shootout_winner_id not in (None, self.home_id, self.away_id):
            raise ValueError("shootout winner must be one of the two teams")
        return self


class OddsQuote(BaseModel):
    """One bookmaker's decimal 1X2 prices for one match at one point in time."""

    model_config = ConfigDict(frozen=True)

    match_id: str = Field(min_length=1)
    bookmaker: str = Field(min_length=1)
    fetched_at_utc: dt.datetime
    home: float = Field(gt=1.0)
    draw: float = Field(gt=1.0)
    away: float = Field(gt=1.0)

    @field_validator("fetched_at_utc")
    @classmethod
    def _tz_aware_utc(cls, v: dt.datetime) -> dt.datetime:
        if v.tzinfo is None or v.utcoffset() != dt.timedelta(0):
            raise ValueError("fetched_at_utc must be timezone-aware UTC")
        return v


MATCH_COLUMNS: tuple[str, ...] = (
    "match_id",
    "date",
    "home_id",
    "away_id",
    "home_goals",
    "away_goals",
    "neutral",
    "tournament",
    "stage",
    "group",
    "status",
    "went_to_shootout",
    "shootout_winner_id",
)


def matches_to_frame(matches: Iterable[Match]) -> pd.DataFrame:
    """Build the canonical matches frame (``date`` as datetime64, sorted, stable columns)."""
    rows = [m.model_dump() for m in matches]
    frame = pd.DataFrame(rows, columns=list(MATCH_COLUMNS))
    frame["date"] = pd.to_datetime(frame["date"])
    frame["home_goals"] = frame["home_goals"].astype("Int64")
    frame["away_goals"] = frame["away_goals"].astype("Int64")
    for col in ("stage", "status"):
        frame[col] = frame[col].map(lambda v: None if v is None else str(v))
    return frame.sort_values(["date", "match_id"], ignore_index=True)
