"""Exact 2026 group and best-thirds ranking rules.

Implemented from the official FIFA World Cup 2026 Regulations (Annex via the
cited regulations PDF). Ranking of teams level on points, in order:

    a. most points in matches among the tied teams (head-to-head)
    b. superior goal difference in those head-to-head matches
    c. most goals scored in those head-to-head matches
       -- if some teams are still level after a-c, criteria a-c are REAPPLIED
          to only those still-level teams (the head-to-head table is recomputed
          among that smaller set); if that does not separate them, continue:
    d. superior goal difference in all group matches
    e. most goals scored in all group matches
    f. highest team conduct ("fair play") score
    g/h. better position in the (progressively older) FIFA Men's World Ranking

Head-to-head comes FIRST (a-c before d-e) — this is the 2026 order and differs
from the pre-2026 World Cup order (overall GD first). Criteria f and g/h need
disciplinary records and FIFA rankings that a goal-only match model does not
produce, so the simulator passes a single deterministic ``tiebreak_key`` per
team that stands in for f+g+h (a seeded draw-of-lots). Reaching it requires two
teams identical on points, head-to-head, overall GD and overall goals — rare
enough that the approximation does not move advancement probabilities.

Functions operate on plain Python data (no pandas) so they are cheap to call
inside the Monte-Carlo inner loop.
"""

import dataclasses
from collections.abc import Iterable, Mapping, Sequence

MatchResult = tuple[str, str, int, int]
"""(home_id, away_id, home_goals, away_goals) for a played match."""


@dataclasses.dataclass(frozen=True)
class TeamStanding:
    """A team's overall record in its group."""

    team_id: str
    played: int
    points: int
    goals_for: int
    goals_against: int

    @property
    def goal_diff(self) -> int:
        """Overall goal difference."""
        return self.goals_for - self.goals_against


def standings(team_ids: Sequence[str], results: Iterable[MatchResult]) -> dict[str, TeamStanding]:
    """Overall points/goals record per team from the played matches."""
    pts = dict.fromkeys(team_ids, 0)
    gf = dict.fromkeys(team_ids, 0)
    ga = dict.fromkeys(team_ids, 0)
    played = dict.fromkeys(team_ids, 0)
    for home, away, hg, ag in results:
        gf[home] += hg
        ga[home] += ag
        gf[away] += ag
        ga[away] += hg
        played[home] += 1
        played[away] += 1
        if hg > ag:
            pts[home] += 3
        elif hg < ag:
            pts[away] += 3
        else:
            pts[home] += 1
            pts[away] += 1
    return {t: TeamStanding(t, played[t], pts[t], gf[t], ga[t]) for t in team_ids}


def _h2h_table(
    teams: Sequence[str], results: Sequence[MatchResult]
) -> dict[str, tuple[int, int, int]]:
    """Head-to-head (points, goal_diff, goals_for) among ``teams`` only."""
    member = set(teams)
    pts = dict.fromkeys(teams, 0)
    gd = dict.fromkeys(teams, 0)
    gf = dict.fromkeys(teams, 0)
    for home, away, hg, ag in results:
        if home not in member or away not in member:
            continue
        gf[home] += hg
        gf[away] += ag
        gd[home] += hg - ag
        gd[away] += ag - hg
        if hg > ag:
            pts[home] += 3
        elif hg < ag:
            pts[away] += 3
        else:
            pts[home] += 1
            pts[away] += 1
    return {t: (pts[t], gd[t], gf[t]) for t in teams}


def _break_ties(
    teams: Sequence[str],
    results: Sequence[MatchResult],
    overall: Mapping[str, TeamStanding],
    tiebreak_key: Mapping[str, float],
) -> list[str]:
    """Order teams that are level on points, best first (criteria a-h)."""
    if len(teams) == 1:
        return list(teams)

    h2h = _h2h_table(teams, results)
    # group by identical head-to-head (points, GD, goals), best first
    ordered = sorted(teams, key=lambda t: h2h[t], reverse=True)
    blocks: list[list[str]] = []
    for t in ordered:
        if blocks and h2h[blocks[-1][0]] == h2h[t]:
            blocks[-1].append(t)
        else:
            blocks.append([t])

    result: list[str] = []
    for block in blocks:
        if len(block) == 1:
            result.append(block[0])
        elif len(block) == len(teams):
            # head-to-head did not separate anyone: fall through to d-e then f/g/h
            result.extend(
                sorted(
                    block,
                    key=lambda t: (
                        overall[t].goal_diff,
                        overall[t].goals_for,
                        tiebreak_key[t],
                    ),
                    reverse=True,
                )
            )
        else:
            # a strict subset is still level: REAPPLY a-c to just this subset
            result.extend(_break_ties(block, results, overall, tiebreak_key))
    return result


def rank_group(
    team_ids: Sequence[str],
    results: Sequence[MatchResult],
    tiebreak_key: Mapping[str, float],
) -> list[str]:
    """Rank a group's teams best-first per the exact 2026 criteria.

    ``tiebreak_key`` is the deterministic final separator (stands in for
    fair-play + FIFA ranking); larger is better.
    """
    overall = standings(team_ids, results)
    by_points = sorted(team_ids, key=lambda t: overall[t].points, reverse=True)
    blocks: list[list[str]] = []
    for t in by_points:
        if blocks and overall[blocks[-1][0]].points == overall[t].points:
            blocks[-1].append(t)
        else:
            blocks.append([t])
    ranked: list[str] = []
    for block in blocks:
        ranked.extend(_break_ties(block, results, overall, tiebreak_key))
    return ranked


@dataclasses.dataclass(frozen=True)
class ThirdPlaceRecord:
    """A third-placed team's record, carrying its group for the Annex C key."""

    team_id: str
    group: str
    points: int
    goal_diff: int
    goals_for: int


def rank_thirds(
    thirds: Sequence[ThirdPlaceRecord],
    tiebreak_key: Mapping[str, float],
) -> list[ThirdPlaceRecord]:
    """Rank third-placed teams best-first: points -> GD -> goals -> f/g/h.

    No head-to-head step: third-placed teams come from different groups and
    never met. The caller takes the first eight as the qualifiers.
    """
    return sorted(
        thirds,
        key=lambda r: (r.points, r.goal_diff, r.goals_for, tiebreak_key[r.team_id]),
        reverse=True,
    )
