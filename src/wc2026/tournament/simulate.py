"""Monte-Carlo tournament simulation, conditioned on completed matches.

Given a fitted ``ScorelineForecaster`` and the canonical matches frame (with the
real group results so far), simulate the rest of the World Cup forward N times
and report each team's probability of reaching every stage and winning the cup.

Design:

- **Full scorelines, not 1X2.** Group tiebreakers need goal difference and goals
  scored, so every unplayed match is sampled as an actual ``(home, away)`` goal
  pair from the model's scoreline grid. Completed matches are fixed to their real
  result (the conditioning).
- **Exact 2026 group ranking** (``tournament.ranking``): head-to-head first, then
  overall GD/goals, with a seeded lot standing in for the unsimulatable
  fair-play/FIFA-ranking steps.
- **Real Annex C allocation** (``tournament.annex_c``) maps the eight qualifying
  thirds to their Round-of-32 slots.
- **Knockouts** are played down the published bracket
  (``tournament.structure``). A drawn knockout is resolved by extra time
  (reduced-rate Poisson over 30 minutes, from the tie's own scoring rates) and
  then penalties (near-coin-flip with a small documented tilt to the favourite).
  Extra time and penalties are NEVER folded into the 90-minute goal model.

The knockout rounds are vectorised across all N sims at once; only the group
ranking + third-place allocation runs per sim.
"""

import dataclasses
import datetime as dt

import numpy as np
import numpy.typing as npt
import pandas as pd

from wc2026.models.base import Fixture, ScorelineForecaster
from wc2026.tournament.annex_c import r32_assignment
from wc2026.tournament.ranking import ThirdPlaceRecord, rank_group, rank_thirds, standings
from wc2026.tournament.structure import BRACKET, GROUPS

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]

STAGES: tuple[str, ...] = (
    "reach_r32",
    "reach_r16",
    "reach_qf",
    "reach_sf",
    "reach_final",
    "champion",
)


@dataclasses.dataclass(frozen=True)
class KnockoutPolicy:
    """How drawn knockout matches are resolved.

    Extra-time goals are sampled as independent Poisson with each side's
    90-minute scoring rate scaled by ``et_minutes / 90``. If still level,
    penalties go to the favourite (higher 90-minute win probability) with
    probability ``pen_favorite_winprob`` - a near-coin-flip, the small tilt
    documenting that shootouts only marginally favour the better side.
    """

    et_minutes: float = 30.0
    pen_favorite_winprob: float = 0.52

    @property
    def et_scale(self) -> float:
        """Fraction of a 90-minute scoring rate expected over extra time."""
        return self.et_minutes / 90.0


@dataclasses.dataclass(frozen=True)
class _GroupMatch:
    home: str
    away: str
    home_goals: IntArray  # (n_sims,) - constant for completed, sampled otherwise
    away_goals: IntArray


@dataclasses.dataclass(frozen=True)
class _Groups:
    teams_by_group: dict[str, tuple[str, ...]]
    matches_by_group: dict[str, list[_GroupMatch]]


def _sample_scorelines(matrix: FloatArray, u: FloatArray, k: int) -> tuple[IntArray, IntArray]:
    """Inverse-CDF sample of (home, away) goals from a scoreline grid."""
    cdf = matrix.ravel().cumsum()
    flat = np.clip((cdf[None, :] < u[:, None]).sum(axis=1), 0, k * k - 1)
    return flat // k, flat % k


def _build_groups(
    matches: pd.DataFrame,
    model: ScorelineForecaster,
    n_sims: int,
    rng: np.random.Generator,
    today: dt.date,
) -> _Groups:
    """Assemble each group's teams and per-match goal arrays (sampled/fixed)."""
    wc = matches[(matches["tournament"] == "fifa_world_cup") & (matches["stage"] == "group")]
    teams_by_group: dict[str, tuple[str, ...]] = {}
    matches_by_group: dict[str, list[_GroupMatch]] = {}
    for group in GROUPS:
        rows = wc[wc["group"] == group]
        teams = sorted(set(rows["home_id"]) | set(rows["away_id"]))
        teams_by_group[group] = tuple(teams)
        group_matches = []
        for r in rows.itertuples(index=False):
            if r.status == "finished":
                hg = np.full(n_sims, int(r.home_goals), dtype=np.int64)  # type: ignore[arg-type]
                ag = np.full(n_sims, int(r.away_goals), dtype=np.int64)  # type: ignore[arg-type]
            else:
                fixture = Fixture(str(r.home_id), str(r.away_id), today, neutral=bool(r.neutral))
                matrix = model.predict_scoreline(fixture).matrix
                hg, ag = _sample_scorelines(matrix, rng.random(n_sims), matrix.shape[0])
            group_matches.append(_GroupMatch(str(r.home_id), str(r.away_id), hg, ag))
        matches_by_group[group] = group_matches
    return _Groups(teams_by_group, matches_by_group)


def _precompute_ko_grids(
    teams: list[str], model: ScorelineForecaster, today: dt.date
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Per ordered team pair: flat-CDF, expected home/away goals, and P(side_a wins in 90)."""
    n = len(teams)
    sample = model.predict_scoreline(Fixture(teams[0], teams[1], today, neutral=True)).matrix
    k = sample.shape[0]
    cdf = np.zeros((n, n, k * k))
    eg_home = np.zeros((n, n))
    eg_away = np.zeros((n, n))
    reg_winprob = np.zeros((n, n))
    support = np.arange(k)
    for i, ti in enumerate(teams):
        for j, tj in enumerate(teams):
            if i == j:
                continue
            m = model.predict_scoreline(Fixture(ti, tj, today, neutral=True)).matrix
            cdf[i, j] = m.ravel().cumsum()
            eg_home[i, j] = float((support * m.sum(axis=1)).sum())
            eg_away[i, j] = float((support * m.sum(axis=0)).sum())
            reg_winprob[i, j] = float(np.tril(m, -1).sum())
    return cdf, eg_home, eg_away, reg_winprob


class _KnockoutEngine:
    """Vectorised play of one knockout match across all sims."""

    def __init__(
        self,
        cdf: FloatArray,
        eg_home: FloatArray,
        eg_away: FloatArray,
        reg_winprob: FloatArray,
        k: int,
        policy: KnockoutPolicy,
        rng: np.random.Generator,
    ) -> None:
        self._cdf = cdf
        self._eg_home = eg_home
        self._eg_away = eg_away
        self._reg = reg_winprob
        self._k = k
        self._policy = policy
        self._rng = rng

    def play(self, a: IntArray, b: IntArray) -> tuple[IntArray, IntArray]:
        """Return (winner_idx, loser_idx) for each sim given the two team-index arrays."""
        n = len(a)
        cdf = self._cdf[a, b]
        u = self._rng.random(n)
        flat = np.clip((cdf < u[:, None]).sum(axis=1), 0, self._k * self._k - 1)
        hg = flat // self._k
        ag = flat % self._k
        a_win = hg > ag
        b_win = ag > hg

        level = ~(a_win | b_win)
        if level.any():
            eh = self._rng.poisson(self._eg_home[a, b] * self._policy.et_scale)
            ea = self._rng.poisson(self._eg_away[a, b] * self._policy.et_scale)
            hg2 = hg + np.where(level, eh, 0)
            ag2 = ag + np.where(level, ea, 0)
            a_win |= level & (hg2 > ag2)
            b_win |= level & (ag2 > hg2)

        level = ~(a_win | b_win)
        if level.any():
            fav_is_a = self._reg[a, b] >= self._reg[b, a]
            fav_wins = self._rng.random(n) < self._policy.pen_favorite_winprob
            a_pen = np.where(fav_is_a, fav_wins, ~fav_wins)
            a_win |= level & a_pen
            b_win |= level & ~a_pen

        winner = np.where(a_win, a, b)
        loser = np.where(a_win, b, a)
        return winner, loser


def _resolve_groups_per_sim(
    groups: _Groups,
    team_idx: dict[str, int],
    n_sims: int,
    lots: FloatArray,
) -> tuple[IntArray, IntArray]:
    """Per sim: rank every group, allocate thirds via Annex C, fill the R32.

    Returns (r32_a, r32_b), each (n_sims, 16) team-index arrays for matches 73-88.
    """
    r32_matches = [m for m in BRACKET if 73 <= m.match_no <= 88]
    r32_a = np.empty((n_sims, 16), dtype=np.int64)
    r32_b = np.empty((n_sims, 16), dtype=np.int64)

    for s in range(n_sims):
        winners: dict[str, str] = {}
        runners: dict[str, str] = {}
        third_of_group: dict[str, str] = {}
        third_records = []
        for group in GROUPS:
            team_ids = groups.teams_by_group[group]
            results = [
                (gm.home, gm.away, int(gm.home_goals[s]), int(gm.away_goals[s]))
                for gm in groups.matches_by_group[group]
            ]
            key = {t: float(lots[s, team_idx[t]]) for t in team_ids}
            ranked = rank_group(team_ids, results, key)
            table = standings(team_ids, results)
            winners[group] = ranked[0]
            runners[group] = ranked[1]
            third = ranked[2]
            third_of_group[group] = third
            third_records.append(
                ThirdPlaceRecord(
                    third,
                    group,
                    table[third].points,
                    table[third].goal_diff,
                    table[third].goals_for,
                )
            )

        ranked_thirds = rank_thirds(
            third_records, {r.team_id: float(lots[s, team_idx[r.team_id]]) for r in third_records}
        )
        qualifying = ranked_thirds[:8]
        assignment = r32_assignment(frozenset(r.group for r in qualifying))
        third_for_match = {m: third_of_group[g] for m, g in assignment.items()}

        for col, match in enumerate(r32_matches):
            r32_a[s, col] = team_idx[_resolve_group_slot(match.side_a, winners, runners)]
            r32_b[s, col] = team_idx[
                _resolve_group_slot(match.side_b, winners, runners, third_for_match)
            ]
    return r32_a, r32_b


def _resolve_group_slot(
    slot: tuple[str, str | int],
    winners: dict[str, str],
    runners: dict[str, str],
    third_for_match: dict[int, str] | None = None,
) -> str:
    kind, key = slot
    if kind == "1":
        return winners[str(key)]
    if kind == "2":
        return runners[str(key)]
    if kind == "3":
        assert third_for_match is not None
        return third_for_match[int(key)]
    raise ValueError(f"not a group slot: {slot}")


def simulate_tournament(
    matches: pd.DataFrame,
    model: ScorelineForecaster,
    n_sims: int,
    rng: np.random.Generator,
    policy: KnockoutPolicy | None = None,
    today: dt.date | None = None,
) -> pd.DataFrame:
    """Per-team probabilities of reaching each stage and winning the cup.

    Returns a frame indexed by team_id with columns ``group`` and the six
    ``STAGES`` probabilities, sorted by P(champion) descending.
    """
    policy = policy or KnockoutPolicy()
    today = today or dt.datetime.now(dt.UTC).date()

    groups = _build_groups(matches, model, n_sims, rng, today)
    all_teams = sorted(t for ts in groups.teams_by_group.values() for t in ts)
    team_idx = {t: i for i, t in enumerate(all_teams)}
    group_of = {t: g for g, ts in groups.teams_by_group.items() for t in ts}

    lots = rng.random((n_sims, len(all_teams)))
    r32_a, r32_b = _resolve_groups_per_sim(groups, team_idx, n_sims, lots)

    cdf, eg_home, eg_away, reg = _precompute_ko_grids(all_teams, model, today)
    engine = _KnockoutEngine(cdf, eg_home, eg_away, reg, round(cdf.shape[2] ** 0.5), policy, rng)

    counts = {stage: np.zeros(len(all_teams), dtype=np.int64) for stage in STAGES}
    # everyone in the R32 "reaches R32"
    for col in range(16):
        np.add.at(counts["reach_r32"], r32_a[:, col], 1)
        np.add.at(counts["reach_r32"], r32_b[:, col], 1)

    winners: dict[int, IntArray] = {}
    losers: dict[int, IntArray] = {}
    stage_for_winner = {  # winners of these matches reach the keyed stage
        "reach_r16": range(73, 89),
        "reach_qf": range(89, 97),
        "reach_sf": range(97, 101),
        "reach_final": range(101, 103),
    }

    for match in BRACKET:
        if 73 <= match.match_no <= 88:
            col = match.match_no - 73
            a, b = r32_a[:, col], r32_b[:, col]
        else:
            a = _resolve_ko_side(match.side_a, winners, losers)
            b = _resolve_ko_side(match.side_b, winners, losers)
        w, lo = engine.play(a, b)
        winners[match.match_no] = w
        losers[match.match_no] = lo

    for stage, match_range in stage_for_winner.items():
        for mno in match_range:
            np.add.at(counts[stage], winners[mno], 1)
    np.add.at(counts["champion"], winners[104], 1)

    frame = pd.DataFrame(
        {
            "team_id": all_teams,
            "group": [group_of[t] for t in all_teams],
            **{stage: counts[stage] / n_sims for stage in STAGES},
        }
    )
    return frame.sort_values("champion", ascending=False, ignore_index=True)


def _resolve_ko_side(
    slot: tuple[str, str | int], winners: dict[int, IntArray], losers: dict[int, IntArray]
) -> IntArray:
    kind, key = slot
    if kind == "W":
        return winners[int(key)]
    if kind == "L":
        return losers[int(key)]
    raise ValueError(f"not a knockout slot: {slot}")
