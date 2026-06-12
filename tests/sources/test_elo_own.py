import datetime as dt

import pytest

from wc2026.data.schema import Match, MatchStatus, matches_to_frame
from wc2026.data.sources.elo_own import compute_elo, goal_diff_multiplier, tournament_k

AS_OF = dt.date(2026, 1, 1)


def match(
    home: str,
    away: str,
    goals: tuple[int, int],
    *,
    date: dt.date = dt.date(2020, 1, 1),
    neutral: bool = True,
    tournament: str = "friendly",
    match_id: str | None = None,
) -> Match:
    return Match(
        match_id=match_id or f"m_{date:%Y%m%d}_{home}_{away}",
        date=date,
        home_id=home,
        away_id=away,
        home_goals=goals[0],
        away_goals=goals[1],
        neutral=neutral,
        tournament=tournament,
        status=MatchStatus.FINISHED,
    )


def rating(frame, team):  # type: ignore[no-untyped-def]
    return float(frame.set_index("team_id").loc[team, "rating"])


class TestKFactors:
    @pytest.mark.parametrize(
        ("tournament", "k"),
        [
            ("fifa_world_cup", 60.0),
            ("copa_america", 50.0),
            ("uefa_euro", 50.0),
            ("fifa_world_cup_qualification", 40.0),
            ("uefa_nations_league", 40.0),
            ("gulf_cup", 30.0),
            ("friendly", 20.0),
        ],
    )
    def test_mapping(self, tournament: str, k: float) -> None:
        assert tournament_k(tournament) == k

    @pytest.mark.parametrize(
        ("margin", "mult"), [(0, 1.0), (1, 1.0), (2, 1.5), (3, 1.75), (4, 1.875), (5, 2.0)]
    )
    def test_goal_diff(self, margin: int, mult: float) -> None:
        assert goal_diff_multiplier(margin) == mult


class TestComputeElo:
    def test_neutral_friendly_win_by_one(self) -> None:
        # equal ratings, neutral: expected 0.5; friendly K=20 -> +-10
        frame = compute_elo(matches_to_frame([match("a", "b", (1, 0))]), as_of=AS_OF)
        assert rating(frame, "a") == pytest.approx(1510.0)
        assert rating(frame, "b") == pytest.approx(1490.0)

    def test_margin_multiplier(self) -> None:
        frame = compute_elo(matches_to_frame([match("a", "b", (2, 0))]), as_of=AS_OF)
        assert rating(frame, "a") == pytest.approx(1515.0)  # 20 * 1.5 * 0.5

    def test_draw_between_equals_changes_nothing(self) -> None:
        frame = compute_elo(matches_to_frame([match("a", "b", (1, 1))]), as_of=AS_OF)
        assert rating(frame, "a") == pytest.approx(1500.0)
        assert rating(frame, "b") == pytest.approx(1500.0)

    def test_home_advantage_only_when_not_neutral(self) -> None:
        # home win at home: expected_home = 1/(1+10^-0.25) ~ 0.640065 -> smaller gain
        home_game = compute_elo(
            matches_to_frame([match("a", "b", (1, 0), neutral=False)]), as_of=AS_OF
        )
        assert rating(home_game, "a") == pytest.approx(1500 + 20 * (1 - 0.6400649998), abs=1e-6)

    def test_zero_sum(self) -> None:
        matches = [
            match("a", "b", (3, 1), date=dt.date(2020, 1, 1)),
            match("b", "c", (0, 2), date=dt.date(2020, 2, 1), tournament="fifa_world_cup"),
            match("c", "a", (1, 1), date=dt.date(2020, 3, 1), neutral=False),
        ]
        frame = compute_elo(matches_to_frame(matches), as_of=AS_OF)
        assert frame["rating"].sum() == pytest.approx(3 * 1500.0)

    def test_strictly_before_as_of_is_a_leak_guard(self) -> None:
        matches = [
            match("a", "b", (1, 0), date=dt.date(2020, 1, 1)),
            match("a", "b", (0, 5), date=dt.date(2026, 1, 1)),  # ON the cutoff: excluded
            match("a", "b", (0, 5), date=dt.date(2026, 3, 1)),  # after: excluded
        ]
        frame = compute_elo(matches_to_frame(matches), as_of=AS_OF)
        assert rating(frame, "a") == pytest.approx(1510.0)
        assert int(frame.set_index("team_id").loc["a", "n_matches"]) == 1

    def test_replay_is_chronological_regardless_of_input_order(self) -> None:
        # a beats b BEFORE b's big win over c; feeding rows reversed must not matter
        first = match("a", "b", (1, 0), date=dt.date(2020, 1, 1))
        second = match("b", "c", (4, 0), date=dt.date(2021, 1, 1))
        forward = compute_elo(matches_to_frame([first, second]), as_of=AS_OF)
        reverse = compute_elo(matches_to_frame([second, first]), as_of=AS_OF)
        assert rating(forward, "b") == pytest.approx(rating(reverse, "b"))

    def test_shootout_match_is_a_draw(self) -> None:
        shootout = Match(
            match_id="m1",
            date=dt.date(2022, 12, 18),
            home_id="argentina",
            away_id="france",
            home_goals=3,
            away_goals=3,
            neutral=True,
            tournament="fifa_world_cup",
            status=MatchStatus.FINISHED,
            went_to_shootout=True,
            shootout_winner_id="argentina",
        )
        frame = compute_elo(matches_to_frame([shootout]), as_of=AS_OF)
        assert rating(frame, "argentina") == pytest.approx(1500.0)  # draw, equal ratings
