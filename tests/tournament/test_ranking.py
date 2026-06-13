"""Hand-constructed tests for the exact 2026 group tiebreakers."""

from wc2026.tournament.ranking import (
    MatchResult,
    ThirdPlaceRecord,
    rank_group,
    rank_thirds,
    standings,
)

TEAMS = ["a", "b", "c", "d"]
# a uniform key so tests exercise the football criteria, not the lot
NEUTRAL_KEY = dict.fromkeys(TEAMS, 0.0)


def round_robin(scores: dict[tuple[str, str], tuple[int, int]]) -> list[MatchResult]:
    return [(h, a, hg, ag) for (h, a), (hg, ag) in scores.items()]


def test_standings_basic() -> None:
    results = round_robin({("a", "b"): (2, 0), ("a", "c"): (1, 1)})
    table = standings(["a", "b", "c"], results)
    assert table["a"].points == 4  # win + draw
    assert table["a"].goal_diff == 2
    assert table["b"].points == 0
    assert table["c"].points == 1


def test_clear_points_order() -> None:
    results = round_robin(
        {
            ("a", "b"): (1, 0),
            ("a", "c"): (1, 0),
            ("a", "d"): (1, 0),
            ("b", "c"): (1, 0),
            ("b", "d"): (1, 0),
            ("c", "d"): (1, 0),
        }
    )
    assert rank_group(TEAMS, results, NEUTRAL_KEY) == ["a", "b", "c", "d"]


def test_head_to_head_beats_overall_goal_difference() -> None:
    # a and b both 6 pts (clean 2-way tie at the top). b has a far better OVERALL
    # goal difference (+9 vs +1), but a beat b head-to-head -> a ranks above b
    # (2026: head-to-head is applied before overall GD).
    results = round_robin(
        {
            ("a", "b"): (1, 0),  # a wins the head-to-head
            ("a", "c"): (1, 0),
            ("a", "d"): (0, 1),  # a drops points to d
            ("b", "c"): (5, 0),  # b runs up the score elsewhere
            ("b", "d"): (5, 0),
            ("c", "d"): (0, 0),
        }
    )
    table = standings(TEAMS, results)
    assert table["a"].points == table["b"].points == 6
    assert table["b"].goal_diff == 9 and table["a"].goal_diff == 1  # b far better overall
    ranked = rank_group(TEAMS, results, NEUTRAL_KEY)
    assert set(ranked[:2]) == {"a", "b"}
    assert ranked.index("a") < ranked.index("b")  # a wins on head-to-head


def test_overall_gd_used_when_head_to_head_level() -> None:
    # a and b drew head-to-head and have equal points; the THREE-way logic does
    # not separate them, so overall GD decides: a has the better overall GD.
    results = round_robin(
        {
            ("a", "b"): (1, 1),  # h2h drawn
            ("a", "c"): (3, 0),  # a: big overall GD
            ("a", "d"): (0, 0),
            ("b", "c"): (1, 0),
            ("b", "d"): (0, 0),
            ("c", "d"): (0, 0),
        }
    )
    table = standings(TEAMS, results)
    assert table["a"].points == table["b"].points
    assert table["a"].goal_diff > table["b"].goal_diff
    ranked = rank_group(TEAMS, results, NEUTRAL_KEY)
    assert ranked.index("a") < ranked.index("b")


def test_cyclic_three_way_tie_falls_through_to_overall_gd() -> None:
    # a beats b, b beats c, c beats a (a perfect cycle): all three tie on points
    # AND the head-to-head mini-table is all-level (each 3 pts, 0 GD, 1 goal), so
    # criteria a-c cannot separate them and the tie falls through to OVERALL goal
    # difference. The differing margins vs d set that order: a +3 > b +2 > c +1.
    results = round_robin(
        {
            ("a", "b"): (1, 0),
            ("b", "c"): (1, 0),
            ("c", "a"): (1, 0),
            ("a", "d"): (3, 0),
            ("b", "d"): (2, 0),
            ("c", "d"): (1, 0),
        }
    )
    table = standings(TEAMS, results)
    assert table["a"].points == table["b"].points == table["c"].points == 6
    assert (table["a"].goal_diff, table["b"].goal_diff, table["c"].goal_diff) == (3, 2, 1)
    assert rank_group(TEAMS, results, NEUTRAL_KEY) == ["a", "b", "c", "d"]


def test_tiebreak_key_is_last_resort() -> None:
    # two teams identical on every football criterion: the key decides
    results = round_robin(
        {
            ("a", "b"): (0, 0),
            ("a", "c"): (1, 0),
            ("a", "d"): (0, 1),
            ("b", "c"): (1, 0),
            ("b", "d"): (0, 1),
            ("c", "d"): (0, 0),
        }
    )
    table = standings(TEAMS, results)
    assert (table["a"].points, table["a"].goal_diff, table["a"].goals_for) == (
        table["b"].points,
        table["b"].goal_diff,
        table["b"].goals_for,
    )
    key = {"a": 1.0, "b": 0.0, "c": 0.0, "d": 0.0}
    assert rank_group(TEAMS, results, key).index("a") < rank_group(TEAMS, results, key).index("b")
    key_rev = {"a": 0.0, "b": 1.0, "c": 0.0, "d": 0.0}
    assert rank_group(TEAMS, results, key_rev).index("b") < rank_group(
        TEAMS, results, key_rev
    ).index("a")


class TestRankThirds:
    def test_orders_by_points_then_gd_then_goals(self) -> None:
        thirds = [
            ThirdPlaceRecord("t1", "A", points=4, goal_diff=1, goals_for=3),
            ThirdPlaceRecord("t2", "B", points=4, goal_diff=2, goals_for=2),  # better GD
            ThirdPlaceRecord("t3", "C", points=6, goal_diff=0, goals_for=1),  # most points
            ThirdPlaceRecord("t4", "D", points=4, goal_diff=1, goals_for=5),  # more goals than t1
        ]
        key = dict.fromkeys(["t1", "t2", "t3", "t4"], 0.0)
        order = [r.team_id for r in rank_thirds(thirds, key)]
        assert order == ["t3", "t2", "t4", "t1"]

    def test_key_breaks_exact_ties(self) -> None:
        thirds = [
            ThirdPlaceRecord("x", "A", points=3, goal_diff=0, goals_for=2),
            ThirdPlaceRecord("y", "B", points=3, goal_diff=0, goals_for=2),
        ]
        order = [r.team_id for r in rank_thirds(thirds, {"x": 0.9, "y": 0.1})]
        assert order == ["x", "y"]
