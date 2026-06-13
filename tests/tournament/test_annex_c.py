"""Tests for the Annex C third-place allocation: all 495 combinations valid."""

from itertools import combinations

import pytest

from wc2026.tournament.annex_c import THIRD_PLACE_MATCHES, r32_assignment

GROUPS = "ABCDEFGHIJKL"
# slot -> groups whose third may fill it (the published "Best 3rd place X/Y/Z")
ALLOWED = {
    74: set("ABCDF"),
    77: set("CDFGH"),
    79: set("CEFHI"),
    80: set("EHIJK"),
    81: set("BEFIJ"),
    82: set("AEHIJ"),
    85: set("EFGIJ"),
    87: set("DEIJL"),
}


def test_row1_golden_value() -> None:
    # combination EFGHIJKL is row 1 of the published table
    assignment = r32_assignment(frozenset("EFGHIJKL"))
    assert assignment == {74: "F", 77: "G", 79: "E", 80: "K", 81: "I", 82: "H", 85: "J", 87: "L"}


def test_last_row_golden_value() -> None:
    # combination ABCDEFGH is row 495
    assignment = r32_assignment(frozenset("ABCDEFGH"))
    assert assignment == {74: "C", 77: "F", 79: "H", 80: "E", 81: "B", 82: "A", 85: "G", 87: "D"}


def test_all_495_combinations_valid() -> None:
    seen = 0
    for combo in combinations(GROUPS, 8):
        groups = frozenset(combo)
        assignment = r32_assignment(groups)
        # exactly the 8 third-place matches are assigned
        assert set(assignment) == set(THIRD_PLACE_MATCHES)
        # the 8 assigned groups are exactly the qualifying groups (a perfect matching)
        assert set(assignment.values()) == groups
        # each assignment respects its slot's allowed-groups constraint
        for match_no, group in assignment.items():
            assert group in ALLOWED[match_no], f"3{group} -> M{match_no} for {combo}"
        seen += 1
    assert seen == 495


def test_wrong_size_input_raises() -> None:
    with pytest.raises(KeyError, match="8 qualifying groups"):
        r32_assignment(frozenset("ABCDEFG"))  # only 7
    with pytest.raises(KeyError, match="8 qualifying groups"):
        r32_assignment(frozenset("ABCDEFGHI"))  # 9


def test_assignment_is_a_bijection_for_a_sample() -> None:
    assignment = r32_assignment(frozenset("ABCDEFGH"))
    assert len(set(assignment.values())) == 8  # no group assigned twice
