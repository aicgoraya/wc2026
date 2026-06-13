"""Golden tests pinning the 2026 knockout bracket to the published structure."""

from wc2026.data.schema import Stage
from wc2026.tournament.annex_c import THIRD_PLACE_MATCHES
from wc2026.tournament.structure import (
    BRACKET,
    GROUPS,
    THIRD_PLACE_SLOTS,
    bracket_2026,
    bracket_by_match,
)

# The published Round-of-32 pairings (Matches 73-88), verbatim from the
# official bracket. This is the golden reference the implementation must match.
GOLDEN_R32 = {
    73: (("2", "A"), ("2", "B")),
    74: (("1", "E"), ("3", 74)),
    75: (("1", "F"), ("2", "C")),
    76: (("1", "C"), ("2", "F")),
    77: (("1", "I"), ("3", 77)),
    78: (("2", "E"), ("2", "I")),
    79: (("1", "A"), ("3", 79)),
    80: (("1", "L"), ("3", 80)),
    81: (("1", "D"), ("3", 81)),
    82: (("1", "G"), ("3", 82)),
    83: (("2", "K"), ("2", "L")),
    84: (("1", "H"), ("2", "J")),
    85: (("1", "B"), ("3", 85)),
    86: (("1", "J"), ("2", "H")),
    87: (("1", "K"), ("3", 87)),
    88: (("2", "D"), ("2", "G")),
}
GOLDEN_KO = {  # match-winner chain R16 -> final
    89: (("W", 74), ("W", 77)),
    90: (("W", 73), ("W", 75)),
    91: (("W", 76), ("W", 78)),
    92: (("W", 79), ("W", 80)),
    93: (("W", 83), ("W", 84)),
    94: (("W", 81), ("W", 82)),
    95: (("W", 86), ("W", 88)),
    96: (("W", 85), ("W", 87)),
    97: (("W", 89), ("W", 90)),
    98: (("W", 93), ("W", 94)),
    99: (("W", 91), ("W", 92)),
    100: (("W", 95), ("W", 96)),
    101: (("W", 97), ("W", 98)),
    102: (("W", 99), ("W", 100)),
    103: (("L", 101), ("L", 102)),
    104: (("W", 101), ("W", 102)),
}


def test_groups_are_a_to_l() -> None:
    assert tuple("ABCDEFGHIJKL") == GROUPS


def test_bracket_has_32_matches_73_to_104() -> None:
    assert len(BRACKET) == 32
    assert [m.match_no for m in BRACKET] == list(range(73, 105))


def test_r32_pairings_golden() -> None:
    by_match = bracket_by_match()
    for match_no, (a, b) in GOLDEN_R32.items():
        assert (by_match[match_no].side_a, by_match[match_no].side_b) == (a, b)
        assert by_match[match_no].stage is Stage.R32


def test_knockout_chain_golden() -> None:
    by_match = bracket_by_match()
    for match_no, (a, b) in GOLDEN_KO.items():
        assert (by_match[match_no].side_a, by_match[match_no].side_b) == (a, b)


def test_stage_assignment() -> None:
    by_match = bracket_by_match()
    assert by_match[89].stage is Stage.R16
    assert by_match[97].stage is Stage.QF
    assert by_match[101].stage is Stage.SF
    assert by_match[103].stage is Stage.THIRD
    assert by_match[104].stage is Stage.FINAL


def test_third_place_slots_match_annex_c_keys() -> None:
    assert THIRD_PLACE_SLOTS == THIRD_PLACE_MATCHES
    third_sides = {m.match_no for m in BRACKET if m.side_b[0] == "3" or m.side_a[0] == "3"}
    assert third_sides == set(THIRD_PLACE_SLOTS)


def test_every_group_position_appears_exactly_once_in_r32() -> None:
    # 12 winners + 12 runners-up + 8 thirds = 32 teams across 16 R32 matches
    slots = [s for m in BRACKET[:16] for s in (m.side_a, m.side_b)]
    winners = sorted(s[1] for s in slots if s[0] == "1")
    runners = sorted(s[1] for s in slots if s[0] == "2")
    assert winners == list("ABCDEFGHIJKL")
    assert runners == list("ABCDEFGHIJKL")
    assert sum(1 for s in slots if s[0] == "3") == 8


def test_bracket_2026_is_stable() -> None:
    assert bracket_2026() is BRACKET
