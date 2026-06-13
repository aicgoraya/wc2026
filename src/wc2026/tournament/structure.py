"""Static 2026 knockout structure: the bracket from Round of 32 to the Final.

Slot references are the published group-position / match-winner labels for
matches 73-104, transcribed from the official bracket (FIFA regulations, via
the 2026 FIFA World Cup knockout-stage article). A ``SlotRef`` is one of:

- ``("1", "A")`` winner of group A
- ``("2", "A")`` runner-up of group A
- ``("3", 74)`` the best-third-placed team assigned to match 74 (resolved at
  runtime via ``tournament.annex_c``)
- ``("W", 73)`` winner of match 73
- ``("L", 101)`` loser of match 101 (third-place playoff only)

The eight third-place slots and the group-winner each faces are exactly the
ones the Annex C table is keyed on (matches 74, 77, 79, 80, 81, 82, 85, 87).
"""

import dataclasses

from wc2026.data.schema import Stage

GROUPS: tuple[str, ...] = tuple("ABCDEFGHIJKL")

SlotRef = tuple[str, str | int]
"""A bracket slot: (kind, key). kind in {'1','2','3','W','L'}."""


@dataclasses.dataclass(frozen=True)
class BracketMatch:
    """One knockout fixture: number, stage, and its two slot references."""

    match_no: int
    stage: Stage
    side_a: SlotRef
    side_b: SlotRef


# Round of 32 (matches 73-88). Third-place sides carry the match number so the
# Annex C assignment can resolve them; the allowed-groups annotation is in
# tools/build_annex_c.py and enforced there.
_R32: tuple[BracketMatch, ...] = (
    BracketMatch(73, Stage.R32, ("2", "A"), ("2", "B")),
    BracketMatch(74, Stage.R32, ("1", "E"), ("3", 74)),
    BracketMatch(75, Stage.R32, ("1", "F"), ("2", "C")),
    BracketMatch(76, Stage.R32, ("1", "C"), ("2", "F")),
    BracketMatch(77, Stage.R32, ("1", "I"), ("3", 77)),
    BracketMatch(78, Stage.R32, ("2", "E"), ("2", "I")),
    BracketMatch(79, Stage.R32, ("1", "A"), ("3", 79)),
    BracketMatch(80, Stage.R32, ("1", "L"), ("3", 80)),
    BracketMatch(81, Stage.R32, ("1", "D"), ("3", 81)),
    BracketMatch(82, Stage.R32, ("1", "G"), ("3", 82)),
    BracketMatch(83, Stage.R32, ("2", "K"), ("2", "L")),
    BracketMatch(84, Stage.R32, ("1", "H"), ("2", "J")),
    BracketMatch(85, Stage.R32, ("1", "B"), ("3", 85)),
    BracketMatch(86, Stage.R32, ("1", "J"), ("2", "H")),
    BracketMatch(87, Stage.R32, ("1", "K"), ("3", 87)),
    BracketMatch(88, Stage.R32, ("2", "D"), ("2", "G")),
)

# Round of 16 (89-96), quarter-finals (97-100), semis (101-102),
# third-place playoff (103), final (104).
_R16: tuple[BracketMatch, ...] = (
    BracketMatch(89, Stage.R16, ("W", 74), ("W", 77)),
    BracketMatch(90, Stage.R16, ("W", 73), ("W", 75)),
    BracketMatch(91, Stage.R16, ("W", 76), ("W", 78)),
    BracketMatch(92, Stage.R16, ("W", 79), ("W", 80)),
    BracketMatch(93, Stage.R16, ("W", 83), ("W", 84)),
    BracketMatch(94, Stage.R16, ("W", 81), ("W", 82)),
    BracketMatch(95, Stage.R16, ("W", 86), ("W", 88)),
    BracketMatch(96, Stage.R16, ("W", 85), ("W", 87)),
)
_QF: tuple[BracketMatch, ...] = (
    BracketMatch(97, Stage.QF, ("W", 89), ("W", 90)),
    BracketMatch(98, Stage.QF, ("W", 93), ("W", 94)),
    BracketMatch(99, Stage.QF, ("W", 91), ("W", 92)),
    BracketMatch(100, Stage.QF, ("W", 95), ("W", 96)),
)
_SF: tuple[BracketMatch, ...] = (
    BracketMatch(101, Stage.SF, ("W", 97), ("W", 98)),
    BracketMatch(102, Stage.SF, ("W", 99), ("W", 100)),
)
_FINALS: tuple[BracketMatch, ...] = (
    BracketMatch(103, Stage.THIRD, ("L", 101), ("L", 102)),
    BracketMatch(104, Stage.FINAL, ("W", 101), ("W", 102)),
)

BRACKET: tuple[BracketMatch, ...] = (*_R32, *_R16, *_QF, *_SF, *_FINALS)
"""All 32 knockout fixtures, match 73 through 104, in order."""

THIRD_PLACE_SLOTS: tuple[int, ...] = (74, 77, 79, 80, 81, 82, 85, 87)
"""R32 matches whose 'b' side is a best-third-placed team (Annex C keys)."""


def bracket_2026() -> tuple[BracketMatch, ...]:
    """The full R32->final bracket with slot references (matches 73-104)."""
    return BRACKET


def bracket_by_match() -> dict[int, BracketMatch]:
    """The bracket indexed by match number."""
    return {m.match_no: m for m in BRACKET}
