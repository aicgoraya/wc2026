"""Static 2026 structure: groups A-L, the R32 slot map (matches 73-88), knockout chain.

Golden-tested against the published bracket. Implemented from the official
FIFA regulations PDF in Phase 3.
"""

import dataclasses

from wc2026.data.schema import Stage


@dataclasses.dataclass(frozen=True)
class SlotRef:
    """A bracket slot reference, e.g. winner of group C ('1C') or of match 74 ('W74')."""

    code: str


@dataclasses.dataclass(frozen=True)
class BracketMatch:
    """One knockout fixture: match number, stage, and its two slot references."""

    match_no: int
    stage: Stage
    side_a: SlotRef
    side_b: SlotRef


def bracket_2026() -> tuple[BracketMatch, ...]:
    """The full R32→final bracket with slot references (matches 73-104)."""
    raise NotImplementedError("ships in Phase 3")
