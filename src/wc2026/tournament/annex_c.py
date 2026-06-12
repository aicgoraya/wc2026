"""The Annex C third-place allocation: 495 combinations → R32 slot assignments.

Extracted programmatically from the official FIFA regulations PDF into a CSV
checked into the repo, validated (every combination assigns 8 distinct groups
to 8 distinct match slots) and spot-checked by hand.
"""

from collections.abc import Mapping


def r32_assignment(qualified_groups: frozenset[str]) -> Mapping[int, str]:
    """Map an 8-group combination of qualified thirds to ``{match_no: group}``.

    Raises ``KeyError`` only on inputs that are not one of the 495 valid
    combinations (i.e. never for a legitimate tournament state).
    """
    raise NotImplementedError("ships in Phase 3")
