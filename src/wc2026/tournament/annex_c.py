"""The Annex C third-place allocation: 495 combinations → R32 slot assignments.

Loads the packaged ``annex_c_third_place.csv`` (495 rows, parsed from the
official FIFA 2026 regulations via ``tools/build_annex_c.py`` and validated:
every combination assigns its 8 qualifying groups to the 8 third-place slots,
each respecting the slot's allowed groups). Lookup is keyed by the frozenset
of the 8 groups whose third-placed team qualifies.
"""

import csv
from collections.abc import Mapping
from functools import cache
from importlib import resources

THIRD_PLACE_MATCHES: tuple[int, ...] = (74, 77, 79, 80, 81, 82, 85, 87)


@cache
def _table() -> Mapping[frozenset[str], Mapping[int, str]]:
    text = (
        resources.files("wc2026.tournament") / "resources" / "annex_c_third_place.csv"
    ).read_text()
    table: dict[frozenset[str], Mapping[int, str]] = {}
    for row in csv.DictReader(text.splitlines()):
        key = frozenset(row["groups"])
        table[key] = {m: row[f"m{m}"] for m in THIRD_PLACE_MATCHES}
    if len(table) != 495:
        raise ValueError(f"expected 495 Annex C combinations, loaded {len(table)}")
    return table


def r32_assignment(qualified_groups: frozenset[str]) -> Mapping[int, str]:
    """Map an 8-group combination of qualifying thirds to ``{match_no: group}``.

    ``match_no`` is one of the 8 third-place R32 matches; the value is the group
    whose third-placed team plays there. Raises ``KeyError`` only for inputs
    that are not one of the 495 valid 8-of-12 combinations.
    """
    if len(qualified_groups) != 8:
        raise KeyError(
            f"expected 8 qualifying groups, got {len(qualified_groups)}: {qualified_groups}"
        )
    return _table()[qualified_groups]
