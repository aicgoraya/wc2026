"""Provenance script: rebuild the Annex C third-place allocation CSV.

Parses the FIFA World Cup 2026 third-place allocation table (495 combinations)
from Wikipedia's transcription of Annex C of the official FIFA regulations
(https://digitalhub.fifa.com/m/636f5c9c6f29771f/original/FWC2026_regulations_EN.pdf),
validates every row against the independently-sourced Round-of-32 slot
constraints, and writes
``src/wc2026/tournament/resources/annex_c_third_place.csv``.

Run from the repo root: ``uv run python tools/build_annex_c.py``. The committed
CSV is the source of truth at runtime; this script only regenerates/re-verifies
it. Network access required (Wikipedia action API).
"""

import csv
import json
import re
import urllib.request
from itertools import combinations
from pathlib import Path

API = (
    "https://en.wikipedia.org/w/api.php?action=parse"
    "&page=Template:2026_FIFA_World_Cup_third-place_table"
    "&format=json&prop=wikitext&formatversion=2"
)

# Table column order -> group winner -> R32 match number, from the published
# Round-of-32 match list (Matches 73-88). These are the 8 matches that feature
# a best-third-placed team.
COL_WINNER = ("A", "B", "D", "E", "G", "I", "K", "L")
WINNER_MATCH = {"A": 79, "B": 85, "D": 81, "E": 74, "G": 82, "I": 77, "K": 87, "L": 80}
ALLOWED = {  # match -> groups whose third may fill it (the "Best 3rd place X/Y/Z" labels)
    74: set("ABCDF"),
    77: set("CDFGH"),
    79: set("CEFHI"),
    80: set("EHIJK"),
    81: set("BEFIJ"),
    82: set("AEHIJ"),
    85: set("EFGIJ"),
    87: set("DEIJL"),
}
GROUPS = list("ABCDEFGHIJKL")
MATCH_COLS = (74, 77, 79, 80, 81, 82, 85, 87)


def _bold(cell: str) -> str | None:
    m = re.match(r"'''([A-L])'''", cell.strip())
    return m.group(1) if m else None


def _third(cell: str) -> str | None:
    m = re.match(r"3([A-L])", cell.strip())
    return m.group(1) if m else None


def parse_rows(wikitext: str) -> dict[int, tuple[list[str], list[str]]]:
    """Extract {row_no: (qualifying_groups, assignments)} from the template."""
    lines = wikitext.split("\n")
    rows: dict[int, tuple[list[str], list[str]]] = {}
    for i, line in enumerate(lines):
        m = re.match(r'! scope="row" \| (\d+)', line.strip())
        if not m:
            continue
        num = int(m.group(1))
        cells = [c.strip() for c in lines[i + 1].lstrip("|").split("||")]
        quals = [GROUPS[k] for k, c in enumerate(cells[:12]) if _bold(c)]
        # Row 1 carries its 8 assignments two lines down (after the rowspan
        # separator cell); rows 2-495 carry all 20 cells inline.
        assign_cells = (
            [c.strip() for c in lines[i + 3].lstrip("|").split("||")]
            if num == 1
            else cells[12:20]
        )
        assigns = [_third(c) for c in assign_cells]
        rows[num] = (quals, [a for a in assigns if a is not None])
    return rows


def validate(rows: dict[int, tuple[list[str], list[str]]]) -> None:
    """Raise if any row violates the structural invariants."""
    if len(rows) != 495:
        raise ValueError(f"expected 495 rows, got {len(rows)}")
    for num, (quals, assigns) in rows.items():
        if len(quals) != 8:
            raise ValueError(f"row {num}: {len(quals)} qualifying groups")
        if set(assigns) != set(quals):
            raise ValueError(f"row {num}: assignments {assigns} != qualifiers {quals}")
        for col, grp in zip(COL_WINNER, assigns, strict=True):
            match_no = WINNER_MATCH[col]
            if grp not in ALLOWED[match_no]:
                raise ValueError(f"row {num}: 3{grp} -> M{match_no} violates slot constraint")
    combos = {frozenset(q) for q, _ in rows.values()}
    if combos != {frozenset(c) for c in combinations(GROUPS, 8)}:
        raise ValueError("rows do not cover all 495 combinations exactly once")


def main() -> None:
    with urllib.request.urlopen(API, timeout=60) as resp:  # noqa: S310 - fixed https URL
        payload = json.load(resp)
    rows = parse_rows(payload["parse"]["wikitext"])
    validate(rows)
    out = (
        Path(__file__).resolve().parent.parent
        / "src/wc2026/tournament/resources/annex_c_third_place.csv"
    )
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["option", "groups", *(f"m{m}" for m in MATCH_COLS)])
        for num in sorted(rows):
            quals, assigns = rows[num]
            by_match = {WINNER_MATCH[col]: grp for col, grp in zip(COL_WINNER, assigns, strict=True)}
            w.writerow([num, "".join(sorted(quals)), *(by_match[m] for m in MATCH_COLS)])
    print(f"validated 495 rows; wrote {out}")


if __name__ == "__main__":
    main()
