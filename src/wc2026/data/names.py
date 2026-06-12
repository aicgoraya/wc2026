"""Team-name resolution: every source's naming quirks map to one canonical id.

Canonical ids are the slugs of the martj42 historical dataset's team names
(the largest corpus, e.g. ``united_states``, ``czech_republic``,
``ivory_coast``), shipped as a packaged list. Resolution FAILS LOUDLY: an
unknown name raises ``UnresolvedTeamNameError`` — a match is never silently
dropped or mis-keyed. New names (or new source spellings) are added to the
override table deliberately, with the coverage report as the review tool.
"""

import csv
import dataclasses
import re
import unicodedata
from collections.abc import Iterable, Mapping
from functools import cache
from importlib import resources


class UnresolvedTeamNameError(LookupError):
    """A source's team name could not be resolved to a canonical team id."""


def canonical_slug(name: str) -> str:
    """Normalize a display name to a slug (``"Côte d'Ivoire"`` → ``"cote_divoire"``)."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    ascii_name = ascii_name.replace("'", "")  # apostrophes join ("d'Ivoire" -> "divoire")
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_")
    if not slug:
        raise ValueError(f"name {name!r} produced an empty slug")
    return slug


# Keyed by (source, canonical_slug(raw name)) -> canonical id. Only entries
# verified against live feeds or the historical corpus belong here.
OVERRIDES: Mapping[str, Mapping[str, str]] = {
    "football_data": {
        "czechia": "czech_republic",
        "bosnia_herzegovina": "bosnia_and_herzegovina",
        "cape_verde_islands": "cape_verde",  # seen live 2026-06-12
        "congo_dr": "dr_congo",  # seen live 2026-06-12
    },
    "odds_api": {
        "usa": "united_states",
        "czechia": "czech_republic",
        "bosnia_herzegovina": "bosnia_and_herzegovina",
    },
}


@dataclasses.dataclass(frozen=True)
class CoverageReport:
    """Resolution coverage of a batch of source names (non-raising diagnostics)."""

    source: str
    mapped: Mapping[str, str]
    unmapped: tuple[str, ...]

    @property
    def coverage(self) -> float:
        """Fraction of names resolved."""
        total = len(self.mapped) + len(self.unmapped)
        return len(self.mapped) / total if total else 1.0


class NameResolver:
    """Resolves source team names against the canonical id universe."""

    def __init__(
        self,
        known_ids: frozenset[str],
        overrides: Mapping[str, Mapping[str, str]] = OVERRIDES,
    ) -> None:
        self._known = known_ids
        self._overrides = overrides

    def resolve(self, source: str, raw_name: str) -> str:
        """Canonical id for a source name; raises ``UnresolvedTeamNameError``."""
        slug = canonical_slug(raw_name)
        slug = self._overrides.get(source, {}).get(slug, slug)
        if slug not in self._known:
            raise UnresolvedTeamNameError(
                f"{source}: cannot resolve team name {raw_name!r} (slug {slug!r});"
                " add an override to wc2026.data.names.OVERRIDES if legitimate"
            )
        return slug

    def coverage(self, source: str, raw_names: Iterable[str]) -> CoverageReport:
        """Try-resolve a batch; collects failures instead of raising."""
        mapped: dict[str, str] = {}
        unmapped: list[str] = []
        for raw_name in sorted(set(raw_names)):
            try:
                mapped[raw_name] = self.resolve(source, raw_name)
            except UnresolvedTeamNameError:
                unmapped.append(raw_name)
        return CoverageReport(source=source, mapped=mapped, unmapped=tuple(unmapped))


@cache
def load_canonical_ids() -> frozenset[str]:
    """The packaged canonical team-id universe (generated from the martj42 corpus)."""
    text = (resources.files("wc2026.data") / "resources" / "canonical_teams.csv").read_text()
    rows = list(csv.DictReader(text.splitlines()))
    return frozenset(row["team_id"] for row in rows)


def default_resolver() -> NameResolver:
    """Resolver over the packaged canonical universe with the builtin overrides."""
    return NameResolver(load_canonical_ids())


def resolve(source: str, raw_name: str) -> str:
    """Convenience: resolve one name with the default resolver."""
    return default_resolver().resolve(source, raw_name)
