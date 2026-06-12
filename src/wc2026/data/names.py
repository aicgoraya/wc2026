"""Team-name resolution: every source's naming quirks map to one canonical slug."""

import re
import unicodedata


def canonical_slug(name: str) -> str:
    """Normalize a display name to a canonical slug (``"Côte d'Ivoire"`` → ``"cote_divoire"``)."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_")
    if not slug:
        raise ValueError(f"name {name!r} produced an empty slug")
    return slug


def resolve(source: str, raw_name: str) -> str:
    """Map a source-specific team name to the canonical slug.

    Backed by an explicit, tested override map per source on top of
    ``canonical_slug`` (the messy part — e.g. "USA" vs "United States").
    """
    raise NotImplementedError("ships in Phase 1b with the historical-results ingest")
