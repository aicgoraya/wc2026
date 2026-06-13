"""Joining odds-api events to canonical fixtures — fail loud, never drop.

An odds event references teams by bookmaker display names and kickoff
timestamp; fixtures use canonical ids and the UTC kickoff DATE. The join key
is (resolved team pair, kickoff date within one day). Any event that resolves
to zero or several fixtures raises ``OddsJoinError`` — market data is too
precious to silently drop, and an ambiguity means the key assumption broke.

Orientation: bookmakers may list the pair in either order (and the canonical
frame swaps host matches), so the join records ``flipped`` and the benchmark
probabilities are reoriented to the fixture's home side before scoring.
"""

import dataclasses

import pandas as pd

from wc2026.data.names import NameResolver, default_resolver
from wc2026.data.store import Store
from wc2026.pipeline.collect import ODDS_DATASET


class OddsJoinError(RuntimeError):
    """One or more odds events could not be joined to exactly one fixture."""


def load_all_quotes(store: Store) -> pd.DataFrame:
    """Concatenate every stored odds snapshot into one time series.

    Snapshots written before the ``fetched_at_utc`` column existed are
    backfilled from their manifest timestamp (verbatim raw responses are
    archived, so nothing was lost).
    """
    frames = []
    for snap_id in store.snapshots(ODDS_DATASET):
        frame = store.read(ODDS_DATASET, "quotes", snap_id)
        if "fetched_at_utc" not in frame.columns:
            manifest = store.read_manifest(ODDS_DATASET, snap_id)
            frame["fetched_at_utc"] = pd.Timestamp(manifest["meta"]["fetched_at_utc"])
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("no odds snapshots in the store")
    out = pd.concat(frames, ignore_index=True)
    out["fetched_at_utc"] = pd.to_datetime(out["fetched_at_utc"], utc=True)
    return out


@dataclasses.dataclass(frozen=True)
class JoinReport:
    """Diagnostics from a (non-raising) join attempt."""

    n_events: int
    joined: pd.DataFrame  # event_id, match_id, flipped
    failures: tuple[str, ...]

    @property
    def coverage(self) -> float:
        """Fraction of events joined to exactly one fixture."""
        return len(self.joined) / self.n_events if self.n_events else 1.0


def _attempt_join(
    events: pd.DataFrame,
    fixtures: pd.DataFrame,
    resolver: NameResolver,
) -> JoinReport:
    candidates: dict[frozenset[str], list[tuple[str, pd.Timestamp, str]]] = {}
    fixture_rows = zip(
        fixtures["match_id"].astype(str),
        fixtures["date"],
        fixtures["home_id"].astype(str),
        fixtures["away_id"].astype(str),
        strict=True,
    )
    for match_id, date, home_id, away_id in fixture_rows:
        candidates.setdefault(frozenset((home_id, away_id)), []).append((match_id, date, home_id))

    rows: list[dict[str, object]] = []
    failures: list[str] = []
    event_rows = zip(
        events["event_id"].astype(str),
        events["commence_time"],
        events["home_name"].astype(str),
        events["away_name"].astype(str),
        strict=True,
    )
    for event_id, commence, home_name, away_name in event_rows:
        try:
            home_id = resolver.resolve("odds_api", home_name)
            away_id = resolver.resolve("odds_api", away_name)
        except Exception as exc:
            failures.append(f"{event_id} ({home_name} vs {away_name}): {exc}")
            continue
        kickoff_date = pd.Timestamp(commence).tz_convert("UTC").normalize().tz_localize(None)
        near = [
            (match_id, fixture_home)
            for match_id, date, fixture_home in candidates.get(frozenset((home_id, away_id)), [])
            if abs((date - kickoff_date).days) <= 1
        ]
        if len(near) != 1:
            failures.append(
                f"{event_id} ({home_name} vs {away_name} @ {commence}): "
                f"{len(near)} candidate fixtures"
            )
            continue
        match_id, fixture_home = near[0]
        rows.append(
            {"event_id": event_id, "match_id": match_id, "flipped": home_id != fixture_home}
        )
    joined = pd.DataFrame(rows, columns=["event_id", "match_id", "flipped"])
    return JoinReport(n_events=len(events), joined=joined, failures=tuple(failures))


def join_events_to_fixtures(
    events: pd.DataFrame,
    fixtures: pd.DataFrame,
    resolver: NameResolver | None = None,
) -> pd.DataFrame:
    """Map every odds event to exactly one fixture; raises ``OddsJoinError``.

    ``events`` needs (event_id, commence_time, home_name, away_name) — one
    row per event; ``fixtures`` is the canonical WC frame. Returns
    (event_id, match_id, flipped).
    """
    report = _attempt_join(events, fixtures, resolver or default_resolver())
    if report.failures:
        detail = "\n  ".join(report.failures[:10])
        raise OddsJoinError(
            f"{len(report.failures)} of {report.n_events} odds events failed to join:\n  {detail}"
        )
    return report.joined


def join_coverage(
    events: pd.DataFrame,
    fixtures: pd.DataFrame,
    resolver: NameResolver | None = None,
) -> JoinReport:
    """Non-raising variant for diagnostics/reporting."""
    return _attempt_join(events, fixtures, resolver or default_resolver())


def unique_events(quotes: pd.DataFrame) -> pd.DataFrame:
    """One row per odds event from a quotes time series."""
    return quotes.drop_duplicates("event_id")[
        ["event_id", "commence_time", "home_name", "away_name"]
    ].reset_index(drop=True)
