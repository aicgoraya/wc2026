"""Collection entrypoints: fetch → archive raw bytes → parse → versioned snapshot.

Raw responses are archived verbatim under ``data/raw/<source>/`` so every
parsed snapshot can be reproduced (or re-parsed after a bug fix) without
re-hitting the APIs — essential for odds, which cannot be re-fetched.
"""

from pathlib import Path

from wc2026.config import Settings
from wc2026.data.merge import merge_canonical
from wc2026.data.sources import martj42
from wc2026.data.sources.base import RawPayload
from wc2026.data.sources.football_data import FootballDataSource
from wc2026.data.sources.odds_api import OddsApiSource, discover_sport_key
from wc2026.data.store import MATCHES_DATASET, SnapshotId, Store

ODDS_DATASET = "odds"
WC_FIXTURES_DATASET = "wc_fixtures"
"""WC fixtures/results as pulled; the merged canonical matches dataset is built in Phase 1b."""


class MissingCredentialError(RuntimeError):
    """A required API key is absent from the environment / .env."""


def _archive_raw(raw: RawPayload, raw_root: Path) -> Path:
    path = raw_root / raw.source / f"{raw.fetched_at_utc:%Y%m%dT%H%M%S_%fZ}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw.content)
    return path


def collect_odds(settings: Settings) -> tuple[SnapshotId, dict[str, str]]:
    """One odds snapshot: discover sport key, fetch h2h quotes, archive, store.

    Returns the snapshot id and the source meta (incl. remaining-quota headers).
    """
    if settings.odds_api_key is None:
        raise MissingCredentialError(
            "ODDS_API_KEY is not set — register the free tier at"
            " https://the-odds-api.com/#get-access and add it to .env"
        )
    api_key = settings.odds_api_key.get_secret_value()
    sport_key = discover_sport_key(api_key)
    source = OddsApiSource(api_key, sport_key)
    raw = source.fetch()
    raw_path = _archive_raw(raw, settings.data_root / "raw")
    quotes = source.parse(raw)
    store = Store(settings.data_root / "snapshots")
    snap_id = store.write_snapshot(
        ODDS_DATASET,
        {"quotes": quotes},
        meta={
            "source": source.name,
            "sport_key": sport_key,
            "raw_path": str(raw_path),
            "fetched_at_utc": raw.fetched_at_utc.isoformat(),
            **raw.meta,
        },
    )
    return snap_id, dict(raw.meta)


def collect_history(settings: Settings) -> tuple[SnapshotId, dict[str, int]]:
    """Build THE canonical matches dataset: martj42 history + live WC feed.

    Requires a prior ``wc_fixtures`` snapshot (run ``snapshot-fixtures`` first)
    — the WC2026 window is owned by the live feed, so merging without it would
    silently drop the tournament being forecast.
    """
    results_raw = martj42.fetch_results()
    shootouts_raw = martj42.fetch_shootouts()
    raw_root = settings.data_root / "raw"
    _archive_raw(results_raw, raw_root)
    _archive_raw(shootouts_raw, raw_root)
    history = martj42.parse(results_raw, shootouts_raw)
    store = Store(settings.data_root / "snapshots")
    wc_fixtures = store.read(WC_FIXTURES_DATASET, "matches")
    merged = merge_canonical(history, wc_fixtures)
    teams = martj42.team_universe(history)
    counts = {"history": len(history), "wc_feed": len(wc_fixtures), "merged": len(merged)}
    snap_id = store.write_snapshot(
        MATCHES_DATASET,
        {"matches": merged, "teams": teams},
        meta={"source": "martj42+football_data", **counts},
    )
    return snap_id, counts


def collect_fixtures(settings: Settings) -> tuple[SnapshotId, dict[str, str]]:
    """One fixtures/results snapshot from football-data.org; archive + store."""
    if settings.football_data_token is None:
        raise MissingCredentialError(
            "FOOTBALL_DATA_TOKEN is not set — register the free tier at"
            " https://www.football-data.org/client/register and add it to .env"
        )
    source = FootballDataSource(settings.football_data_token.get_secret_value())
    raw = source.fetch()
    raw_path = _archive_raw(raw, settings.data_root / "raw")
    matches = source.parse(raw)
    store = Store(settings.data_root / "snapshots")
    snap_id = store.write_snapshot(
        WC_FIXTURES_DATASET,
        {"matches": matches},
        meta={
            "source": source.name,
            "raw_path": str(raw_path),
            "fetched_at_utc": raw.fetched_at_utc.isoformat(),
            **raw.meta,
        },
    )
    return snap_id, dict(raw.meta)
