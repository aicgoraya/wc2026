"""The Source protocol: fetch() does network, parse() is pure and testable."""

import dataclasses
import datetime as dt
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import pandas as pd
import requests


@dataclasses.dataclass(frozen=True)
class RawPayload:
    """Raw bytes from one fetch, with provenance for the snapshot manifest."""

    source: str
    fetched_at_utc: dt.datetime
    content: bytes
    meta: Mapping[str, str] = dataclasses.field(default_factory=dict)


class SourceUnavailableError(RuntimeError):
    """The external source could not be reached or refused the request."""


@runtime_checkable
class Source(Protocol):
    """One external data source (results, fixtures, odds, ratings)."""

    name: str

    def fetch(self) -> RawPayload:
        """Pull raw content from the network; raises ``SourceUnavailableError``."""
        ...

    def parse(self, raw: RawPayload) -> pd.DataFrame:
        """Convert raw content to a canonical frame; pure, no network."""
        ...


def http_get(
    source: str,
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    params: Mapping[str, str] | None = None,
    keep_headers: tuple[str, ...] = (),
    timeout: float = 30.0,
) -> RawPayload:
    """GET with provenance capture; raises ``SourceUnavailableError`` on any failure.

    ``keep_headers`` selects response headers (e.g. rate-limit counters) to
    record in the payload meta. Secrets must go in ``headers``/``params`` — the
    recorded meta only contains the bare URL.
    """
    try:
        response = requests.get(url, headers=dict(headers or {}), params=params, timeout=timeout)
    except requests.RequestException as exc:
        raise SourceUnavailableError(f"{source}: {exc.__class__.__name__}: {exc}") from exc
    if response.status_code != 200:
        raise SourceUnavailableError(
            f"{source}: HTTP {response.status_code} for {url}: {response.text[:300]}"
        )
    meta = {"url": url}
    for header in keep_headers:
        if header in response.headers:
            meta[header] = response.headers[header]
    return RawPayload(
        source=source,
        fetched_at_utc=dt.datetime.now(dt.UTC),
        content=response.content,
        meta=meta,
    )
