"""The Source protocol: fetch() does network, parse() is pure and testable."""

import dataclasses
import datetime as dt
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import pandas as pd


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
