"""Versioned local parquet snapshots — the reproducibility backbone.

Layout: ``<root>/<dataset>/<snapshot_id>/{<frame>.parquet, manifest.json}``.
Snapshots are append-only; every model run records which snapshot it consumed.
"""

import datetime as dt
import json
from pathlib import Path
from typing import Any, Literal, NewType

import pandas as pd

from wc2026.data.schema import MatchStatus

SnapshotId = NewType("SnapshotId", str)

MATCHES_DATASET = "matches"
"""Dataset name under which the canonical matches frame is snapshotted."""


class SnapshotNotFoundError(LookupError):
    """No snapshot exists for the requested dataset/id."""


class Store:
    """Append-only parquet snapshot store under a single root directory."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def write_snapshot(
        self,
        dataset: str,
        frames: dict[str, pd.DataFrame],
        meta: dict[str, Any],
        *,
        now: dt.datetime | None = None,
    ) -> SnapshotId:
        """Write all frames + a manifest as one immutable snapshot; returns its id."""
        if not frames:
            raise ValueError("a snapshot needs at least one frame")
        stamp = now or dt.datetime.now(dt.UTC)
        snap_id = SnapshotId(stamp.strftime("%Y%m%dT%H%M%S_%fZ"))
        path = self._root / dataset / snap_id
        path.mkdir(parents=True, exist_ok=False)
        for frame_name, frame in frames.items():
            frame.to_parquet(path / f"{frame_name}.parquet", index=False)
        manifest = {
            "dataset": dataset,
            "snapshot_id": snap_id,
            "created_utc": stamp.isoformat(),
            "meta": meta,
            "frames": {name: len(frame) for name, frame in frames.items()},
        }
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        return snap_id

    def snapshots(self, dataset: str) -> list[SnapshotId]:
        """All snapshot ids for a dataset, oldest first."""
        base = self._root / dataset
        if not base.is_dir():
            return []
        return sorted(SnapshotId(p.name) for p in base.iterdir() if (p / "manifest.json").exists())

    def latest(self, dataset: str) -> SnapshotId:
        """The most recent snapshot id for a dataset."""
        ids = self.snapshots(dataset)
        if not ids:
            raise SnapshotNotFoundError(f"no snapshots for dataset {dataset!r}")
        return ids[-1]

    def read(
        self,
        dataset: str,
        frame: str,
        snapshot: SnapshotId | Literal["latest"] = "latest",
    ) -> pd.DataFrame:
        """Read one frame from a snapshot (default: the latest one)."""
        snap_id = self.latest(dataset) if snapshot == "latest" else snapshot
        path = self._root / dataset / snap_id / f"{frame}.parquet"
        if not path.exists():
            raise SnapshotNotFoundError(f"{dataset}/{snap_id} has no frame {frame!r}")
        return pd.read_parquet(path)

    def read_manifest(
        self, dataset: str, snapshot: SnapshotId | Literal["latest"] = "latest"
    ) -> dict[str, Any]:
        """Read a snapshot's manifest."""
        snap_id = self.latest(dataset) if snapshot == "latest" else snapshot
        path = self._root / dataset / snap_id / "manifest.json"
        if not path.exists():
            raise SnapshotNotFoundError(f"no manifest for {dataset}/{snap_id}")
        manifest: dict[str, Any] = json.loads(path.read_text())
        return manifest

    def matches_as_of(
        self,
        cutoff: dt.date,
        snapshot: SnapshotId | Literal["latest"] = "latest",
    ) -> pd.DataFrame:
        """Finished matches strictly before ``cutoff`` — THE walk-forward leak guard.

        Every ``Forecaster.fit`` call goes through here so that no model can see
        the future, including the match being predicted (kickoff date == cutoff
        is excluded).
        """
        frame = self.read(MATCHES_DATASET, "matches", snapshot)
        finished = frame["status"] == MatchStatus.FINISHED.value
        before = frame["date"] < pd.Timestamp(cutoff)
        return frame.loc[finished & before].sort_values(["date", "match_id"], ignore_index=True)
