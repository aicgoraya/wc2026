"""Daily refresh: pull → snapshot → refit → re-simulate → regenerate RESULTS.md.

Idempotent: re-running on the same day's data produces an identical snapshot
chain (timestamps aside) and identical results, seeds included.
"""

from collections.abc import Sequence

from wc2026.data.sources.base import Source
from wc2026.data.store import SnapshotId, Store
from wc2026.models.base import Forecaster


def refresh(
    store: Store,
    sources: Sequence[Source],
    models: Sequence[Forecaster],
    n_sims: int,
    seed: int,
) -> SnapshotId:
    """Run the full pipeline; returns the id of the results snapshot it wrote."""
    raise NotImplementedError("assembled incrementally from Phase 1a onward")
