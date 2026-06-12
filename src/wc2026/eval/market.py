"""The betting market as a benchmark model: de-vig odds → implied probabilities.

The market is scored on the same walk-forward scoreboard as every model. Quotes
come from stored snapshots taken at a consistent time before kickoff so the
benchmark is honest.
"""

import dataclasses

import numpy as np
import numpy.typing as npt

from wc2026.data.store import Store

FloatArray = npt.NDArray[np.float64]


def devig_proportional(odds: FloatArray) -> FloatArray:
    """Normalize inverse odds to sum to 1; (n, 3) decimal odds → (n, 3) probs."""
    raise NotImplementedError("ships in Phase 1c")


def devig_shin(odds: FloatArray) -> FloatArray:
    """Shin-method de-vig (accounts for insider-trading skew on longshots)."""
    raise NotImplementedError("ships in Phase 1c")


@dataclasses.dataclass(frozen=True)
class SnapshotPolicy:
    """Which stored quote to use per match: latest at most ``min_hours`` before kickoff."""

    min_hours_before_kickoff: float = 1.0


class MarketForecaster:
    """Consensus de-vigged market probabilities as a ``Forecaster``."""

    name = "market"

    def __init__(self, store: Store, policy: SnapshotPolicy) -> None:
        self._store = store
        self._policy = policy
