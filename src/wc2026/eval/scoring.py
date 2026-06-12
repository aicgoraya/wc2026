"""Proper scoring rules, all returning per-match scores (lower is better).

Outcome encoding everywhere: 0 = home win, 1 = draw, 2 = away win, matching
the (home, draw, away) probability column order.
"""

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def rps(probs: FloatArray, outcomes: IntArray) -> FloatArray:
    """Ranked Probability Score per match; (n, 3) probs, (n,) outcomes → (n,)."""
    raise NotImplementedError("ships in Phase 1c")


def log_loss(probs: FloatArray, outcomes: IntArray) -> FloatArray:
    """Negative log likelihood of the realized outcome, per match."""
    raise NotImplementedError("ships in Phase 1c")


def brier(probs: FloatArray, outcomes: IntArray) -> FloatArray:
    """Multiclass Brier score per match."""
    raise NotImplementedError("ships in Phase 1c")


def bootstrap_ci(per_match: FloatArray, n_boot: int = 10_000, *, seed: int) -> tuple[float, float]:
    """95% bootstrap confidence interval for the mean of per-match scores."""
    raise NotImplementedError("ships in Phase 1c")
