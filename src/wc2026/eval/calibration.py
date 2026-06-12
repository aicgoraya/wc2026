"""Calibration: reliability tables/diagrams and expected calibration error."""

import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.figure import Figure

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def reliability_table(probs: FloatArray, outcomes: IntArray, bins: int = 10) -> pd.DataFrame:
    """Binned predicted-vs-empirical frequencies with counts, pooled over classes."""
    raise NotImplementedError("ships in Phase 1c")


def ece(probs: FloatArray, outcomes: IntArray, bins: int = 10) -> float:
    """Expected calibration error (count-weighted |predicted - empirical|)."""
    raise NotImplementedError("ships in Phase 1c")


def reliability_plot(probs: FloatArray, outcomes: IntArray, *, title: str) -> Figure:
    """Reliability diagram with per-bin confidence bands."""
    raise NotImplementedError("ships in Phase 1c")
