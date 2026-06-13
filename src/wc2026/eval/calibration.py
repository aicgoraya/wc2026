"""Calibration: reliability tables/diagrams and expected calibration error.

Multiclass probabilities are pooled one-vs-rest: every match contributes one
(predicted probability, did it happen) pair per outcome class, the standard
construction for 1X2 reliability.
"""

import matplotlib
import numpy as np
import numpy.typing as npt
import pandas as pd

matplotlib.use("Agg")  # headless: reports render to files, never to a window
from matplotlib.figure import Figure

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def _pooled(probs: FloatArray, outcomes: IntArray) -> tuple[FloatArray, FloatArray]:
    probs = np.asarray(probs, dtype=np.float64)
    outcomes = np.asarray(outcomes)
    hit = np.zeros_like(probs)
    hit[np.arange(len(outcomes)), outcomes] = 1.0
    return probs.ravel(), hit.ravel()


def reliability_table(probs: FloatArray, outcomes: IntArray, bins: int = 10) -> pd.DataFrame:
    """Equal-width bins over pooled one-vs-rest pairs.

    Columns: bin_low, bin_high, n, p_pred (mean predicted), p_emp (empirical
    frequency). Empty bins are dropped.
    """
    pred, hit = _pooled(probs, outcomes)
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(pred, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = which == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin_low": edges[b],
                "bin_high": edges[b + 1],
                "n": int(mask.sum()),
                "p_pred": float(pred[mask].mean()),
                "p_emp": float(hit[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def ece(probs: FloatArray, outcomes: IntArray, bins: int = 10) -> float:
    """Expected calibration error: count-weighted |p_pred - p_emp| over bins."""
    table = reliability_table(probs, outcomes, bins)
    weights = table["n"] / table["n"].sum()
    return float((weights * (table["p_pred"] - table["p_emp"]).abs()).sum())


def reliability_plot(probs: FloatArray, outcomes: IntArray, *, title: str) -> Figure:
    """Reliability diagram (pooled one-vs-rest) with normal-approx error bars."""
    table = reliability_table(probs, outcomes)
    fig = Figure(figsize=(5.0, 5.0), dpi=120)
    ax = fig.add_subplot()
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="grey", label="perfect")
    se = np.sqrt(table["p_emp"] * (1 - table["p_emp"]) / table["n"])
    ax.errorbar(
        table["p_pred"],
        table["p_emp"],
        yerr=1.96 * se,
        fmt="o",
        markersize=4,
        capsize=3,
        label="observed",
    )
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("empirical frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig
