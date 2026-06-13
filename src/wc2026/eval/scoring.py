"""Proper scoring rules, all returning per-match scores (lower is better).

Outcome encoding everywhere: 0 = home win, 1 = draw, 2 = away win, matching
the (home, draw, away) probability column order. The outcomes are ORDERED
(home > draw > away in home-team terms), which is why RPS — which scores the
cumulative distribution — is the primary metric.
"""

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

_EPS = 1e-15


def _one_hot(outcomes: IntArray, n_classes: int = 3) -> FloatArray:
    out = np.zeros((len(outcomes), n_classes), dtype=np.float64)
    out[np.arange(len(outcomes)), outcomes] = 1.0
    return out


def rps(probs: FloatArray, outcomes: IntArray) -> FloatArray:
    """Ranked Probability Score per match; (n, 3) probs, (n,) outcomes → (n,).

    Mean squared gap between predicted and realized CDFs over the ordered
    outcomes; in [0, 1] for 3 classes.
    """
    probs = np.asarray(probs, dtype=np.float64)
    cdf_pred = np.cumsum(probs, axis=1)[:, :-1]
    cdf_true = np.cumsum(_one_hot(np.asarray(outcomes)), axis=1)[:, :-1]
    result: FloatArray = ((cdf_pred - cdf_true) ** 2).mean(axis=1)
    return result


def log_loss(probs: FloatArray, outcomes: IntArray) -> FloatArray:
    """Negative log likelihood of the realized outcome, per match (nats)."""
    probs = np.asarray(probs, dtype=np.float64)
    picked = probs[np.arange(len(probs)), np.asarray(outcomes)]
    result: FloatArray = -np.log(np.clip(picked, _EPS, 1.0))
    return result


def brier(probs: FloatArray, outcomes: IntArray) -> FloatArray:
    """Multiclass Brier score per match; in [0, 2]."""
    probs = np.asarray(probs, dtype=np.float64)
    result: FloatArray = ((probs - _one_hot(np.asarray(outcomes))) ** 2).sum(axis=1)
    return result


def bootstrap_ci(per_match: FloatArray, n_boot: int = 10_000, *, seed: int) -> tuple[float, float]:
    """95% percentile-bootstrap CI for the mean of per-match scores."""
    values = np.asarray(per_match, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def outcome_codes(home_goals: IntArray, away_goals: IntArray) -> IntArray:
    """Encode results to 0/1/2 (home/draw/away)."""
    home = np.asarray(home_goals)
    away = np.asarray(away_goals)
    result: IntArray = np.where(home > away, 0, np.where(home == away, 1, 2)).astype(np.int64)
    return result
