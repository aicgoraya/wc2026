"""Paired model comparison on per-match losses.

Comparing two models by their marginal score CIs is the wrong test: those CIs
share the same matches, so they overlap even when one model is reliably better
match-for-match. The correct object is the per-match loss DIFFERENCE on the
SHARED matches. This module provides:

- a percentile bootstrap of the mean per-match ``delta = loss_a - loss_b``
  (resampling matches, preserving the pairing), and
- the Diebold-Mariano statistic for equal predictive accuracy.

Sign convention: ``delta = loss_a - loss_b``, so a NEGATIVE mean delta and a
CI/DM that excludes 0 means model A is significantly better (lower loss).
"""

import dataclasses

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

FloatArray = npt.NDArray[np.float64]


@dataclasses.dataclass(frozen=True)
class PairedComparison:
    """Paired comparison of model A vs model B on shared per-match losses."""

    model_a: str
    model_b: str
    metric: str
    n: int
    mean_delta: float  # mean(loss_a - loss_b); < 0 => A better
    ci_lo: float
    ci_hi: float
    dm_stat: float  # Diebold-Mariano; > 0 => A worse (higher loss)
    dm_pvalue: float  # two-sided

    @property
    def significant(self) -> bool:
        """True when the paired 95% bootstrap CI excludes zero."""
        return self.ci_lo > 0.0 or self.ci_hi < 0.0

    @property
    def winner(self) -> str | None:
        """The better model if the difference is significant, else None."""
        if not self.significant:
            return None
        return self.model_a if self.mean_delta < 0 else self.model_b


def paired_bootstrap_delta(
    loss_a: FloatArray, loss_b: FloatArray, n_boot: int = 10_000, *, seed: int
) -> tuple[float, float, float]:
    """Bootstrap the mean paired loss difference (A - B); returns (mean, lo, hi).

    Resamples MATCHES (not the two models independently), so the pairing is
    preserved and shared-match correlation cancels. ``loss_a`` and ``loss_b``
    must be aligned element-wise on the same matches.
    """
    a = np.asarray(loss_a, dtype=np.float64)
    b = np.asarray(loss_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError(f"losses must be aligned 1-D arrays, got {a.shape} and {b.shape}")
    delta = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(n_boot, len(delta)))
    means = delta[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(delta.mean()), float(lo), float(hi)


def diebold_mariano(loss_a: FloatArray, loss_b: FloatArray) -> tuple[float, float]:
    """Diebold-Mariano statistic and two-sided p-value for equal accuracy.

    ``d = loss_a - loss_b``; ``DM = mean(d) / sqrt(var(d) / n)`` referred to the
    standard normal. Matches are treated as INDEPENDENT (a cross-section, not a
    single autocorrelated series), so no HAC variance term is applied — this is
    the h=1, no-serial-correlation case. ``DM > 0`` means A has higher loss.
    """
    d = np.asarray(loss_a, dtype=np.float64) - np.asarray(loss_b, dtype=np.float64)
    n = len(d)
    if n < 2:
        return 0.0, 1.0
    var_d = float(d.var(ddof=1))
    if var_d <= 0.0:
        return 0.0, 1.0
    stat = float(d.mean() / np.sqrt(var_d / n))
    pvalue = float(2.0 * norm.sf(abs(stat)))
    return stat, pvalue


def compare(
    model_a: str,
    loss_a: FloatArray,
    model_b: str,
    loss_b: FloatArray,
    *,
    metric: str,
    seed: int,
    n_boot: int = 10_000,
) -> PairedComparison:
    """Full paired comparison: bootstrap CI of the mean delta plus DM test."""
    mean_delta, lo, hi = paired_bootstrap_delta(loss_a, loss_b, n_boot, seed=seed)
    dm_stat, dm_p = diebold_mariano(loss_a, loss_b)
    return PairedComparison(
        model_a=model_a,
        model_b=model_b,
        metric=metric,
        n=len(loss_a),
        mean_delta=mean_delta,
        ci_lo=lo,
        ci_hi=hi,
        dm_stat=dm_stat,
        dm_pvalue=dm_p,
    )
