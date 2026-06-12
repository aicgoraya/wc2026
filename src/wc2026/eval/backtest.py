"""Paper-trading backtest — a simulated analytical exercise, not real wagering.

Where a model's probability diverges from the de-vigged market by more than an
edge threshold, record the hypothetical position at fractional-Kelly size and
track the equity curve. Strictly out-of-sample. The point is to demonstrate
understanding of edge, calibration and sizing — not to claim a money machine.
"""

import dataclasses

import pandas as pd

from wc2026.eval.market import MarketForecaster


@dataclasses.dataclass(frozen=True)
class BacktestConfig:
    """Edge gate, Kelly fraction, stake cap, and RNG seed for any tie-breaking."""

    edge_threshold: float
    kelly_fraction: float
    max_stake: float
    seed: int


@dataclasses.dataclass(frozen=True)
class BacktestResult:
    """Equity curve and summary risk/return metrics of the simulated strategy."""

    equity_curve: "pd.Series[float]"
    n_bets: int
    total_ev: float
    max_drawdown: float
    sharpe_like: float


def backtest(
    model_rows: pd.DataFrame,
    market: MarketForecaster,
    cfg: BacktestConfig,
) -> BacktestResult:
    """Run the simulated strategy over walk-forward predictions."""
    raise NotImplementedError("ships in Phase 1c (flat-stake EV) / Phase 2 (Kelly)")
