"""Phase 5 evaluation: GBM vs the generative ladder + the ensemble.

Runs all four models walk-forward on the shared 2018-2026 window at the matched
180-day cadence (the Elo/DC/Bayes rows are reused from the Phase-4 cache, so
only the fast GBM is computed fresh), then:

- a paired ΔRPS board of the GBM against DC, Bayes and Elo, and
- a leak-free time-split ensemble (convex blend of all four) whose weights are
  fit on an earlier window and tested on a strictly later one.

The honest question is whether either the GBM or the blend beats the best single
generative model — reported straight either way.
"""

import dataclasses
import datetime as dt
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from wc2026.eval.compare import compare
from wc2026.eval.ensemble import (
    EnsembleResult,
    WalkForwardEnsembleResult,
    evaluate_ensemble,
    walk_forward_ensemble,
)
from wc2026.eval.report import scoreboard_row
from wc2026.eval.walkforward import RefitSchedule, walk_forward
from wc2026.models.base import Forecaster
from wc2026.models.bayes_poisson import BayesPoissonForecaster
from wc2026.models.dixon_coles import DixonColesForecaster
from wc2026.models.elo import EloForecaster
from wc2026.models.gbm import GbmForecaster

WINDOW = (dt.date(2018, 1, 1), dt.date(2026, 6, 10))
CADENCE_DAYS = 180
ENSEMBLE_SPLIT = dt.date(2022, 6, 1)


@dataclasses.dataclass(frozen=True)
class ModelComparison:
    """Four-model scoreboard, paired-vs-DC board, and the ensemble result."""

    scoreboard: pd.DataFrame
    paired: pd.DataFrame
    ensemble: EnsembleResult
    rolling_ensemble: WalkForwardEnsembleResult
    window: tuple[dt.date, dt.date]
    cadence_days: int


def run_model_comparison(
    matches: pd.DataFrame,
    *,
    seed: int,
    cache_dir: Path,
    window: tuple[dt.date, dt.date] = WINDOW,
    cadence_days: int = CADENCE_DAYS,
    split_date: dt.date = ENSEMBLE_SPLIT,
) -> ModelComparison:
    """Full four-model paired board + leak-free ensemble; caches per-model rows."""
    schedule = RefitSchedule(every_days=cadence_days)
    gbm = GbmForecaster(matches)
    builders: dict[str, Callable[[], Forecaster]] = {
        "elo_baseline": EloForecaster,
        "dixon_coles": DixonColesForecaster,
        "bayes_poisson": BayesPoissonForecaster,
        "gbm": lambda: gbm,
    }
    tag = f"{window[0]:%Y%m%d}_{window[1]:%Y%m%d}_c{cadence_days}"
    rows: dict[str, pd.DataFrame] = {}
    for name, build in builders.items():
        cache = cache_dir / f"{name}_{tag}.parquet"
        if cache.exists():
            rows[name] = pd.read_parquet(cache)
            continue
        model = build()
        result = walk_forward(lambda m=model: m, matches, window, schedule)  # type: ignore[misc]
        rows[name] = result
        cache.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(cache)

    scoreboard = pd.DataFrame([scoreboard_row(name, rows[name], seed=seed) for name in builders])

    indexed = {name: df.set_index("match_id") for name, df in rows.items()}
    shared = indexed["dixon_coles"].index
    for name in indexed:
        shared = shared.intersection(indexed[name].index)
    paired_rows = []
    for challenger in ("gbm", "bayes_poisson", "elo_baseline"):
        cmp = compare(
            challenger,
            indexed[challenger].loc[shared, "rps"].to_numpy(),
            "dixon_coles",
            indexed["dixon_coles"].loc[shared, "rps"].to_numpy(),
            metric="rps",
            seed=seed,
        )
        paired_rows.append(
            {
                "comparison": f"{cmp.model_a} - dixon_coles",
                "n": cmp.n,
                "mean_dRPS": cmp.mean_delta,
                "ci_lo": cmp.ci_lo,
                "ci_hi": cmp.ci_hi,
                "p": cmp.dm_pvalue,
                "verdict": (
                    f"{cmp.winner} better (p={cmp.dm_pvalue:.1e})"
                    if cmp.winner
                    else "no significant difference"
                ),
            }
        )

    ensemble = evaluate_ensemble(rows, split_date, seed=seed)
    rolling = walk_forward_ensemble(rows, cadence_days=cadence_days, min_train=2000, seed=seed)
    return ModelComparison(
        scoreboard, pd.DataFrame(paired_rows), ensemble, rolling, window, cadence_days
    )
