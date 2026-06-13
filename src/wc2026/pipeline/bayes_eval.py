"""Phase 4 evaluation: Bayesian vs Dixon-Coles vs Elo on matched terms.

MCMC is too expensive to refit per match over the full 2010-2026 board, so the
Bayesian comparison runs on a recent window at a coarse refit cadence, and DC
and Elo are re-scored on the SAME matches at the SAME cadence — only then is a
paired ΔRPS test isolating the model difference honest. The window, cadence and
runtime are reported so the cost is explicit.

Also breaks the comparison down by how much (decayed) data each team has, to
test the headline claim: partial pooling should help most on sparse teams.
"""

import dataclasses
import datetime as dt
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from wc2026.eval.compare import compare
from wc2026.eval.walkforward import RefitSchedule, walk_forward
from wc2026.models.base import Forecaster
from wc2026.models.bayes_poisson import BayesConfig, BayesPoissonForecaster
from wc2026.models.dixon_coles import DixonColesForecaster
from wc2026.models.elo import EloForecaster

BAYES_WINDOW = (dt.date(2018, 1, 1), dt.date(2026, 6, 10))
BAYES_CADENCE_DAYS = 180


@dataclasses.dataclass(frozen=True)
class BayesComparison:
    """Outputs of the Phase 4 comparison run."""

    scoreboard: pd.DataFrame  # per-model rps/log_loss/brier on the shared window
    paired: pd.DataFrame  # paired ΔRPS vs DC and vs Elo
    sparse_split: pd.DataFrame  # Bayes-minus-DC ΔRPS by opponent-data tercile
    window: tuple[dt.date, dt.date]
    cadence_days: int


def _team_match_counts(
    matches: pd.DataFrame, as_of: dt.date, half_life_days: float
) -> dict[str, float]:
    """Decay-weighted appearance count per team up to ``as_of`` (the 'data richness')."""
    cutoff = pd.Timestamp(as_of)
    played = matches[(matches["status"] == "finished") & (matches["date"] < cutoff)]
    age = (cutoff - played["date"]).dt.days.to_numpy(dtype=np.float64)
    w = np.power(0.5, age / half_life_days)
    counts: dict[str, float] = {}
    for col in ("home_id", "away_id"):
        for team, wt in zip(played[col], w, strict=True):
            counts[team] = counts.get(team, 0.0) + float(wt)
    return counts


def run_bayes_comparison(
    matches: pd.DataFrame,
    *,
    seed: int,
    bayes_config: BayesConfig | None = None,
    window: tuple[dt.date, dt.date] = BAYES_WINDOW,
    cadence_days: int = BAYES_CADENCE_DAYS,
    cache_dir: Path | None = None,
) -> BayesComparison:
    """Run all three models walk-forward on the shared window; paired test + sparse split.

    The per-model walk-forward predictions are cached to ``cache_dir`` (one
    parquet per model, keyed by window + cadence) BEFORE any downstream
    analysis, so a bug in the analysis never wastes the expensive Bayesian
    MCMC refits — a re-run loads the cached rows instantly. Delete the cache
    to force a fresh fit on new data.
    """
    schedule = RefitSchedule(every_days=cadence_days)
    builders: dict[str, Callable[[], Forecaster]] = {
        "elo_baseline": EloForecaster,
        "dixon_coles": DixonColesForecaster,
        "bayes_poisson": lambda: BayesPoissonForecaster(bayes_config),
    }
    tag = f"{window[0]:%Y%m%d}_{window[1]:%Y%m%d}_c{cadence_days}"
    rows: dict[str, pd.DataFrame] = {}
    for name, build in builders.items():
        cache = (cache_dir / f"{name}_{tag}.parquet") if cache_dir else None
        if cache is not None and cache.exists():
            rows[name] = pd.read_parquet(cache)
            continue
        model = build()
        result = walk_forward(lambda m=model: m, matches, window, schedule)  # type: ignore[misc]
        rows[name] = result
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            result.to_parquet(cache)

    from wc2026.eval.report import scoreboard_row

    scoreboard = pd.DataFrame([scoreboard_row(name, rows[name], seed=seed) for name in builders])

    base = rows["dixon_coles"].set_index("match_id")
    elo = rows["elo_baseline"].set_index("match_id")
    bayes = rows["bayes_poisson"].set_index("match_id")
    shared = base.index.intersection(bayes.index).intersection(elo.index)
    paired = pd.DataFrame(
        [
            _paired_row(
                compare(
                    "bayes_poisson",
                    bayes.loc[shared, "rps"].to_numpy(),
                    "dixon_coles",
                    base.loc[shared, "rps"].to_numpy(),
                    metric="rps",
                    seed=seed,
                )
            ),
            _paired_row(
                compare(
                    "bayes_poisson",
                    bayes.loc[shared, "rps"].to_numpy(),
                    "elo_baseline",
                    elo.loc[shared, "rps"].to_numpy(),
                    metric="rps",
                    seed=seed,
                )
            ),
        ]
    )

    sparse_split = _sparse_split(matches, bayes.loc[shared], base.loc[shared], bayes_config)
    return BayesComparison(scoreboard, paired, sparse_split, window, cadence_days)


def _paired_row(cmp: object) -> dict[str, object]:
    c = cmp  # PairedComparison
    return {
        "comparison": f"{c.model_a} - {c.model_b}",  # type: ignore[attr-defined]
        "n": c.n,  # type: ignore[attr-defined]
        "mean_dRPS": c.mean_delta,  # type: ignore[attr-defined]
        "ci_lo": c.ci_lo,  # type: ignore[attr-defined]
        "ci_hi": c.ci_hi,  # type: ignore[attr-defined]
        "DM": c.dm_stat,  # type: ignore[attr-defined]
        "p": c.dm_pvalue,  # type: ignore[attr-defined]
        "verdict": (
            f"{c.winner} better (p={c.dm_pvalue:.1e})"  # type: ignore[attr-defined]
            if c.winner  # type: ignore[attr-defined]
            else "no significant difference"
        ),
    }


def _sparse_split(
    matches: pd.DataFrame,
    bayes_rows: pd.DataFrame,
    dc_rows: pd.DataFrame,
    bayes_config: BayesConfig | None,
) -> pd.DataFrame:
    """ΔRPS (Bayes - DC) split by the data-richness of the WEAKER side of each match.

    The headline: partial pooling should help most where a participating team is
    data-poor. Each shared match is bucketed by min(home, away) decayed match
    count as of that match's date; we report mean ΔRPS per tercile.
    """
    half_life = (bayes_config or BayesConfig()).half_life_days
    delta = (bayes_rows["rps"] - dc_rows["rps"]).rename("dRPS")
    meta = bayes_rows[["date", "home_id", "away_id"]].join(delta)

    # cache counts per refit-ish date is overkill; compute per unique date
    richness = []
    for date, group in meta.groupby("date"):
        counts = _team_match_counts(matches, cast(pd.Timestamp, date).date(), half_life)
        for home, away in zip(group["home_id"], group["away_id"], strict=True):
            richness.append(min(counts.get(home, 0.0), counts.get(away, 0.0)))
    meta = meta.assign(min_richness=richness)
    # rank-based terciles, robust to ties / few distinct values (qcut would fail)
    ranks = meta["min_richness"].rank(method="first", pct=True)
    meta["tercile"] = pd.cut(
        ranks, [0.0, 1 / 3, 2 / 3, 1.0], labels=["sparse", "mid", "rich"], include_lowest=True
    )
    out = meta.groupby("tercile", observed=True)["dRPS"].agg(["mean", "count"]).reset_index()
    return out
