"""Assemble the dashboard payload — one JSON snapshot the served app reads.

Built by the refresh pipeline (occasionally), not per request. Sections:

- ``upcoming``: each model + the blend + the de-vigged market line, side by side,
  for scheduled matches in the next fortnight.
- ``win_cup``: live champion/advancement probabilities from the DC simulator.
- ``model_board``: the paired walk-forward scoreboard (from the cached comparison).
- ``live_vs_market``: model/blend/market RPS on COMPLETED World Cup matches that
  have a stored closing-line proxy — leak-free (model predictions are walk-forward
  out-of-sample). Carries ``n`` so the UI can refuse to conclude while it is small.
- ``calibration``: reliability bins for the primary historical walk-forward.
"""

import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from wc2026.data.store import MATCHES_DATASET, Store
from wc2026.eval import calibration, scoring
from wc2026.eval.join import join_events_to_fixtures, load_all_quotes, unique_events
from wc2026.eval.market import BenchmarkPolicy, benchmark_probs, closing_quotes
from wc2026.eval.walkforward import RefitSchedule, walk_forward
from wc2026.models.base import Fixture
from wc2026.models.blend import BLEND_WEIGHTS, BlendForecaster
from wc2026.models.dixon_coles import DixonColesForecaster
from wc2026.models.elo import EloForecaster
from wc2026.models.gbm import GbmForecaster
from wc2026.pipeline.collect import WC_FIXTURES_DATASET
from wc2026.pipeline.evaluate import market_live_rows
from wc2026.tournament.simulate import simulate_tournament

WC2026_START = dt.date(2026, 6, 11)


def _f(x: Any) -> float:
    """Coerce a loosely-typed (itertuples) value to float."""
    return float(x)


def _ts(x: Any) -> pd.Timestamp:
    """Coerce a loosely-typed (itertuples) value to a pandas Timestamp."""
    return pd.Timestamp(x)


def _probs_dict(p: object) -> dict[str, float]:
    return {"home": round(p.home, 4), "draw": round(p.draw, 4), "away": round(p.away, 4)}  # type: ignore[attr-defined]


def build_payload(
    data_root: Path,
    *,
    seed: int,
    n_sims: int = 20_000,
    upcoming_days: int = 14,
    today: dt.date | None = None,
) -> dict[str, Any]:
    """Compute the full dashboard payload from the latest snapshots."""
    today = today or dt.datetime.now(dt.UTC).date()
    store = Store(data_root / "snapshots")
    matches = store.read(MATCHES_DATASET, "matches")
    fixtures = store.read(WC_FIXTURES_DATASET, "matches")

    elo, dc = EloForecaster(), DixonColesForecaster()
    gbm = GbmForecaster(matches)
    blend = BlendForecaster({"dixon_coles": dc, "gbm": gbm})
    for model in (elo, dc, gbm):
        model.fit(matches, as_of=today + dt.timedelta(days=1))

    payload: dict[str, Any] = {
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(),
        "blend_weights": BLEND_WEIGHTS,
        "upcoming": _upcoming(store, fixtures, today, upcoming_days, elo, dc, gbm, blend),
        "win_cup": _win_cup(matches, dc, n_sims, seed),
        "live_vs_market": _live_vs_market(store, matches, seed),
        "model_board": _model_board(data_root),
        "calibration": _calibration(matches, seed),
    }
    return payload


def _market_lookup(store: Store, fixtures: pd.DataFrame) -> dict[str, dict[str, float]]:
    """match_id -> de-vigged market 1X2 (home-oriented), from the closing-line proxy."""
    policy = BenchmarkPolicy()
    quotes = load_all_quotes(store)
    bench = benchmark_probs(closing_quotes(quotes, policy), policy)
    joined = join_events_to_fixtures(unique_events(quotes), fixtures)
    bench = bench.merge(joined, on="event_id", how="inner")
    out: dict[str, dict[str, float]] = {}
    for r in bench.itertuples(index=False):
        ph, pa = (r.p_away, r.p_home) if r.flipped else (r.p_home, r.p_away)
        out[str(r.match_id)] = {
            "home": round(_f(ph), 4),
            "draw": round(_f(r.p_draw), 4),
            "away": round(_f(pa), 4),
        }
    return out


def _upcoming(
    store: Store,
    fixtures: pd.DataFrame,
    today: dt.date,
    days: int,
    elo: EloForecaster,
    dc: DixonColesForecaster,
    gbm: GbmForecaster,
    blend: BlendForecaster,
) -> list[dict[str, Any]]:
    market = _market_lookup(store, fixtures)
    upcoming = fixtures[
        (fixtures["status"] == "scheduled")
        & (fixtures["date"] >= pd.Timestamp(today))
        & (fixtures["date"] <= pd.Timestamp(today + dt.timedelta(days=days)))
    ].sort_values("date")
    rows: list[dict[str, Any]] = []
    for m in upcoming.itertuples(index=False):
        fx = Fixture(str(m.home_id), str(m.away_id), _ts(m.date).date(), neutral=bool(m.neutral))
        rows.append(
            {
                "match_id": str(m.match_id),
                "date": _ts(m.date).date().isoformat(),
                "home": str(m.home_id),
                "away": str(m.away_id),
                "elo": _probs_dict(elo.predict(fx)),
                "dixon_coles": _probs_dict(dc.predict(fx)),
                "gbm": _probs_dict(gbm.predict(fx)),
                "blend": _probs_dict(blend.predict(fx)),
                "market": market.get(str(m.match_id)),
            }
        )
    return rows


def _win_cup(
    matches: pd.DataFrame, dc: DixonColesForecaster, n_sims: int, seed: int
) -> list[dict[str, Any]]:
    table = simulate_tournament(matches, dc, n_sims=n_sims, rng=np.random.default_rng(seed))
    cols = ["reach_r16", "reach_qf", "reach_sf", "reach_final", "champion"]
    return [
        {
            "team": str(r.team_id),
            "group": str(r.group),
            **{c: round(_f(getattr(r, c)) * 100, 1) for c in cols},
        }
        for r in table.head(24).itertuples(index=False)
    ]


def _live_vs_market(store: Store, matches: pd.DataFrame, seed: int) -> dict[str, Any]:
    """Leak-free model/blend/market RPS on completed WC matches with stored lines."""
    today = dt.datetime.now(dt.UTC).date()
    completed = matches[
        (matches["status"] == "finished") & (matches["date"] >= pd.Timestamp(WC2026_START))
    ]
    market_rows = market_live_rows(store, completed)
    if market_rows.empty:
        return {
            "n": 0,
            "note": "no completed World Cup match has a stored pre-kickoff line yet",
            "scoreboard": [],
        }

    window = (WC2026_START, today)
    sched = RefitSchedule(every_days=1)
    preds = {
        "elo_baseline": walk_forward(EloForecaster, matches, window, sched),
        "dixon_coles": walk_forward(DixonColesForecaster, matches, window, sched),
    }
    gbm = GbmForecaster(matches)
    preds["gbm"] = walk_forward(lambda m=gbm: m, matches, window, sched)  # type: ignore[misc]

    shared = set(market_rows["match_id"])
    for df in preds.values():
        shared &= set(df["match_id"])
    shared_ids = sorted(shared)
    if not shared_ids:
        return {
            "n": 0,
            "note": "no overlap between completed-with-lines and model predictions yet",
            "scoreboard": [],
        }

    indexed = {name: df.set_index("match_id").loc[shared_ids] for name, df in preds.items()}
    mkt = market_rows.set_index("match_id").loc[shared_ids]
    outcomes = mkt["outcome"].to_numpy(dtype=np.int64)

    def rps_of(p: "npt.NDArray[np.float64]") -> float:
        return float(scoring.rps(p, outcomes).mean())

    dc_p = indexed["dixon_coles"][["p_home", "p_draw", "p_away"]].to_numpy(np.float64)
    gbm_p = indexed["gbm"][["p_home", "p_draw", "p_away"]].to_numpy(np.float64)
    w = BLEND_WEIGHTS
    tot = sum(w.values())
    blend_p = (w["dixon_coles"] * dc_p + w["gbm"] * gbm_p) / tot
    board = [
        {
            "model": "market",
            "rps": round(rps_of(mkt[["p_home", "p_draw", "p_away"]].to_numpy(np.float64)), 4),
        },
        {"model": "blend", "rps": round(rps_of(blend_p), 4)},
        {"model": "dixon_coles", "rps": round(rps_of(dc_p), 4)},
        {
            "model": "elo_baseline",
            "rps": round(
                rps_of(
                    indexed["elo_baseline"][["p_home", "p_draw", "p_away"]].to_numpy(np.float64)
                ),
                4,
            ),
        },
        {"model": "gbm", "rps": round(rps_of(gbm_p), 4)},
    ]
    return {
        "n": len(shared_ids),
        "note": "small sample — no conclusions" if len(shared_ids) < 30 else "",
        "scoreboard": sorted(board, key=lambda r: r["rps"]),
    }


def _model_board(data_root: Path) -> list[dict[str, Any]]:
    """The four-model paired scoreboard from the cached walk-forward predictions."""
    cache = data_root / "bayes_cache"
    tag = "20180101_20260610_c180"
    board: list[dict[str, Any]] = []
    for name in ("elo_baseline", "dixon_coles", "bayes_poisson", "gbm"):
        path = cache / f"{name}_{tag}.parquet"
        if not path.exists():
            continue
        rps = pd.read_parquet(path)["rps"].mean()
        board.append({"model": name, "rps": round(float(rps), 4)})
    return sorted(board, key=lambda r: r["rps"])


def _calibration(matches: pd.DataFrame, seed: int) -> dict[str, Any]:
    """Reliability bins for DC on a recent walk-forward slice."""
    window = (dt.date(2022, 1, 1), dt.date(2026, 6, 10))
    result = walk_forward(DixonColesForecaster, matches, window, RefitSchedule(every_days=90))
    if result.empty:
        return {"bins": [], "ece": None}
    probs = result[["p_home", "p_draw", "p_away"]].to_numpy(np.float64)
    outcomes = result["outcome"].to_numpy(np.int64)
    table = calibration.reliability_table(probs, outcomes, bins=10)
    return {
        "ece": round(calibration.ece(probs, outcomes), 4),
        "bins": [
            {"p_pred": round(_f(r.p_pred), 3), "p_emp": round(_f(r.p_emp), 3), "n": int(_f(r.n))}
            for r in table.itertuples(index=False)
        ],
    }
