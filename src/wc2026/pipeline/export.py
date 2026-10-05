"""Offline export: train once, evaluate, and write an immutable model artifact.

This is the only place the serving models are fitted. The service
(``wc2026.serving``) loads what this writes and never trains.

``train_and_write`` fits the production pair (Dixon-Coles + LightGBM) on all
finished matches strictly before a cutoff, captures the per-team feature state
at that cutoff, and writes a bundle (see ``serving.artifacts``). Before it
returns it reloads the bundle through the serving code path and checks that
served probabilities equal the in-memory pipeline's on probe fixtures.

``build_evaluation`` produces ``evaluation.json`` from real walk-forward runs:

- ``historical``: Elo / Dixon-Coles / LightGBM on the shared 2018+ window and
  the rolling-weights blend. There is NO market baseline on this track
  (historical closing odds were never collected).
- ``market``: the same models and the fixed production blend against the
  de-vigged closing-line proxy, on the World Cup 2026 matches that have a
  stored pre-kickoff quote.

The Bayesian model is not part of the served blend (its fitted weight was 0)
and is excluded here because its MCMC refits take hours; its comparison stays
in ``results/bayes_comparison.md``.
"""

import datetime as dt
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wc2026.data.schema import MATCH_COLUMNS
from wc2026.data.store import MATCHES_DATASET, Store
from wc2026.eval import scoring
from wc2026.eval.compare import compare
from wc2026.eval.ensemble import walk_forward_ensemble
from wc2026.eval.market import BenchmarkPolicy
from wc2026.eval.report import scoreboard_row
from wc2026.eval.walkforward import RefitSchedule, walk_forward
from wc2026.features.build import team_states_as_of
from wc2026.models.base import Fixture, Forecaster
from wc2026.models.blend import BLEND_WEIGHTS, BlendForecaster
from wc2026.models.dixon_coles import DEFAULT_HALF_LIFE_DAYS, DEFAULT_L2, DixonColesForecaster
from wc2026.models.elo import EloForecaster
from wc2026.models.gbm import DEFAULT_PARAMS, GbmForecaster
from wc2026.pipeline.ensemble_eval import CADENCE_DAYS, WINDOW
from wc2026.pipeline.evaluate import WC2026_START, market_live_rows
from wc2026.serving.artifacts import Manifest, write_bundle
from wc2026.serving.inference import load_artifact

DEFAULT_HORIZON_DAYS = CADENCE_DAYS
"""Predictions are accepted up to this many days past the training cutoff. It
equals the refit cadence of the walk-forward evaluation, so the service is
never staler than the models whose scores are reported."""

PARITY_ATOL = 1e-9
_PROBE_TOURNAMENT = "friendly"


class ExportError(RuntimeError):
    """The export could not produce a valid artifact."""


def _records(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    """DataFrame rows as JSON-safe dicts (numpy scalars and timestamps converted)."""
    out: list[dict[str, Any]] = []
    for row in frame[columns].to_dict("records"):
        clean: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, pd.Timestamp | dt.date):
                clean[str(key)] = value.isoformat()[:10]
            elif isinstance(value, np.integer | int):
                clean[str(key)] = int(value)
            elif isinstance(value, np.floating | float):
                clean[str(key)] = float(value)
            else:
                clean[str(key)] = None if value is None else str(value)
        out.append(clean)
    return out


def _probe_rows(history: pd.DataFrame, cutoff: dt.date, n: int) -> pd.DataFrame:
    """Scheduled probe fixtures at the cutoff between the most recently active teams.

    Scheduled rows never update team state or enter training; they only make
    the in-memory GBM compute features for these fixtures so the served
    predictions can be compared against it.
    """
    recent = history.sort_values("date").tail(400)
    teams = list(dict.fromkeys([*recent["home_id"], *recent["away_id"]]))[: n + 1]
    rows = []
    for i in range(len(teams) - 1):
        row: dict[str, Any] = dict.fromkeys(MATCH_COLUMNS)
        row.update(
            match_id=f"probe_{i}",
            date=pd.Timestamp(cutoff),
            home_id=teams[i],
            away_id=teams[i + 1],
            neutral=bool(i % 2),
            tournament=_PROBE_TOURNAMENT,
            status="scheduled",
            went_to_shootout=False,
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=list(MATCH_COLUMNS))


def train_and_write(
    matches: pd.DataFrame,
    bundle_dir: Path,
    *,
    version: str,
    cutoff: dt.date,
    synthetic: bool,
    training_data: dict[str, Any],
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    evaluation: dict[str, Any] | None = None,
    gbm_params: dict[str, Any] | None = None,
    n_probes: int = 8,
) -> Manifest:
    """Fit the production models strictly before ``cutoff`` and write the bundle."""
    if bundle_dir.exists():
        raise FileExistsError(f"{bundle_dir} already exists; artifact versions are immutable")
    finished = matches[(matches["status"] == "finished") & (matches["date"] < pd.Timestamp(cutoff))]
    if finished.empty:
        raise ExportError(f"no finished matches before {cutoff}")
    probes = _probe_rows(finished, cutoff, n_probes)
    with_probes = pd.concat([finished, probes.astype(finished.dtypes.to_dict())], ignore_index=True)

    dc = DixonColesForecaster()
    dc.fit(finished, as_of=cutoff)
    gbm = GbmForecaster(with_probes, params=gbm_params)
    gbm.fit(with_probes, as_of=cutoff)  # trains on labelled rows < cutoff only
    states = team_states_as_of(finished, cutoff)

    manifest = write_bundle(
        bundle_dir,
        version=version,
        synthetic=synthetic,
        training_cutoff=cutoff,
        max_prediction_date=cutoff + dt.timedelta(days=horizon_days),
        dc_params=dc.params,
        dc_hyperparams={
            "half_life_days": DEFAULT_HALF_LIFE_DAYS,
            "l2": DEFAULT_L2,
            "max_goals": 10,
            "train_window_years": 25,
        },
        gbm_model_string=gbm.booster_string(),
        gbm_params=gbm_params or DEFAULT_PARAMS,
        team_states=states,
        blend_weights=BLEND_WEIGHTS,
        training_data={
            **training_data,
            "n_finished_matches_before_cutoff": len(finished),
            "last_match_date": finished["date"].max().date().isoformat(),
        },
        evaluation=evaluation,
    )

    # Parity gate: what the service will return must equal the existing pipeline.
    served = load_artifact(bundle_dir, expected_version=version).predictor
    reference = BlendForecaster({"dixon_coles": dc, "gbm": gbm})
    for probe in probes.itertuples(index=False):
        fixture = Fixture(
            str(probe.home_id), str(probe.away_id), cutoff, neutral=bool(probe.neutral)
        )
        want = reference.predict(fixture)
        got = served.predict(
            fixture.home_id,
            fixture.away_id,
            cutoff,
            neutral=fixture.neutral,
            tournament=_PROBE_TOURNAMENT,
        ).blend
        if not np.allclose(got.as_array(), want.as_array(), atol=PARITY_ATOL, rtol=0.0):
            raise ExportError(
                f"served prediction differs from the pipeline for {fixture.home_id} v"
                f" {fixture.away_id}: {got} != {want}"
            )
    return manifest


def historical_evaluation(
    matches: pd.DataFrame,
    *,
    window: tuple[dt.date, dt.date],
    cadence_days: int,
    min_train: int,
    seed: int,
) -> dict[str, Any]:
    """Walk-forward scoreboard for Elo/DC/GBM plus the rolling-weights blend."""
    schedule = RefitSchedule(every_days=cadence_days)
    gbm = GbmForecaster(matches)
    builders: dict[str, Callable[[], Forecaster]] = {
        "elo_baseline": EloForecaster,
        "dixon_coles": DixonColesForecaster,
        "gbm": lambda: gbm,
    }
    rows = {
        name: walk_forward(build, matches, window, schedule) for name, build in builders.items()
    }
    board = pd.DataFrame([scoreboard_row(name, rows[name], seed=seed) for name in builders])
    rolling = walk_forward_ensemble(rows, cadence_days=cadence_days, min_train=min_train, seed=seed)
    final_weights = (
        {k: float(v) for k, v in rolling.weights_trajectory.iloc[-1].items() if k in builders}
        if len(rolling.weights_trajectory)
        else {}
    )
    return {
        "description": "Walk-forward evaluation on all international matches in the window."
        " Every prediction comes from a model refit only on matches strictly before its"
        " refit cutoff.",
        "window": {"start": window[0].isoformat(), "end": window[1].isoformat()},
        "refit_cadence_days": cadence_days,
        "baseline": None,
        "baseline_note": "No market baseline on this track: closing odds were not collected"
        " for these matches. elo_baseline is a simple model reference, not a market line.",
        "models": _records(
            board, ["model", "n", "rps", "rps_ci_lo", "rps_ci_hi", "log_loss", "brier", "ece"]
        ),
        "blend": {
            "method": f"Convex blend of the three models; weights refit every {cadence_days}"
            " days on expanding, strictly earlier out-of-sample predictions (at least"
            f" {min_train}) and applied to the next block.",
            "n": rolling.n_oos,
            "scoreboard": _records(rolling.scoreboard, ["model", "rps"]),
            "paired": _records(
                rolling.paired, ["comparison", "n", "mean_dRPS", "ci_lo", "ci_hi", "p", "verdict"]
            ),
            "last_fitted_weights": final_weights,
        },
    }


def market_evaluation(store: Store, matches: pd.DataFrame, *, seed: int) -> dict[str, Any]:
    """Models and the fixed production blend vs the de-vigged closing-line proxy."""
    policy = BenchmarkPolicy()
    base: dict[str, Any] = {
        "description": "FIFA World Cup 2026 matches with a stored pre-kickoff odds quote."
        " Model predictions are walk-forward (refit daily, strictly before each match).",
        "baseline": "market",
        "baseline_definition": "De-vigged (proportional) 1X2 closing-line proxy: the last odds"
        " snapshot stored before kickoff, from the preferred sharp book when present"
        f" ({', '.join(policy.preferred_books)}), else the mean of the sharp set, else the"
        f" {policy.consensus_size} lowest-margin books.",
        "caveats": [
            "Odds were snapshotted every 6 hours, so the proxy can trail the true closing"
            " line by up to ~6 hours.",
            "One tournament; the sample is small. See n and the confidence intervals.",
        ],
    }
    completed = matches[
        (matches["status"] == "finished")
        & (matches["tournament"] == "fifa_world_cup")
        & (matches["date"] >= pd.Timestamp(WC2026_START))
    ]
    market_rows = market_live_rows(store, completed) if len(completed) else pd.DataFrame()
    if market_rows.empty:
        return {
            **base,
            "available": False,
            "n": 0,
            "reason": "no completed match has a stored line",
        }

    window = (WC2026_START, completed["date"].max().date())
    schedule = RefitSchedule(every_days=1)
    gbm = GbmForecaster(matches)
    preds: dict[str, pd.DataFrame] = {
        "elo_baseline": walk_forward(EloForecaster, matches, window, schedule),
        "dixon_coles": walk_forward(DixonColesForecaster, matches, window, schedule),
        "gbm": walk_forward(lambda: gbm, matches, window, schedule),
    }
    shared = set(market_rows["match_id"])
    for frame in preds.values():
        shared &= set(frame["match_id"])
    ids = sorted(shared)
    if not ids:
        return {**base, "available": False, "n": 0, "reason": "no overlap with model predictions"}

    cols = ["p_home", "p_draw", "p_away"]
    market = market_rows.drop_duplicates("match_id").set_index("match_id").loc[ids]
    outcomes = market["outcome"].to_numpy(dtype=np.int64)
    probs = {
        name: f.set_index("match_id").loc[ids, cols].to_numpy(np.float64)
        for name, f in preds.items()
    }
    total = sum(BLEND_WEIGHTS.values())
    probs["blend"] = sum(w / total * probs[name] for name, w in BLEND_WEIGHTS.items())  # type: ignore[assignment]
    probs["market"] = market[cols].to_numpy(np.float64)

    per_match = {name: scoring.rps(p, outcomes) for name, p in probs.items()}
    board = []
    for name, loss in per_match.items():
        lo, hi = scoring.bootstrap_ci(loss, seed=seed)
        board.append(
            {
                "model": name,
                "kind": "market_baseline" if name == "market" else "model",
                "n": len(ids),
                "rps": float(loss.mean()),
                "rps_ci_lo": lo,
                "rps_ci_hi": hi,
                "log_loss": float(scoring.log_loss(probs[name], outcomes).mean()),
            }
        )
    paired = []
    for name in ("blend", "dixon_coles", "gbm", "elo_baseline"):
        cmp = compare(name, per_match[name], "market", per_match["market"], metric="rps", seed=seed)
        paired.append(
            {
                "comparison": f"{name} - market",
                "n": cmp.n,
                "mean_dRPS": cmp.mean_delta,
                "ci_lo": cmp.ci_lo,
                "ci_hi": cmp.ci_hi,
                "p": cmp.dm_pvalue,
                "verdict": f"{cmp.winner} better" if cmp.winner else "no significant difference",
            }
        )
    return {
        **base,
        "available": True,
        "window": {"start": window[0].isoformat(), "end": window[1].isoformat()},
        "n": len(ids),
        "n_tournament_matches": len(completed),
        "coverage_note": "Matches without a stored pre-kickoff quote are excluded (odds"
        " collection began on 2026-06-12, after the opening matches).",
        "blend_weights": dict(BLEND_WEIGHTS),
        "blend_note": "Fixed production weights, frozen on 2026-06-13 from pre-tournament data.",
        "scoreboard": sorted(board, key=lambda r: r["rps"]),
        "paired_vs_market": paired,
        "sign_convention": "mean_dRPS = model - market; negative means the model scored better.",
    }


def build_evaluation(store: Store, matches: pd.DataFrame, *, seed: int) -> dict[str, Any]:
    """The full ``evaluation.json`` for an artifact built from real data."""
    return {
        "schema_version": 1,
        "synthetic": False,
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "metric": "rps",
        "metric_note": "Ranked Probability Score, mean per match; lower is better.",
        "seed": seed,
        "excluded_models": {
            "bayes_poisson": "not in the served blend (fitted weight 0); MCMC refits take"
            " hours. See results/bayes_comparison.md."
        },
        "historical": historical_evaluation(
            matches, window=WINDOW, cadence_days=CADENCE_DAYS, min_train=2000, seed=seed
        ),
        "market": market_evaluation(store, matches, seed=seed),
    }


def export_artifact(
    data_root: Path,
    out_root: Path,
    *,
    version: str,
    seed: int,
    cutoff: dt.date | None = None,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    with_evaluation: bool = True,
) -> tuple[Manifest, Path]:
    """Train on the latest canonical matches snapshot and write ``out_root/<version>``."""
    bundle_dir = out_root / version
    if bundle_dir.exists():
        raise FileExistsError(f"{bundle_dir} already exists; artifact versions are immutable")
    store = Store(data_root / "snapshots")
    snapshot = store.latest(MATCHES_DATASET)
    matches = store.read(MATCHES_DATASET, "matches", snapshot)
    finished = matches[matches["status"] == "finished"]
    cutoff = cutoff or (finished["date"].max().date() + dt.timedelta(days=1))
    evaluation = build_evaluation(store, matches, seed=seed) if with_evaluation else None
    manifest = train_and_write(
        matches,
        bundle_dir,
        version=version,
        cutoff=cutoff,
        synthetic=False,
        horizon_days=horizon_days,
        evaluation=evaluation,
        training_data={
            "matches_snapshot": str(snapshot),
            "sources": "martj42/international_results (history) + football-data.org (WC 2026)",
        },
    )
    return manifest, bundle_dir
