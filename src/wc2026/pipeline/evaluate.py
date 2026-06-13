"""Orchestration of the full evaluation: walk-forward runs → RESULTS.md.

This is the module ``wc2026 report`` runs. It reads the latest canonical
snapshots, evaluates every registered model walk-forward, scores the market
on the live track via the closing-line proxy, and regenerates RESULTS.md
plus the reliability plots.
"""

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from wc2026.data.sources.elo_own import CONTINENTAL_FINALS
from wc2026.data.store import MATCHES_DATASET, Store
from wc2026.eval import calibration, scoring
from wc2026.eval.join import join_events_to_fixtures, load_all_quotes, unique_events
from wc2026.eval.market import BenchmarkPolicy, benchmark_probs, closing_quotes
from wc2026.eval.report import render_results_md, scoreboard_row
from wc2026.eval.walkforward import RefitSchedule, walk_forward
from wc2026.models.elo import EloForecaster
from wc2026.pipeline.collect import WC_FIXTURES_DATASET

PRIMARY_START = dt.date(2010, 1, 1)
WC2026_START = dt.date(2026, 6, 11)
TOURNAMENT_SLICE = frozenset({"fifa_world_cup"}) | CONTINENTAL_FINALS


def market_live_rows(store: Store, completed: pd.DataFrame) -> pd.DataFrame:
    """Market closing-line probabilities scored on completed WC matches.

    Only matches whose first stored quote precedes kickoff get a market row —
    collection started 2026-06-12, so earlier matches have no closing proxy.
    """
    policy = BenchmarkPolicy()
    quotes = load_all_quotes(store)
    closing = closing_quotes(quotes, policy)
    bench = benchmark_probs(closing, policy)
    fixtures = store.read(WC_FIXTURES_DATASET, "matches")
    joined = join_events_to_fixtures(unique_events(quotes), fixtures)
    bench = bench.merge(joined, on="event_id", how="inner")

    flipped = bench["flipped"].to_numpy(dtype=bool)
    p_home = np.where(flipped, bench["p_away"], bench["p_home"])
    p_away = np.where(flipped, bench["p_home"], bench["p_away"])
    bench = bench.assign(p_home=p_home, p_away=p_away)

    rows = bench.merge(
        completed[["match_id", "date", "tournament", "home_goals", "away_goals"]],
        on="match_id",
        how="inner",
    )
    if rows.empty:
        return rows
    rows["outcome"] = scoring.outcome_codes(
        rows["home_goals"].to_numpy(), rows["away_goals"].to_numpy()
    )
    probs = rows[["p_home", "p_draw", "p_away"]].to_numpy(dtype=np.float64)
    outcomes = rows["outcome"].to_numpy(dtype=np.int64)
    rows["rps"] = scoring.rps(probs, outcomes)
    rows["log_loss"] = scoring.log_loss(probs, outcomes)
    rows["brier"] = scoring.brier(probs, outcomes)
    return rows


def run_report(
    data_root: Path,
    *,
    out_md: Path,
    plots_dir: Path,
    seed: int,
    today: dt.date | None = None,
) -> str:
    """Run the full evaluation and write RESULTS.md; returns the markdown."""
    today = today or dt.datetime.now(dt.UTC).date()
    store = Store(data_root / "snapshots")
    matches = store.read(MATCHES_DATASET, "matches")

    primary_end = WC2026_START - dt.timedelta(days=1)
    elo_primary = walk_forward(
        EloForecaster,
        matches,
        (PRIMARY_START, primary_end),
        RefitSchedule(every_days=30),
    )
    in_slice = elo_primary["tournament"].isin(TOURNAMENT_SLICE)
    primary_rows = [scoreboard_row("elo_baseline", elo_primary, seed=seed)]
    tournament_rows = [scoreboard_row("elo_baseline", elo_primary[in_slice], seed=seed)]

    plots_dir.mkdir(parents=True, exist_ok=True)
    probs = elo_primary[["p_home", "p_draw", "p_away"]].to_numpy(dtype=np.float64)
    outcomes = elo_primary["outcome"].to_numpy(dtype=np.int64)
    fig = calibration.reliability_plot(
        probs, outcomes, title=f"Elo baseline reliability ({PRIMARY_START} to {primary_end})"
    )
    plot_path = plots_dir / "reliability_elo_primary.png"
    fig.savefig(plot_path)

    elo_live = walk_forward(
        EloForecaster, matches, (WC2026_START, today), RefitSchedule(every_days=1)
    )
    completed = matches[
        (matches["status"] == "finished") & (matches["date"] >= pd.Timestamp(WC2026_START))
    ]
    market_rows = market_live_rows(store, completed)

    live_rows = []
    live_notes = [
        f"Live Elo rows cover all {len(elo_live)} completed WC matches;"
        " market rows exist only where a pre-kickoff quote was stored"
        f" (collection began 2026-06-12): {len(market_rows)} matches so far.",
        "The Elo-vs-market RPS gap is quantified on their COMMON matches as they"
        " accumulate; the baseline is expected to lose to the market — that gap is"
        " the target for Dixon-Coles and the Bayesian model.",
    ]
    if len(elo_live):
        live_rows.append(scoreboard_row("elo_baseline (all completed)", elo_live, seed=seed))
    if len(market_rows):
        live_rows.append(scoreboard_row("market (matches w/ lines)", market_rows, seed=seed))
        common = elo_live[elo_live["match_id"].isin(market_rows["match_id"])]
        if len(common):
            live_rows.append(scoreboard_row("elo_baseline (common subset)", common, seed=seed))

    markdown = render_results_md(
        generated_utc=dt.datetime.now(dt.UTC),
        primary_rows=primary_rows,
        primary_window=(PRIMARY_START, primary_end),
        tournament_rows=tournament_rows,
        live_rows=live_rows,
        live_notes=live_notes,
        plot_paths={"Elo reliability (primary)": str(plot_path.relative_to(out_md.parent))},
    )
    out_md.write_text(markdown)
    return markdown
