"""The ``wc2026`` command-line interface."""

import typer

app = typer.Typer(no_args_is_help=True, help="WC2026 probabilistic forecasting engine.")


def _not_yet(phase: str) -> None:
    typer.secho(f"Not implemented yet — ships in {phase}.", fg=typer.colors.YELLOW, err=True)
    raise typer.Exit(code=1)


@app.command()
def snapshot_odds() -> None:
    """Pull current 1X2 odds for all WC fixtures and store a snapshot."""
    from wc2026.config import get_settings
    from wc2026.pipeline.collect import MissingCredentialError, collect_odds

    try:
        snap_id, meta = collect_odds(get_settings())
    except MissingCredentialError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    remaining = meta.get("x-requests-remaining", "?")
    typer.echo(f"odds snapshot {snap_id} written (quota remaining: {remaining})")


@app.command()
def snapshot_fixtures() -> None:
    """Pull WC fixtures + live results from football-data.org and store a snapshot."""
    from wc2026.config import get_settings
    from wc2026.pipeline.collect import MissingCredentialError, collect_fixtures

    try:
        snap_id, meta = collect_fixtures(get_settings())
    except MissingCredentialError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    remaining = meta.get("X-Requests-Available-Minute", "?")
    typer.echo(f"fixtures snapshot {snap_id} written (calls remaining this minute: {remaining})")


@app.command()
def ingest_history() -> None:
    """Build the canonical matches dataset: full history merged with the live WC feed."""
    from wc2026.config import get_settings
    from wc2026.pipeline.collect import collect_history

    snap_id, counts = collect_history(get_settings())
    typer.echo(
        f"matches snapshot {snap_id} written "
        f"(history {counts['history']} + wc_feed {counts['wc_feed']} -> {counts['merged']})"
    )


@app.command()
def refresh() -> None:
    """Pull new results and odds, refit models, re-simulate, regenerate RESULTS.md."""
    _not_yet("Phase 1")


@app.command()
def predict(days: int = 7) -> None:
    """Print model-vs-market probabilities for upcoming WC matches."""
    import datetime as dt

    import pandas as pd

    from wc2026.config import get_settings
    from wc2026.data.store import MATCHES_DATASET, Store
    from wc2026.eval.join import join_events_to_fixtures, load_all_quotes, unique_events
    from wc2026.eval.market import BenchmarkPolicy, benchmark_probs, closing_quotes
    from wc2026.models.base import Fixture
    from wc2026.pipeline.collect import WC_FIXTURES_DATASET
    from wc2026.pipeline.evaluate import model_lineup

    settings = get_settings()
    store = Store(settings.data_root / "snapshots")
    today = dt.datetime.now(dt.UTC).date()

    models = model_lineup()
    canonical = store.read(MATCHES_DATASET, "matches")
    for model in models:
        model.fit(canonical, as_of=today + dt.timedelta(days=1))

    fixtures = store.read(WC_FIXTURES_DATASET, "matches")
    upcoming = fixtures[
        (fixtures["status"] == "scheduled")
        & (fixtures["date"] >= pd.Timestamp(today))
        & (fixtures["date"] <= pd.Timestamp(today + dt.timedelta(days=days)))
    ].sort_values("date")

    policy = BenchmarkPolicy()
    quotes = load_all_quotes(store)
    bench = benchmark_probs(closing_quotes(quotes, policy), policy)
    joined = join_events_to_fixtures(unique_events(quotes), fixtures)
    bench = bench.merge(joined, on="event_id").set_index("match_id")

    headers = "  ".join(f"{m.name + ' (H/D/A)':18s}" for m in models)
    typer.echo(f"{'date':10s}  {'fixture':34s}  {headers}  {'market (H/D/A)':18s}")
    rows = zip(
        upcoming["match_id"].astype(str),
        upcoming["date"],
        upcoming["home_id"].astype(str),
        upcoming["away_id"].astype(str),
        upcoming["neutral"].astype(bool),
        strict=True,
    )
    for match_id, date, home_id, away_id, neutral in rows:
        fixture = Fixture(home_id=home_id, away_id=away_id, date=date.date(), neutral=neutral)
        cells = []
        for model in models:
            p = model.predict(fixture)
            cells.append(f"{p.home:.2f}/{p.draw:.2f}/{p.away:.2f}")
        if match_id in bench.index:
            b = bench.loc[match_id]
            ph, pa = (b["p_away"], b["p_home"]) if b["flipped"] else (b["p_home"], b["p_away"])
            market_s = f"{ph:.2f}/{b['p_draw']:.2f}/{pa:.2f}"
        else:
            market_s = "no line stored"
        name = f"{home_id} v {away_id}"
        model_cols = "  ".join(f"{c:18s}" for c in cells)
        typer.echo(f"{date.date()!s:10s}  {name:34s}  {model_cols}  {market_s:18s}")


@app.command()
def tune_dc() -> None:
    """Reproduce the leak-free Dixon-Coles half-life selection (inner window)."""
    from wc2026.config import get_settings
    from wc2026.data.store import MATCHES_DATASET, Store
    from wc2026.pipeline.tune import VALIDATION_WINDOW, select_half_life

    settings = get_settings()
    matches = Store(settings.data_root / "snapshots").read(MATCHES_DATASET, "matches")
    best, table = select_half_life(matches)
    typer.echo(f"validation window: {VALIDATION_WINDOW[0]} -> {VALIDATION_WINDOW[1]}")
    typer.echo(table.to_string(index=False))
    typer.echo(f"selected half-life: {best:.0f} days")


@app.command()
def simulate() -> None:
    """Run the Monte-Carlo tournament simulation and print advancement probabilities."""
    _not_yet("Phase 3")


@app.command()
def backtest() -> None:
    """Run the simulated paper-trading backtest (no real wagering)."""
    _not_yet("Phase 2")


@app.command()
def report() -> None:
    """Run the full walk-forward evaluation and regenerate RESULTS.md (minutes)."""
    from pathlib import Path

    from wc2026.config import get_settings
    from wc2026.pipeline.evaluate import run_report

    settings = get_settings()
    run_report(
        settings.data_root,
        out_md=Path("RESULTS.md"),
        plots_dir=Path("results"),
        seed=settings.default_seed,
    )
    typer.echo("RESULTS.md and results/ plots regenerated")
