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
    remaining = meta.get("X-RequestsAvailable", "?")
    typer.echo(f"fixtures snapshot {snap_id} written (calls remaining: {remaining})")


@app.command()
def refresh() -> None:
    """Pull new results and odds, refit models, re-simulate, regenerate RESULTS.md."""
    _not_yet("Phase 1")


@app.command()
def predict() -> None:
    """Print model-vs-market probabilities for upcoming matches."""
    _not_yet("Phase 1c")


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
    """Regenerate RESULTS.md from the latest evaluation snapshot."""
    _not_yet("Phase 1c")
