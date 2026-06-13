"""The daily refresh: pull -> ingest -> regenerate -> dashboard. Idempotent.

One command brings everything up to date and is safe to run repeatedly:
collectors append new versioned snapshots, the canonical matches dataset is
rebuilt, RESULTS.md is regenerated, and the dashboard payload is written. Each
step is wrapped so a missing API key (collectors) degrades gracefully rather
than aborting the run — the rest still refreshes from the latest stored data.

Deliberately does NOT run the expensive Bayesian MCMC comparison (``model-
compare`` / ``bayes-compare`` are separate, occasional commands); routine
refresh stays fast.
"""

import dataclasses
import datetime as dt
import json
from pathlib import Path

from wc2026.config import Settings
from wc2026.pipeline.collect import (
    MissingCredentialError,
    collect_fixtures,
    collect_history,
    collect_odds,
)
from wc2026.pipeline.evaluate import run_report


@dataclasses.dataclass
class RefreshReport:
    """What each step of a refresh did (for logging / the CLI)."""

    steps: list[str] = dataclasses.field(default_factory=list)

    def ok(self, msg: str) -> None:
        """Record a completed step."""
        self.steps.append(f"ok    {msg}")

    def skip(self, msg: str) -> None:
        """Record a skipped step (e.g. a missing credential)."""
        self.steps.append(f"skip  {msg}")


def refresh(
    settings: Settings,
    *,
    out_md: Path = Path("RESULTS.md"),
    plots_dir: Path = Path("results"),
    dashboard_json: Path = Path("results/dashboard.json"),
    n_sims: int = 20_000,
    today: dt.date | None = None,
) -> RefreshReport:
    """Pull new data, rebuild the canonical dataset, regenerate RESULTS + dashboard."""
    report = RefreshReport()

    try:
        snap, _meta = collect_fixtures(settings)
        report.ok(f"fixtures snapshot {snap}")
    except MissingCredentialError:
        report.skip("fixtures (FOOTBALL_DATA_TOKEN not set)")

    try:
        snap, _meta = collect_odds(settings)
        report.ok(f"odds snapshot {snap}")
    except MissingCredentialError:
        report.skip("odds (ODDS_API_KEY not set)")

    try:
        snap, counts = collect_history(settings)
        report.ok(f"canonical matches snapshot {snap} ({counts['merged']} rows)")
    except FileNotFoundError:
        report.skip("ingest-history (no WC fixtures snapshot yet)")
        return report  # nothing downstream can run without the matches dataset

    run_report(settings.data_root, out_md=out_md, plots_dir=plots_dir, seed=settings.default_seed)
    report.ok(f"regenerated {out_md}")

    # build the dashboard payload from the freshly-updated snapshots
    from wc2026.dashboard.data import build_payload

    payload = build_payload(
        settings.data_root, seed=settings.default_seed, n_sims=n_sims, today=today
    )
    dashboard_json.parent.mkdir(parents=True, exist_ok=True)
    dashboard_json.write_text(json.dumps(payload, indent=2))
    n_live = payload["live_vs_market"]["n"]
    report.ok(f"wrote {dashboard_json} (win-cup, upcoming, live-vs-market n={n_live})")
    return report
