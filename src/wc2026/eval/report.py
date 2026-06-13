"""RESULTS.md rendering: the scoreboard of all models vs the market.

Two strictly separated tracks:

- PRIMARY: long historical walk-forward — real sample sizes, no market
  benchmark possible (historical odds are paywalled on the free tier).
- LIVE: the WC 2026 scoreboard accumulating match by match, market included
  via the closing-line proxy — small-sample by construction until the
  tournament progresses, and labeled as such.
"""

import datetime as dt
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from wc2026.eval import calibration, scoring


def scoreboard_row(name: str, rows: pd.DataFrame, *, seed: int) -> dict[str, object]:
    """Summary metrics for one model's per-match walk-forward rows."""
    probs = rows[["p_home", "p_draw", "p_away"]].to_numpy(dtype=np.float64)
    outcomes = rows["outcome"].to_numpy(dtype=np.int64)
    rps_per_match = rows["rps"].to_numpy(dtype=np.float64)
    lo, hi = scoring.bootstrap_ci(rps_per_match, seed=seed)
    return {
        "model": name,
        "n": len(rows),
        "rps": float(rps_per_match.mean()),
        "rps_ci_lo": lo,
        "rps_ci_hi": hi,
        "log_loss": float(rows["log_loss"].mean()),
        "brier": float(rows["brier"].mean()),
        "ece": calibration.ece(probs, outcomes),
    }


def md_table(rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> str:
    """A minimal GitHub-markdown table (no extra dependency)."""

    def fmt(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines.extend("| " + " | ".join(fmt(row.get(c, "—")) for c in columns) + " |" for row in rows)
    return "\n".join(lines)


def render_results_md(
    *,
    generated_utc: dt.datetime,
    primary_rows: Sequence[Mapping[str, object]],
    primary_window: tuple[dt.date, dt.date],
    tournament_rows: Sequence[Mapping[str, object]],
    live_rows: Sequence[Mapping[str, object]],
    live_notes: Sequence[str],
    plot_paths: Mapping[str, str],
    paired_rows: Sequence[Mapping[str, object]] = (),
    methodology_notes: Sequence[str] = (),
) -> str:
    """Assemble RESULTS.md from precomputed scoreboard rows."""
    cols = ["model", "n", "rps", "rps_ci_lo", "rps_ci_hi", "log_loss", "brier", "ece"]
    paired_cols = ["comparison", "n", "mean_dRPS", "ci_lo", "ci_hi", "DM", "p", "verdict"]
    start, end = primary_window
    parts = [
        "# RESULTS — live scoreboard",
        "",
        f"_Generated {generated_utc:%Y-%m-%d %H:%M} UTC. Lower is better on every metric;"
        " RPS is primary. `ece` = expected calibration error. CIs are 95% bootstrap._",
        "",
        "## Track A — PRIMARY: historical walk-forward (real sample size)",
        "",
        f"All internationals, {start} → {end}; every prediction fit strictly on",
        "pre-cutoff matches (ratings AND the ordinal link). No market column here:",
        "historical closing odds are paywalled on the free tier, so the market is",
        "only benchmarked on the live track below.",
        "",
        md_table(primary_rows, cols),
        "",
        *(f"- {note}" for note in methodology_notes),
        "",
        "### Paired significance (per-match ΔRPS on shared matches)",
        "",
        "The marginal CIs above share the same matches, so they overlap even when one",
        "model is reliably better game-by-game. The honest test is the paired per-match",
        "RPS difference: `mean_dRPS = challenger − baseline` (negative ⇒ challenger",
        "better), with a 95% bootstrap CI over matches and a Diebold–Mariano statistic.",
        "",
        md_table(paired_rows, paired_cols) if paired_rows else "_Only one model on the board._",
        "",
        "### Tournament-only slice (World Cup + continental finals)",
        "",
        md_table(tournament_rows, cols),
        "",
    ]
    for label, path in plot_paths.items():
        parts.extend([f"![{label}]({path})", ""])
    parts.extend(
        [
            "## Track B — LIVE: World Cup 2026 (SMALL SAMPLE — no conclusions yet)",
            "",
            "Accumulates match by match as the tournament runs. The market column is",
            "the de-vigged closing-line proxy (sharpest book; see README). With the",
            "current sample size this table is reported for transparency only —",
            "**do not read anything into it yet.**",
            "",
            md_table(live_rows, cols) if live_rows else "_No scoreable matches yet._",
            "",
        ]
    )
    parts.extend(f"- {note}" for note in live_notes)
    parts.append("")
    return "\n".join(parts)
