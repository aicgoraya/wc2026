"""RESULTS.md generation: the live scoreboard of all models vs the market."""

from pathlib import Path

import pandas as pd


def write_results_md(scoreboard: pd.DataFrame, out: Path) -> None:
    """Render the scoreboard (RPS/log-loss/Brier + CIs, calibration summary) to markdown."""
    raise NotImplementedError("ships in Phase 1c")
