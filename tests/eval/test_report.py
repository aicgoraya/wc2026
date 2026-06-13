import datetime as dt

import numpy as np
import pandas as pd

from wc2026.eval.report import md_table, render_results_md, scoreboard_row


def fake_rows(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(np.ones(3), size=n)
    outcomes = np.array([rng.choice(3, p=p) for p in probs])
    frame = pd.DataFrame(probs, columns=["p_home", "p_draw", "p_away"])
    frame["outcome"] = outcomes
    from wc2026.eval import scoring

    frame["rps"] = scoring.rps(probs, outcomes)
    frame["log_loss"] = scoring.log_loss(probs, outcomes)
    frame["brier"] = scoring.brier(probs, outcomes)
    return frame


class TestScoreboardRow:
    def test_fields_and_ci_brackets_mean(self) -> None:
        row = scoreboard_row("m", fake_rows(400), seed=1)
        assert row["model"] == "m" and row["n"] == 400
        assert row["rps_ci_lo"] <= row["rps"] <= row["rps_ci_hi"]  # type: ignore[operator]

    def test_deterministic(self) -> None:
        rows = fake_rows(100)
        assert scoreboard_row("m", rows, seed=5) == scoreboard_row("m", rows, seed=5)


def test_md_table_renders_missing_as_dash() -> None:
    table = md_table([{"a": 1.23456, "b": "x"}], ["a", "b", "c"])
    assert "| 1.2346 | x | — |" in table
    assert table.splitlines()[0] == "| a | b | c |"


def test_render_results_md_sections() -> None:
    row = scoreboard_row("elo_baseline", fake_rows(50), seed=2)
    md = render_results_md(
        generated_utc=dt.datetime(2026, 6, 13, tzinfo=dt.UTC),
        primary_rows=[row],
        primary_window=(dt.date(2010, 1, 1), dt.date(2026, 6, 10)),
        tournament_rows=[row],
        live_rows=[],
        live_notes=["note one"],
        plot_paths={"plot": "results/x.png"},
    )
    assert "## Track A — PRIMARY" in md
    assert "## Track B — LIVE" in md
    assert "SMALL SAMPLE" in md
    assert "_No scoreable matches yet._" in md
    assert "![plot](results/x.png)" in md
    assert "- note one" in md
    assert "elo_baseline" in md
