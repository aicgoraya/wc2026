import datetime as dt
from typing import ClassVar

import pandas as pd

from wc2026.data.schema import Match, MatchStatus, matches_to_frame
from wc2026.eval.walkforward import RefitSchedule, walk_forward
from wc2026.models.base import Fixture, OutcomeProbs


def match(i: int, date: dt.date, home: str = "a", away: str = "b") -> Match:
    return Match(
        match_id=f"m{i}",
        date=date,
        home_id=home,
        away_id=away,
        home_goals=1,
        away_goals=0,
        neutral=True,
        tournament="friendly",
        status=MatchStatus.FINISHED,
    )


class SpyModel:
    """Records every fit's history extent and as_of; predicts uniform."""

    name = "spy"
    fits: ClassVar[list[tuple[pd.Timestamp | None, dt.date]]] = []

    def fit(self, history: pd.DataFrame, as_of: dt.date) -> None:
        max_date = history["date"].max() if len(history) else None
        SpyModel.fits.append((max_date, as_of))
        self._fitted_at = as_of

    def predict(self, fixture: Fixture) -> OutcomeProbs:
        self.last_predicted = fixture
        return OutcomeProbs(1 / 3, 1 / 3, 1 / 3)


class TestWalkForward:
    def test_every_fit_strictly_precedes_its_predictions(self) -> None:
        SpyModel.fits = []
        matches = matches_to_frame(
            [match(i, dt.date(2024, 1, 1) + dt.timedelta(days=7 * i)) for i in range(30)]
        )
        result = walk_forward(
            SpyModel,
            matches,
            (dt.date(2024, 3, 1), dt.date(2024, 8, 1)),
            RefitSchedule(every_days=30),
        )
        assert len(result) > 10
        # harness guarantee: each prediction's fit cutoff is <= the match date,
        # and the fit only saw matches strictly before the cutoff
        assert (result["fit_cutoff"] <= result["date"]).all()
        for max_history_date, as_of in SpyModel.fits:
            if max_history_date is not None:
                assert max_history_date < pd.Timestamp(as_of)
        assert len(SpyModel.fits) > 1  # actually refit across the window

    def test_scores_attached(self) -> None:
        matches = matches_to_frame([match(0, dt.date(2024, 6, 1))])
        result = walk_forward(SpyModel, matches, (dt.date(2024, 1, 1), dt.date(2024, 12, 31)))
        row = result.iloc[0]
        assert row["outcome"] == 0  # 1-0 home win
        assert row["rps"] == ((1 / 3 - 1) ** 2 + (2 / 3 - 1) ** 2) / 2
        assert {"log_loss", "brier", "p_home"} <= set(result.columns)

    def test_unplayed_and_out_of_window_excluded(self) -> None:
        played = match(0, dt.date(2024, 6, 1))
        early = match(1, dt.date(2023, 1, 1))
        scheduled = Match(
            match_id="sched",
            date=dt.date(2024, 6, 20),
            home_id="a",
            away_id="b",
            neutral=True,
            tournament="friendly",
            status=MatchStatus.SCHEDULED,
        )
        frame = matches_to_frame([played, early, scheduled])
        result = walk_forward(SpyModel, frame, (dt.date(2024, 1, 1), dt.date(2024, 12, 31)))
        assert list(result["match_id"]) == ["m0"]
