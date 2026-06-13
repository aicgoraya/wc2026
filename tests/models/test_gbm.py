import datetime as dt

import numpy as np
import pytest

from wc2026.data.schema import Match, MatchStatus, matches_to_frame
from wc2026.models.base import Fixture
from wc2026.models.gbm import GbmForecaster

pytest.importorskip("lightgbm")


def synthetic_matches(n_rounds: int = 80, seed: int = 0) -> list[Match]:
    """Round-robin where stronger teams (higher strength) win more often."""
    rng = np.random.default_rng(seed)
    strength = {"alpha": 1.0, "bravo": 0.5, "charlie": 0.0, "delta": -0.5, "echo": -1.0}
    teams = list(strength)
    matches = []
    day = dt.date(2015, 1, 1)
    i = 0
    for _ in range(n_rounds):
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                i += 1
                day += dt.timedelta(days=1)
                edge = strength[home] - strength[away]
                # 3-way softmax so draws genuinely exist AND strength matters
                logits = np.array([0.3 + 1.0 * edge, 0.0, 0.3 - 1.0 * edge])
                p = np.exp(logits) / np.exp(logits).sum()
                outcome = int(rng.choice(3, p=p))
                goals = {0: (2, 0), 1: (1, 1), 2: (0, 2)}[outcome]
                matches.append(
                    Match(
                        match_id=f"s{i}",
                        date=day,
                        home_id=home,
                        away_id=away,
                        home_goals=goals[0],
                        away_goals=goals[1],
                        neutral=True,
                        tournament="friendly",
                        status=MatchStatus.FINISHED,
                    )
                )
    return matches


@pytest.fixture(scope="module")
def fitted() -> tuple[GbmForecaster, list[Match]]:
    matches = synthetic_matches()
    frame = matches_to_frame(matches)
    model = GbmForecaster(frame)
    model.fit(frame, as_of=dt.date(2026, 1, 1))
    return model, matches


class TestGbm:
    def test_too_few_rows_raises(self) -> None:
        frame = matches_to_frame(synthetic_matches(n_rounds=2))
        with pytest.raises(ValueError, match="at least 500"):
            GbmForecaster(frame).fit(frame, as_of=dt.date(2026, 1, 1))

    def test_predict_before_fit_raises(self) -> None:
        frame = matches_to_frame(synthetic_matches(n_rounds=3))
        with pytest.raises(RuntimeError, match="fit"):
            GbmForecaster(frame).predict(Fixture("alpha", "echo", dt.date(2026, 1, 1)))

    def test_strong_home_team_favored(self, fitted: tuple[GbmForecaster, list[Match]]) -> None:
        model, matches = fitted
        # a LATE matchup, after ratings/form have built up from prior games
        target = [m for m in matches if m.home_id == "alpha" and m.away_id == "echo"][-1]
        probs = model.predict(Fixture("alpha", "echo", target.date, neutral=True))
        assert probs.home > probs.away  # strong team favored
        rev = [m for m in matches if m.home_id == "echo" and m.away_id == "alpha"][-1]
        probs_rev = model.predict(Fixture("echo", "alpha", rev.date, neutral=True))
        assert probs_rev.away > probs_rev.home

    def test_probs_valid(self, fitted: tuple[GbmForecaster, list[Match]]) -> None:
        model, matches = fitted
        target = matches[-1]
        probs = model.predict(Fixture(target.home_id, target.away_id, target.date, neutral=True))
        assert sum(probs) == pytest.approx(1.0, abs=1e-6)

    def test_missing_features_raises(self, fitted: tuple[GbmForecaster, list[Match]]) -> None:
        model, _ = fitted
        with pytest.raises(KeyError, match="no precomputed features"):
            model.predict(Fixture("alpha", "echo", dt.date(1999, 1, 1)))

    def test_feature_importances(self, fitted: tuple[GbmForecaster, list[Match]]) -> None:
        model, _ = fitted
        imp = model.feature_importances()
        assert "elo_diff" in imp
        assert sum(imp.values()) > 0
