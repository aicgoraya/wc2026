import datetime as dt

import numpy as np
import pytest

from wc2026.data.schema import Match, MatchStatus, matches_to_frame
from wc2026.models.base import Fixture
from wc2026.models.elo import (
    EloForecaster,
    OrdinalParams,
    fit_ordinal_logistic,
    predict_ordinal,
)


class TestOrdinalLogistic:
    PARAMS = OrdinalParams(beta=0.006, theta_lo=-0.5, theta_hi=0.4)

    def test_probs_sum_to_one_and_are_monotonic(self) -> None:
        x = np.linspace(-400, 400, 31)
        probs = predict_ordinal(self.PARAMS, x)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-12)
        assert (np.diff(probs[:, 0]) > 0).all()  # home prob grows with rating edge
        assert (np.diff(probs[:, 2]) < 0).all()
        assert probs[15, 1] == max(probs[:, 1])  # draw peaks near level ratings

    def test_parameter_recovery(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.normal(0, 150, size=20_000)
        true_probs = predict_ordinal(self.PARAMS, x)
        y = np.array([rng.choice(3, p=p) for p in true_probs])
        fitted = fit_ordinal_logistic(x, y)
        assert fitted.beta == pytest.approx(self.PARAMS.beta, rel=0.10)
        assert fitted.theta_lo == pytest.approx(self.PARAMS.theta_lo, abs=0.06)
        assert fitted.theta_hi == pytest.approx(self.PARAMS.theta_hi, abs=0.06)

    def test_too_few_samples_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 100"):
            fit_ordinal_logistic(np.zeros(50), np.zeros(50, dtype=np.int64))


def synthetic_history(n_rounds: int = 40, seed: int = 5) -> list[Match]:
    """Stochastic round-robin: stronger teams win more often, draws happen.

    Outcomes are sampled (not deterministic) so the ordinal link fit is never
    perfectly separable — separable data would push beta to infinity.
    """
    rng = np.random.default_rng(seed)
    teams = {"strong": 2, "mid": 1, "mid2": 1, "weak": 0}
    goals = {0: (2, 0), 1: (1, 1), 2: (0, 1)}  # outcome code -> scoreline
    matches = []
    day = dt.date(2018, 1, 1)
    i = 0
    for _ in range(n_rounds):
        for home, hs in teams.items():
            for away, as_ in teams.items():
                if home == away:
                    continue
                i += 1
                day = day + dt.timedelta(days=2)
                edge = hs - as_  # in [-2, 2]
                p_home = 0.40 + 0.15 * edge
                p_away = 0.35 - 0.15 * edge
                outcome = int(rng.choice(3, p=[p_home, 1 - p_home - p_away, p_away]))
                home_goals, away_goals = goals[outcome]
                matches.append(
                    Match(
                        match_id=f"s{i}",
                        date=day,
                        home_id=home,
                        away_id=away,
                        home_goals=home_goals,
                        away_goals=away_goals,
                        neutral=True,
                        tournament="friendly",
                        status=MatchStatus.FINISHED,
                    )
                )
    return matches


class TestEloForecaster:
    def test_fit_predict_orders_teams(self) -> None:
        model = EloForecaster()
        model.fit(matches_to_frame(synthetic_history()), as_of=dt.date(2026, 1, 1))
        probs = model.predict(Fixture("strong", "weak", dt.date(2026, 6, 1)))
        assert probs.home > 0.5 > probs.away
        reverse = model.predict(Fixture("weak", "strong", dt.date(2026, 6, 1)))
        assert reverse.away > 0.5 > reverse.home

    def test_unseen_team_gets_initial_rating(self) -> None:
        model = EloForecaster()
        model.fit(matches_to_frame(synthetic_history()), as_of=dt.date(2026, 1, 1))
        probs = model.predict(Fixture("strong", "atlantis", dt.date(2026, 6, 1)))
        assert probs.home > probs.away  # strong is rated above 1500

    def test_no_leak_future_matches_ignored(self) -> None:
        history = synthetic_history()
        future = Match(
            match_id="future",
            date=dt.date(2026, 5, 1),
            home_id="weak",
            away_id="strong",
            home_goals=9,
            away_goals=0,
            neutral=True,
            tournament="friendly",
            status=MatchStatus.FINISHED,
        )
        as_of = dt.date(2026, 1, 1)
        clean = EloForecaster()
        clean.fit(matches_to_frame(history), as_of=as_of)
        poisoned = EloForecaster()
        poisoned.fit(matches_to_frame([*history, future]), as_of=as_of)
        fixture = Fixture("strong", "weak", dt.date(2026, 6, 1))
        assert clean.predict(fixture) == poisoned.predict(fixture)

    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="fit"):
            EloForecaster().predict(Fixture("a", "b", dt.date(2026, 1, 1)))

    def test_home_advantage_shifts_probs(self) -> None:
        model = EloForecaster()
        model.fit(matches_to_frame(synthetic_history()), as_of=dt.date(2026, 1, 1))
        neutral = model.predict(Fixture("mid", "mid2", dt.date(2026, 6, 1), neutral=True))
        at_home = model.predict(Fixture("mid", "mid2", dt.date(2026, 6, 1), neutral=False))
        assert at_home.home > neutral.home
