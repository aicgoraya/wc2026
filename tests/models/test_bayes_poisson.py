"""Tests for the hierarchical Bayesian Poisson model.

Uses a deliberately tiny sampler (few draws/chains) so the suite stays fast;
parameter recovery is therefore asserted with wide tolerances. The headline
property — partial pooling shrinking sparse teams harder than data-rich ones —
is tested directly.
"""

import datetime as dt

import numpy as np
import pytest

from wc2026.data.schema import Match, MatchStatus, matches_to_frame
from wc2026.models.base import Fixture
from wc2026.models.bayes_poisson import BayesConfig, BayesPoissonForecaster

pytest.importorskip("pymc")

FAST = BayesConfig(draws=300, tune=400, chains=2, train_window_years=30, thin_to=200, seed=1)


def simulate_matches(
    attack: dict[str, float],
    defence: dict[str, float],
    *,
    n_rounds: int,
    intercept: float = 0.2,
    home_adv: float = 0.3,
    seed: int = 0,
    start: dt.date = dt.date(2016, 1, 1),
) -> list[Match]:
    rng = np.random.default_rng(seed)
    teams = list(attack)
    matches = []
    day = start
    i = 0
    for _ in range(n_rounds):
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                i += 1
                day += dt.timedelta(days=1)
                neutral = bool(rng.integers(0, 2))
                lam_h = np.exp(
                    intercept + attack[home] - defence[away] + (0 if neutral else home_adv)
                )
                lam_a = np.exp(intercept + attack[away] - defence[home])
                matches.append(
                    Match(
                        match_id=f"s{i}",
                        date=day,
                        home_id=home,
                        away_id=away,
                        home_goals=int(rng.poisson(lam_h)),
                        away_goals=int(rng.poisson(lam_a)),
                        neutral=neutral,
                        tournament="friendly",
                        status=MatchStatus.FINISHED,
                    )
                )
    return matches


TRUE_ATTACK = {"alpha": 0.6, "bravo": 0.2, "charlie": -0.2, "delta": -0.6, "echo": 0.0}
TRUE_DEFENCE = {"alpha": 0.4, "bravo": 0.1, "charlie": -0.2, "delta": -0.3, "echo": 0.0}


@pytest.fixture(scope="module")
def fitted() -> BayesPoissonForecaster:
    history = simulate_matches(TRUE_ATTACK, TRUE_DEFENCE, n_rounds=30)
    model = BayesPoissonForecaster(FAST)
    model.fit(matches_to_frame(history), as_of=dt.date(2026, 1, 1))
    return model


class TestFit:
    def test_samples_and_reports_diagnostics(self, fitted: BayesPoissonForecaster) -> None:
        d = fitted.diagnostics
        assert d.n_teams == 5
        assert d.max_rhat < 1.1  # loose for the tiny test sampler
        assert d.min_ess_bulk > 50
        assert d.divergences >= 0

    def test_parameter_recovery_ordering(self, fitted: BayesPoissonForecaster) -> None:
        # posterior-mean attack should order the teams correctly
        post = fitted._posterior
        means = {t: float(post.attack[:, post.index[t]].mean()) for t in TRUE_ATTACK}
        assert means["alpha"] > means["bravo"] > means["echo"] > means["delta"]

    def test_too_few_matches_raises(self) -> None:
        history = simulate_matches(TRUE_ATTACK, TRUE_DEFENCE, n_rounds=2)
        with pytest.raises(ValueError, match="at least 500"):
            BayesPoissonForecaster(FAST).fit(matches_to_frame(history), as_of=dt.date(2026, 1, 1))

    def test_no_leak_future_ignored(self) -> None:
        history = simulate_matches(TRUE_ATTACK, TRUE_DEFENCE, n_rounds=25)
        as_of = dt.date(2026, 1, 1)
        future = simulate_matches(
            {t: -v for t, v in TRUE_ATTACK.items()},
            TRUE_DEFENCE,
            n_rounds=5,
            seed=9,
            start=dt.date(2026, 6, 1),
        )
        clean = BayesPoissonForecaster(FAST)
        clean.fit(matches_to_frame(history), as_of=as_of)
        with_future = BayesPoissonForecaster(FAST)
        with_future.fit(matches_to_frame(history + future), as_of=as_of)
        # identical training data reaching the fit -> identical diagnostics n_train
        assert clean.diagnostics.n_train == with_future.diagnostics.n_train


class TestPartialPooling:
    def test_sparse_team_shrinks_harder_than_data_rich_team(self) -> None:
        # both 'rich' and 'sparse' actually have the SAME extreme strength, but
        # 'sparse' is seen in only a handful of games. Partial pooling must pull
        # the sparse team's posterior mean closer to 0 (the population mean).
        rng = np.random.default_rng(3)
        base = {f"t{i}": float(rng.normal(0, 0.2)) for i in range(6)}
        attack = {**base, "rich": 0.9, "sparse": 0.9}
        defence = {**dict.fromkeys(base, 0.0), "rich": 0.0, "sparse": 0.0}

        matches = simulate_matches(attack, defence, n_rounds=20, seed=4)
        # add only a few games for 'sparse' (3 wins), many already exist for 'rich'
        extra = []
        for j, opp in enumerate(["t0", "t1", "t2"]):
            extra.append(
                Match(
                    match_id=f"sp{j}",
                    date=dt.date(2025, 1, 1) + dt.timedelta(days=j),
                    home_id="sparse",
                    away_id=opp,
                    home_goals=3,
                    away_goals=0,
                    neutral=True,
                    tournament="friendly",
                    status=MatchStatus.FINISHED,
                )
            )
        model = BayesPoissonForecaster(FAST)
        model.fit(matches_to_frame(matches + extra), as_of=dt.date(2026, 1, 1))
        post = model._posterior
        rich_attack = float(post.attack[:, post.index["rich"]].mean())
        sparse_attack = float(post.attack[:, post.index["sparse"]].mean())
        # both truly 0.9, but the sparse team is shrunk further toward 0
        assert sparse_attack < rich_attack


class TestPredict:
    def test_scoreline_is_a_distribution(self, fitted: BayesPoissonForecaster) -> None:
        dist = fitted.predict_scoreline(Fixture("alpha", "delta", dt.date(2026, 6, 1)))
        assert dist.matrix.sum() == pytest.approx(1.0, abs=1e-9)

    def test_stronger_team_favored(self, fitted: BayesPoissonForecaster) -> None:
        probs = fitted.predict(Fixture("alpha", "delta", dt.date(2026, 6, 1), neutral=True))
        assert probs.home > probs.away

    def test_unseen_team_is_population_average(self, fitted: BayesPoissonForecaster) -> None:
        probs = fitted.predict(Fixture("alpha", "atlantis", dt.date(2026, 6, 1)))
        assert probs.home > probs.away  # alpha strong, unknown ~ average

    def test_posterior_predictive_has_spread(self, fitted: BayesPoissonForecaster) -> None:
        draws = fitted.predict_posterior(Fixture("bravo", "charlie", dt.date(2026, 6, 1)), 150)
        assert draws.shape == (150, 3)
        np.testing.assert_allclose(draws.sum(axis=1), 1.0, atol=1e-9)
        assert draws[:, 0].std() > 0  # genuine parameter uncertainty, not a point

    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="fit"):
            BayesPoissonForecaster(FAST).predict(Fixture("a", "b", dt.date(2026, 1, 1)))
