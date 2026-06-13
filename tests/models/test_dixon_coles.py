import datetime as dt

import numpy as np
import pytest
from scipy.optimize import approx_fprime
from scipy.stats import poisson

from wc2026.data.schema import Match, MatchStatus, matches_to_frame
from wc2026.models.base import Fixture
from wc2026.models.dixon_coles import (
    DixonColesForecaster,
    _FitData,
    _nll_and_grad,
)


def random_fit_data(n: int = 60, n_teams: int = 5, seed: int = 0) -> _FitData:
    rng = np.random.default_rng(seed)
    return _FitData(
        home_idx=rng.integers(0, n_teams, n),
        away_idx=(rng.integers(0, n_teams, n) + 1) % n_teams,
        goals_h=rng.poisson(1.4, n),
        goals_a=rng.poisson(1.1, n),
        is_home=rng.integers(0, 2, n).astype(np.float64),
        weights=rng.uniform(0.2, 1.0, n),
        n_teams=n_teams,
    )


class TestGradient:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_analytic_matches_numeric(self, seed: int) -> None:
        data = random_fit_data(seed=seed)
        rng = np.random.default_rng(seed + 100)
        theta = np.concatenate([[0.2, 0.25, 0.06, -0.08], rng.normal(0, 0.3, 2 * data.n_teams)])
        _, analytic = _nll_and_grad(theta, data, l2=3.0)
        numeric = approx_fprime(theta, lambda t: _nll_and_grad(t, data, l2=3.0)[0], 1e-7)
        np.testing.assert_allclose(analytic, numeric, rtol=2e-4, atol=2e-4)


def simulate_matches(
    attack: dict[str, float],
    defence: dict[str, float],
    *,
    n_rounds: int,
    intercept: float = 0.25,
    home_adv: float = 0.3,
    neutral_adv: float = 0.0,
    seed: int = 0,
    start: dt.date = dt.date(2015, 1, 1),
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
                venue = neutral_adv if neutral else home_adv
                lam_h = np.exp(intercept + attack[home] - defence[away] + venue)
                lam_a = np.exp(intercept + attack[away] - defence[home])
                matches.append(
                    Match(
                        match_id=f"sim{i}",
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


TRUE_ATTACK = {"alpha": 0.5, "bravo": 0.2, "charlie": -0.1, "delta": -0.6, "echo": 0.0}
TRUE_DEFENCE = {"alpha": 0.4, "bravo": 0.0, "charlie": -0.2, "delta": -0.2, "echo": 0.0}


@pytest.fixture(scope="module")
def fitted() -> DixonColesForecaster:
    history = simulate_matches(TRUE_ATTACK, TRUE_DEFENCE, n_rounds=40)
    model = DixonColesForecaster(half_life_days=1e6, l2=1.0)  # ~no decay for recovery
    model.fit(matches_to_frame(history), as_of=dt.date(2020, 1, 1))
    return model


class TestFit:
    def test_parameter_recovery(self, fitted: DixonColesForecaster) -> None:
        params = fitted.params
        idx = params.index()
        true_centered = {t: a - np.mean(list(TRUE_ATTACK.values())) for t, a in TRUE_ATTACK.items()}
        for team, true_a in true_centered.items():
            assert params.attack[idx[team]] == pytest.approx(true_a, abs=0.12)
        assert params.home_adv == pytest.approx(0.3, abs=0.1)

    def test_sum_to_zero_exactly(self, fitted: DixonColesForecaster) -> None:
        assert fitted.params.attack.mean() == pytest.approx(0.0, abs=1e-12)
        assert fitted.params.defence.mean() == pytest.approx(0.0, abs=1e-12)

    def test_too_few_matches_raises(self) -> None:
        history = simulate_matches(TRUE_ATTACK, TRUE_DEFENCE, n_rounds=2)
        with pytest.raises(ValueError, match="at least 500"):
            DixonColesForecaster().fit(matches_to_frame(history), as_of=dt.date(2020, 1, 1))

    def test_no_leak_future_ignored(self) -> None:
        history = simulate_matches(TRUE_ATTACK, TRUE_DEFENCE, n_rounds=30)
        future = simulate_matches(
            TRUE_ATTACK, TRUE_DEFENCE, n_rounds=3, seed=9, start=dt.date(2021, 1, 1)
        )
        as_of = dt.date(2020, 6, 1)
        clean = DixonColesForecaster(l2=1.0)
        clean.fit(matches_to_frame(history), as_of=as_of)
        poisoned = DixonColesForecaster(l2=1.0)
        poisoned.fit(matches_to_frame(history + future), as_of=as_of)
        fixture = Fixture("alpha", "delta", dt.date(2020, 7, 1))
        assert clean.predict(fixture) == poisoned.predict(fixture)

    def test_shrinkage_pulls_sparse_teams_to_average(self) -> None:
        history = simulate_matches(TRUE_ATTACK, TRUE_DEFENCE, n_rounds=40)
        # a team seen twice, winning both 3-0: raw MLE would inflate it
        sparse = [
            Match(
                match_id=f"sp{i}",
                date=dt.date(2019, 6, 1 + i),
                home_id="newcomer",
                away_id="echo",
                home_goals=3,
                away_goals=0,
                neutral=True,
                tournament="friendly",
                status=MatchStatus.FINISHED,
            )
            for i in range(2)
        ]
        model = DixonColesForecaster(half_life_days=1e6, l2=5.0)
        model.fit(matches_to_frame(history + sparse), as_of=dt.date(2020, 1, 1))
        params = model.params
        newcomer_attack = params.attack[params.index()["newcomer"]]
        alpha_attack = params.attack[params.index()["alpha"]]
        assert 0 < newcomer_attack < alpha_attack  # nudged up, NOT past the proven best

    def test_neutral_listed_home_edge_recovered(self) -> None:
        # data generated WITH a listed-home edge at neutral venues: the fitted
        # neutral_adv must pick it up, and predictions at neutral must be
        # asymmetric between the listed orientations
        history = simulate_matches(TRUE_ATTACK, TRUE_DEFENCE, n_rounds=40, neutral_adv=0.18, seed=7)
        model = DixonColesForecaster(half_life_days=1e6, l2=1.0)
        model.fit(matches_to_frame(history), as_of=dt.date(2020, 1, 1))
        assert model.params.neutral_adv == pytest.approx(0.18, abs=0.08)
        listed = model.predict(Fixture("echo", "bravo", dt.date(2020, 7, 1), neutral=True))
        flipped = model.predict(Fixture("bravo", "echo", dt.date(2020, 7, 1), neutral=True))
        assert listed.home > flipped.away  # the listed side keeps the neutral edge

    def test_neutral_adv_near_zero_when_absent(self, fitted: DixonColesForecaster) -> None:
        # the module fixture's data has no neutral edge: coefficient stays small
        assert abs(fitted.params.neutral_adv) < 0.08

    def test_decay_prefers_recent_form(self) -> None:
        # zulu was great long ago, terrible recently; with a short half-life
        # the model must rate them below average
        old = simulate_matches(
            {"zulu": 0.8, "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0},
            {"zulu": 0.5, "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0},
            n_rounds=25,
            seed=3,
            start=dt.date(2000, 1, 1),
        )
        recent = simulate_matches(
            {"zulu": -0.8, "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0},
            {"zulu": -0.5, "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0},
            n_rounds=25,
            seed=4,
            start=dt.date(2018, 1, 1),
        )
        model = DixonColesForecaster(half_life_days=365.0, l2=1.0)
        model.fit(matches_to_frame(old + recent), as_of=dt.date(2020, 1, 1))
        params = model.params
        assert params.attack[params.index()["zulu"]] < 0


class TestPredict:
    def test_grid_cells_hand_computed(self, fitted: DixonColesForecaster) -> None:
        params = fitted.params
        fixture = Fixture("alpha", "delta", dt.date(2020, 7, 1), neutral=True)
        dist = fitted.predict_scoreline(fixture)
        idx = params.index()
        lam_h = np.exp(
            params.intercept
            + params.attack[idx["alpha"]]
            - params.defence[idx["delta"]]
            + params.neutral_adv  # fixture is neutral: listed-home term applies
        )
        lam_a = np.exp(
            params.intercept + params.attack[idx["delta"]] - params.defence[idx["alpha"]]
        )
        raw = np.outer(poisson.pmf(np.arange(11), lam_h), poisson.pmf(np.arange(11), lam_a))
        raw[10, :] = 0  # tail cells differ; compare an interior + the tau cells
        expected_00 = raw[0, 0] * (1 - lam_h * lam_a * params.rho)
        expected_11 = raw[1, 1] * (1 - params.rho)
        norm_estimate = dist.prob(0, 0) / expected_00
        assert dist.prob(1, 1) / expected_11 == pytest.approx(norm_estimate, rel=1e-6)
        assert dist.prob(2, 2) / raw[2, 2] == pytest.approx(norm_estimate, rel=1e-6)

    def test_home_advantage_only_when_true_home(self, fitted: DixonColesForecaster) -> None:
        neutral = fitted.predict(Fixture("echo", "charlie", dt.date(2020, 7, 1), neutral=True))
        at_home = fitted.predict(Fixture("echo", "charlie", dt.date(2020, 7, 1), neutral=False))
        assert at_home.home > neutral.home

    def test_unseen_team_is_average(self, fitted: DixonColesForecaster) -> None:
        probs = fitted.predict(Fixture("alpha", "atlantis", dt.date(2020, 7, 1)))
        assert probs.home > probs.away  # alpha is above-average, unknown is average
        dist = fitted.predict_scoreline(Fixture("atlantis", "unknown2", dt.date(2020, 7, 1)))
        assert dist.matrix.sum() == pytest.approx(1.0, abs=1e-9)

    def test_better_attack_means_higher_win_prob(self, fitted: DixonColesForecaster) -> None:
        strong = fitted.predict(Fixture("alpha", "delta", dt.date(2020, 7, 1)))
        weak = fitted.predict(Fixture("delta", "alpha", dt.date(2020, 7, 1)))
        assert strong.home > 0.5 > weak.home

    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="fit"):
            DixonColesForecaster().predict(Fixture("a", "b", dt.date(2020, 1, 1)))
