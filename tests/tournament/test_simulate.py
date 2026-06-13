import datetime as dt
import itertools

import numpy as np
import pytest

from wc2026.data.schema import Match, MatchStatus, Stage, matches_to_frame
from wc2026.models.base import Fixture, ScorelineDist
from wc2026.tournament.simulate import (
    STAGES,
    KnockoutPolicy,
    simulate_tournament,
)

GROUPS = "ABCDEFGHIJKL"


def make_wc_fixtures(strengths: dict[str, float]) -> list[Match]:
    """12 groups of 4, full round-robin, all scheduled (neutral)."""
    matches = []
    day = dt.date(2026, 6, 11)
    mid = 0
    for g in GROUPS:
        teams = [f"{g.lower()}_t{i}" for i in range(4)]
        for home, away in itertools.combinations(teams, 2):
            mid += 1
            matches.append(
                Match(
                    match_id=f"wc_{mid}",
                    date=day,
                    home_id=home,
                    away_id=away,
                    neutral=True,
                    tournament="fifa_world_cup",
                    stage=Stage.GROUP,
                    group=g,
                    status=MatchStatus.SCHEDULED,
                )
            )
    return matches


class StubModel:
    """Independent-Poisson scoreline model driven by a per-team strength dict."""

    name = "stub"

    def __init__(self, strengths: dict[str, float], k: int = 8) -> None:
        self._s = strengths
        self._k = k

    def fit(self, history, as_of) -> None:  # type: ignore[no-untyped-def]
        pass

    def predict(self, fixture: Fixture):  # type: ignore[no-untyped-def]
        return self.predict_scoreline(fixture).outcome_probs()

    def predict_scoreline(self, fixture: Fixture) -> ScorelineDist:
        from scipy.stats import poisson

        sh = self._s.get(fixture.home_id, 0.0)
        sa = self._s.get(fixture.away_id, 0.0)
        lam_h = float(np.exp(0.2 + 0.5 * (sh - sa)))
        lam_a = float(np.exp(0.2 + 0.5 * (sa - sh)))
        support = np.arange(self._k + 1)
        ph = poisson.pmf(support, lam_h)
        pa = poisson.pmf(support, lam_a)
        ph[self._k] = 1 - poisson.cdf(self._k - 1, lam_h)
        pa[self._k] = 1 - poisson.cdf(self._k - 1, lam_a)
        grid = np.outer(ph, pa)
        return ScorelineDist(grid / grid.sum())


@pytest.fixture(scope="module")
def sim_result():  # type: ignore[no-untyped-def]
    rng_strength = np.random.default_rng(0)
    teams = [f"{g.lower()}_t{i}" for g in GROUPS for i in range(4)]
    strengths = {t: float(rng_strength.normal(0, 1)) for t in teams}
    strengths["a_t0"] = 4.0  # a clear favourite
    frame = matches_to_frame(make_wc_fixtures(strengths))
    model = StubModel(strengths)
    return simulate_tournament(
        frame, model, n_sims=4000, rng=np.random.default_rng(7), policy=KnockoutPolicy()
    )


class TestInvariants:
    def test_stage_totals_are_exact(self, sim_result) -> None:  # type: ignore[no-untyped-def]
        # exactly this many teams reach each stage every single sim
        expected = {
            "reach_r32": 32,
            "reach_r16": 16,
            "reach_qf": 8,
            "reach_sf": 4,
            "reach_final": 2,
            "champion": 1,
        }
        for stage, total in expected.items():
            assert sim_result[stage].sum() == pytest.approx(total, abs=1e-9)

    def test_probabilities_in_range(self, sim_result) -> None:  # type: ignore[no-untyped-def]
        for stage in STAGES:
            assert (sim_result[stage] >= 0).all() and (sim_result[stage] <= 1).all()

    def test_monotone_across_stages(self, sim_result) -> None:  # type: ignore[no-untyped-def]
        for earlier, later in itertools.pairwise(STAGES):
            assert (sim_result[earlier] >= sim_result[later] - 1e-12).all()

    def test_all_48_teams_present(self, sim_result) -> None:  # type: ignore[no-untyped-def]
        assert len(sim_result) == 48
        assert sim_result["group"].value_counts().to_dict() == dict.fromkeys(GROUPS, 4)


class TestFavorite:
    def test_strong_team_leads_the_field(self, sim_result) -> None:  # type: ignore[no-untyped-def]
        top = sim_result.iloc[0]
        assert top["team_id"] == "a_t0"
        assert top["champion"] > 0.15  # well above the 1/48 ~ 0.02 baseline
        assert top["reach_r32"] > 0.95  # a dominant team almost always advances


def test_conditioning_on_completed_results() -> None:
    # force group A's t0 to have lost all three real matches: it must not win the group
    strengths = {f"{g.lower()}_t{i}": 0.0 for g in GROUPS for i in range(4)}
    fixtures = make_wc_fixtures(strengths)
    finished = []
    for m in fixtures:
        if m.group == "A" and "a_t0" in (m.home_id, m.away_id):
            # a_t0 loses 0-3 every game
            loses_home = m.home_id == "a_t0"
            finished.append(
                m.model_copy(
                    update={
                        "home_goals": 0 if loses_home else 3,
                        "away_goals": 3 if loses_home else 0,
                        "status": MatchStatus.FINISHED,
                    }
                )
            )
        else:
            finished.append(m)
    frame = matches_to_frame(finished)
    result = simulate_tournament(
        frame, StubModel(strengths), n_sims=500, rng=np.random.default_rng(1)
    )
    a_t0 = result.set_index("team_id").loc["a_t0"]
    assert a_t0["reach_r16"] < 0.10  # bottom of its group after losing all three
