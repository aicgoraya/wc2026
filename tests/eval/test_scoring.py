import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from wc2026.eval.scoring import bootstrap_ci, brier, log_loss, outcome_codes, rps


class TestHandComputed:
    def test_rps(self) -> None:
        # p=(0.5,0.3,0.2), home win: ((0.5-1)^2 + (0.8-1)^2)/2 = 0.145
        out = rps(np.array([[0.5, 0.3, 0.2]]), np.array([0]))
        assert out[0] == pytest.approx(0.145)

    def test_rps_rewards_mass_near_outcome(self) -> None:
        # same prob on the true class, but mass adjacent vs far: RPS must differ
        near = rps(np.array([[0.5, 0.5, 0.0]]), np.array([0]))[0]
        far = rps(np.array([[0.5, 0.0, 0.5]]), np.array([0]))[0]
        assert near < far  # log loss and Brier would tie these

    def test_perfect_forecast_scores_zero(self) -> None:
        probs = np.array([[1.0, 0.0, 0.0]])
        assert rps(probs, np.array([0]))[0] == 0.0
        assert brier(probs, np.array([0]))[0] == 0.0
        assert log_loss(probs, np.array([0]))[0] == pytest.approx(0.0)

    def test_log_loss(self) -> None:
        out = log_loss(np.array([[0.5, 0.3, 0.2]]), np.array([1]))
        assert out[0] == pytest.approx(-np.log(0.3))

    def test_brier(self) -> None:
        # (0.5-0)^2 + (0.3-0)^2 + (0.2-1)^2 = 0.25 + 0.09 + 0.64 = 0.98
        out = brier(np.array([[0.5, 0.3, 0.2]]), np.array([2]))
        assert out[0] == pytest.approx(0.98)

    def test_uniform_rps(self) -> None:
        # (1/3-1)^2 + (2/3-1)^2 = 4/9+1/9 = 5/9; /2 = 5/18
        out = rps(np.array([[1 / 3, 1 / 3, 1 / 3]]), np.array([0]))
        assert out[0] == pytest.approx(5 / 18)


@settings(max_examples=50, deadline=None)
@given(
    raw=arrays(
        np.float64,
        (8, 3),
        elements=st.floats(min_value=1e-6, max_value=1.0, allow_nan=False),
    ),
    outcomes=arrays(np.int64, (8,), elements=st.integers(min_value=0, max_value=2)),
)
def test_score_bounds(raw: np.ndarray, outcomes: np.ndarray) -> None:  # type: ignore[type-arg]
    probs = raw / raw.sum(axis=1, keepdims=True)
    assert ((rps(probs, outcomes) >= 0) & (rps(probs, outcomes) <= 1)).all()
    assert (log_loss(probs, outcomes) >= 0).all()
    b = brier(probs, outcomes)
    assert ((b >= 0) & (b <= 2)).all()


class TestBootstrapCI:
    def test_contains_mean_and_deterministic(self) -> None:
        rng = np.random.default_rng(0)
        scores = rng.normal(0.2, 0.05, size=500)
        lo, hi = bootstrap_ci(scores, n_boot=2000, seed=7)
        assert lo < scores.mean() < hi
        assert bootstrap_ci(scores, n_boot=2000, seed=7) == (lo, hi)

    def test_narrows_with_sample_size(self) -> None:
        rng = np.random.default_rng(1)
        small = rng.normal(0.2, 0.05, size=50)
        big = np.tile(small, 40)
        lo_s, hi_s = bootstrap_ci(small, n_boot=2000, seed=3)
        lo_b, hi_b = bootstrap_ci(big, n_boot=2000, seed=3)
        assert (hi_b - lo_b) < (hi_s - lo_s)


def test_outcome_codes() -> None:
    home = np.array([2, 1, 0])
    away = np.array([1, 1, 3])
    np.testing.assert_array_equal(outcome_codes(home, away), [0, 1, 2])
