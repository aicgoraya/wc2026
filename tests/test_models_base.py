import datetime as dt

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from wc2026.data.schema import Stage
from wc2026.models.base import Fixture, OutcomeProbs, ScorelineDist


class TestOutcomeProbs:
    def test_validated_ok(self) -> None:
        probs = OutcomeProbs(0.5, 0.3, 0.2).validated()
        assert probs.home == 0.5

    @pytest.mark.parametrize(
        "bad",
        [
            OutcomeProbs(0.5, 0.3, 0.3),  # sums to 1.1
            OutcomeProbs(-0.1, 0.6, 0.5),  # negative
            OutcomeProbs(1.2, -0.1, -0.1),  # out of range
            OutcomeProbs(float("nan"), 0.5, 0.5),
        ],
    )
    def test_validated_rejects(self, bad: OutcomeProbs) -> None:
        with pytest.raises(ValueError, match="probabilities"):
            bad.validated()

    def test_as_array(self) -> None:
        arr = OutcomeProbs(0.2, 0.3, 0.5).as_array()
        assert arr.shape == (3,)
        assert arr.tolist() == [0.2, 0.3, 0.5]


def uniform_dist(k: int) -> ScorelineDist:
    size = (k + 1) ** 2
    return ScorelineDist(np.full((k + 1, k + 1), 1.0 / size))


class TestScorelineDist:
    def test_hand_computed_outcomes(self) -> None:
        # P(0-0)=.1 P(0-1)=.2 P(1-0)=.3 P(1-1)=.4
        dist = ScorelineDist(np.array([[0.1, 0.2], [0.3, 0.4]]))
        probs = dist.outcome_probs()
        assert probs.home == pytest.approx(0.3)
        assert probs.draw == pytest.approx(0.5)
        assert probs.away == pytest.approx(0.2)

    def test_prob_lookup_and_bounds(self) -> None:
        dist = ScorelineDist(np.array([[0.1, 0.2], [0.3, 0.4]]))
        assert dist.prob(1, 0) == pytest.approx(0.3)
        assert dist.max_goals == 1
        with pytest.raises(ValueError, match="outside"):
            dist.prob(2, 0)

    def test_top_scorelines(self) -> None:
        dist = ScorelineDist(np.array([[0.1, 0.2], [0.3, 0.4]]))
        assert dist.top_scorelines(2) == [(1, 1, pytest.approx(0.4)), (1, 0, pytest.approx(0.3))]

    def test_sample_shape_range_and_determinism(self) -> None:
        dist = uniform_dist(5)
        a = dist.sample(np.random.default_rng(7), 500)
        b = dist.sample(np.random.default_rng(7), 500)
        assert a.shape == (500, 2)
        assert a.min() >= 0 and a.max() <= 5
        np.testing.assert_array_equal(a, b)

    def test_sample_matches_marginals(self) -> None:
        dist = ScorelineDist(np.array([[0.0, 0.0], [1.0, 0.0]]))  # always 1-0
        draws = dist.sample(np.random.default_rng(0), 50)
        assert (draws == [1, 0]).all()

    @pytest.mark.parametrize(
        "matrix",
        [
            np.array([[0.5, 0.5]]),  # not square
            np.array([[1.0]]),  # K < 1
            np.array([[0.6, -0.1], [0.3, 0.2]]),  # negative
            np.array([[0.5, 0.5], [0.5, 0.5]]),  # sums to 2
            np.array([[np.inf, 0.0], [0.0, 0.0]]),  # non-finite
        ],
    )
    def test_invalid_matrix_rejected(self, matrix: np.ndarray) -> None:  # type: ignore[type-arg]
        with pytest.raises(ValueError, match="matrix"):
            ScorelineDist(matrix)

    def test_matrix_is_readonly(self) -> None:
        dist = uniform_dist(2)
        with pytest.raises(ValueError, match="read-only"):
            dist.matrix[0, 0] = 1.0


@settings(max_examples=50, deadline=None)
@given(
    raw=arrays(
        np.float64,
        (7, 7),
        elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    ).filter(lambda m: m.sum() > 1e-6)
)
def test_outcome_probs_always_a_distribution(raw: np.ndarray) -> None:  # type: ignore[type-arg]
    dist = ScorelineDist(raw / raw.sum())
    probs = dist.outcome_probs()  # .validated() inside would raise on violation
    assert probs.home + probs.draw + probs.away == pytest.approx(1.0)
    assert min(probs) >= 0.0


def test_fixture_defaults() -> None:
    fixture = Fixture("argentina", "france", dt.date(2026, 7, 19), stage=Stage.FINAL)
    assert fixture.neutral is True
    assert fixture.stage is Stage.FINAL
