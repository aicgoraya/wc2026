import datetime as dt

import pytest

from wc2026.models.base import Fixture, OutcomeProbs
from wc2026.models.blend import BLEND_WEIGHTS, BlendForecaster


class StubModel:
    def __init__(self, probs: OutcomeProbs) -> None:
        self._probs = probs
        self.name = "stub"
        self.fitted_as_of: dt.date | None = None

    def fit(self, history, as_of) -> None:  # type: ignore[no-untyped-def]
        self.fitted_as_of = as_of

    def predict(self, fixture: Fixture) -> OutcomeProbs:
        return self._probs


def test_blend_is_convex_combination() -> None:
    a = StubModel(OutcomeProbs(0.6, 0.3, 0.1))
    b = StubModel(OutcomeProbs(0.2, 0.3, 0.5))
    blend = BlendForecaster(
        {"dixon_coles": a, "gbm": b}, weights={"dixon_coles": 0.75, "gbm": 0.25}
    )
    p = blend.predict(Fixture("x", "y", dt.date(2026, 6, 1)))
    assert p.home == pytest.approx(0.75 * 0.6 + 0.25 * 0.2)
    assert p.draw == pytest.approx(0.3)
    assert sum(p) == pytest.approx(1.0)


def test_weights_normalised() -> None:
    a = StubModel(OutcomeProbs(1.0, 0.0, 0.0))
    b = StubModel(OutcomeProbs(0.0, 0.0, 1.0))
    blend = BlendForecaster({"dixon_coles": a, "gbm": b}, weights={"dixon_coles": 2.0, "gbm": 2.0})
    p = blend.predict(Fixture("x", "y", dt.date(2026, 6, 1)))
    assert p.home == pytest.approx(0.5) and p.away == pytest.approx(0.5)


def test_fit_propagates_to_all_bases() -> None:
    a, b = StubModel(OutcomeProbs(0.5, 0.3, 0.2)), StubModel(OutcomeProbs(0.5, 0.3, 0.2))
    blend = BlendForecaster({"dixon_coles": a, "gbm": b})
    blend.fit(history=None, as_of=dt.date(2026, 6, 13))  # type: ignore[arg-type]
    assert a.fitted_as_of == dt.date(2026, 6, 13)
    assert b.fitted_as_of == dt.date(2026, 6, 13)


def test_unknown_weight_key_raises() -> None:
    a = StubModel(OutcomeProbs(0.5, 0.3, 0.2))
    with pytest.raises(ValueError, match="unknown models"):
        BlendForecaster({"dixon_coles": a}, weights={"missing": 1.0})


def test_default_weights_sum_positive() -> None:
    assert set(BLEND_WEIGHTS) == {"dixon_coles", "gbm"}
    assert sum(BLEND_WEIGHTS.values()) > 0
