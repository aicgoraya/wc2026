import datetime as dt

import numpy as np
import pandas as pd
import pytest

from wc2026.eval.ensemble import blend, evaluate_ensemble, fit_blend_weights


class TestFitBlendWeights:
    def test_weights_form_a_simplex(self) -> None:
        rng = np.random.default_rng(0)
        n = 500
        probs = [rng.dirichlet(np.ones(3), n), rng.dirichlet(np.ones(3), n)]
        outcomes = rng.integers(0, 3, n)
        w = fit_blend_weights(probs, outcomes)
        assert w.shape == (2,)
        assert (w >= 0).all()
        assert w.sum() == pytest.approx(1.0)

    def test_puts_weight_on_the_informative_model(self) -> None:
        # model A's probabilities match the outcomes; model B is noise.
        rng = np.random.default_rng(1)
        n = 2000
        outcomes = rng.integers(0, 3, n)
        good = np.full((n, 3), 0.05)
        good[np.arange(n), outcomes] = 0.90
        noise = rng.dirichlet(np.ones(3), n)
        w = fit_blend_weights([good, noise], outcomes)
        assert w[0] > 0.8  # nearly all weight on the informative model

    def test_blend_combines(self) -> None:
        a = np.array([[0.6, 0.3, 0.1]])
        b = np.array([[0.2, 0.3, 0.5]])
        out = blend([a, b], np.array([0.5, 0.5]))
        np.testing.assert_allclose(out, [[0.4, 0.3, 0.3]])
        assert out.sum() == pytest.approx(1.0)


def _model_frame(probs: np.ndarray, outcomes: np.ndarray, dates: list[dt.date]) -> pd.DataFrame:  # type: ignore[type-arg]
    n = len(outcomes)
    return pd.DataFrame(
        {
            "match_id": [f"m{i}" for i in range(n)],
            "date": pd.to_datetime(dates),
            "p_home": probs[:, 0],
            "p_draw": probs[:, 1],
            "p_away": probs[:, 2],
            "outcome": outcomes,
        }
    )


class TestEvaluateEnsemble:
    def test_blend_beats_a_weak_member_out_of_sample(self) -> None:
        rng = np.random.default_rng(2)
        n = 4000
        outcomes = rng.integers(0, 3, n)
        dates = [dt.date(2018, 1, 1) + dt.timedelta(days=i) for i in range(n)]
        # strong model: confident & correct; weak model: noise
        strong = np.full((n, 3), 0.1)
        strong[np.arange(n), outcomes] = 0.8
        strong = strong / strong.sum(axis=1, keepdims=True)
        weak = rng.dirichlet(np.ones(3), n)
        rows = {
            "strong": _model_frame(strong, outcomes, dates),
            "weak": _model_frame(weak, outcomes, dates),
        }
        split = dates[n // 2]
        result = evaluate_ensemble(rows, split, seed=3)
        assert result.n_train > 0 and result.n_test > 0
        assert result.weights["strong"] > result.weights["weak"]
        # blend should not be worse than the strong base on the test window
        blend_rps = result.scoreboard.loc[result.scoreboard["model"] == "blend", "rps"].item()
        strong_rps = result.scoreboard.loc[result.scoreboard["model"] == "strong", "rps"].item()
        assert blend_rps <= strong_rps + 0.005

    def test_two_complementary_models_blend_better_than_either(self) -> None:
        # each model is good on a disjoint half of matches -> the blend wins
        rng = np.random.default_rng(4)
        n = 6000
        outcomes = rng.integers(0, 3, n)
        dates = [dt.date(2018, 1, 1) + dt.timedelta(days=i // 4) for i in range(n)]
        first_half = np.arange(n) % 2 == 0
        a = rng.dirichlet(np.ones(3), n)
        b = rng.dirichlet(np.ones(3), n)
        # model A sharp on even matches, B sharp on odd matches
        a[first_half] = 0.1
        a[first_half, outcomes[first_half]] = 0.8
        b[~first_half] = 0.1
        b[~first_half, outcomes[~first_half]] = 0.8
        a = a / a.sum(axis=1, keepdims=True)
        b = b / b.sum(axis=1, keepdims=True)
        rows = {"a": _model_frame(a, outcomes, dates), "b": _model_frame(b, outcomes, dates)}
        result = evaluate_ensemble(rows, dates[n // 2], seed=5)
        blend_rps = result.scoreboard.loc[result.scoreboard["model"] == "blend", "rps"].item()
        base_min = result.scoreboard[result.scoreboard["model"] != "blend"]["rps"].min()
        assert blend_rps < base_min  # complementary models: blend strictly better
        assert result.blend_beats_best_single
