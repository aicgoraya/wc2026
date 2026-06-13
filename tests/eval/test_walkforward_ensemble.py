import datetime as dt

import numpy as np
import pandas as pd

from wc2026.eval.ensemble import walk_forward_ensemble


def model_frame(probs: np.ndarray, outcomes: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:  # type: ignore[type-arg]
    n = len(outcomes)
    return pd.DataFrame(
        {
            "match_id": [f"m{i}" for i in range(n)],
            "date": dates,
            "p_home": probs[:, 0],
            "p_draw": probs[:, 1],
            "p_away": probs[:, 2],
            "outcome": outcomes,
        }
    )


def test_rolling_weights_blend_not_worse_than_strong_base() -> None:
    rng = np.random.default_rng(0)
    n = 6000
    dates = pd.to_datetime([dt.date(2018, 1, 1) + dt.timedelta(days=i // 3) for i in range(n)])
    outcomes = rng.integers(0, 3, n)
    strong = np.full((n, 3), 0.12)
    strong[np.arange(n), outcomes] = 0.76
    strong = strong / strong.sum(axis=1, keepdims=True)
    weak = rng.dirichlet(np.ones(3), n)
    rows = {
        "dixon_coles": model_frame(strong, outcomes, dates),
        "weak": model_frame(weak, outcomes, dates),
    }
    result = walk_forward_ensemble(rows, cadence_days=180, min_train=1000, seed=1)
    assert result.n_oos > 0
    blend_rps = result.scoreboard.loc[result.scoreboard["model"] == "blend", "rps"].item()
    strong_rps = result.scoreboard.loc[result.scoreboard["model"] == "dixon_coles", "rps"].item()
    assert blend_rps <= strong_rps + 0.003  # weight rolls onto the strong model
    # the weak model gets little weight on average
    assert result.weights_trajectory["weak"].mean() < 0.3
    assert not result.weights_trajectory.empty


def test_complementary_models_blend_beats_each_walk_forward() -> None:
    rng = np.random.default_rng(2)
    n = 8000
    dates = pd.to_datetime([dt.date(2018, 1, 1) + dt.timedelta(days=i // 4) for i in range(n)])
    outcomes = rng.integers(0, 3, n)
    even = np.arange(n) % 2 == 0
    a = rng.dirichlet(np.ones(3), n)
    b = rng.dirichlet(np.ones(3), n)
    a[even] = 0.1
    a[even, outcomes[even]] = 0.8
    b[~even] = 0.1
    b[~even, outcomes[~even]] = 0.8
    a = a / a.sum(axis=1, keepdims=True)
    b = b / b.sum(axis=1, keepdims=True)
    rows = {"dixon_coles": model_frame(a, outcomes, dates), "gbm": model_frame(b, outcomes, dates)}
    result = walk_forward_ensemble(rows, cadence_days=120, min_train=1500, seed=3)
    blend_rps = result.scoreboard.loc[result.scoreboard["model"] == "blend", "rps"].item()
    base_min = result.scoreboard[result.scoreboard["model"] != "blend"]["rps"].min()
    assert blend_rps < base_min  # blend strictly better, out-of-sample, rolling
    assert (result.paired["verdict"].str.startswith("blend")).any()
    # per-year stability series is populated
    assert set(result.per_period.columns) == {"year", "mean_dRPS", "n"}
    assert result.per_period["n"].sum() == result.n_oos
