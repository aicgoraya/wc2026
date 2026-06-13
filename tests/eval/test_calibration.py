import numpy as np

from wc2026.eval.calibration import ece, reliability_plot, reliability_table


def synthetic(n: int, *, calibrated: bool, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[type-arg]
    """Random 3-class forecasts; outcomes drawn FROM the forecast if calibrated."""
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(np.ones(3), size=n)
    if calibrated:
        outcomes = np.array([rng.choice(3, p=p) for p in probs])
    else:  # outcomes ignore the forecast entirely
        outcomes = rng.integers(0, 3, size=n)
    return probs, outcomes.astype(np.int64)


class TestReliability:
    def test_calibrated_forecasts_have_low_ece(self) -> None:
        probs, outcomes = synthetic(4000, calibrated=True)
        assert ece(probs, outcomes) < 0.03

    def test_miscalibrated_forecasts_have_higher_ece(self) -> None:
        probs, good = synthetic(4000, calibrated=True)
        _, bad = synthetic(4000, calibrated=False, seed=1)
        assert ece(probs, bad) > ece(probs, good)

    def test_table_shape_and_counts(self) -> None:
        probs, outcomes = synthetic(500, calibrated=True)
        table = reliability_table(probs, outcomes, bins=10)
        assert table["n"].sum() == 500 * 3  # pooled one-vs-rest
        assert (
            (table["p_pred"] >= table["bin_low"]) & (table["p_pred"] <= table["bin_high"])
        ).all()

    def test_plot_smoke(self) -> None:
        probs, outcomes = synthetic(300, calibrated=True)
        fig = reliability_plot(probs, outcomes, title="test")
        assert fig.axes[0].get_title() == "test"
