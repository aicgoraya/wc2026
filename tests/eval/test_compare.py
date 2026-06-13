import numpy as np
import pytest

from wc2026.eval.compare import (
    compare,
    diebold_mariano,
    paired_bootstrap_delta,
)


class TestPairedBootstrap:
    def test_a_uniformly_better_has_negative_ci(self) -> None:
        rng = np.random.default_rng(0)
        loss_b = rng.uniform(0.1, 0.3, 2000)
        loss_a = loss_b - 0.02  # A always 0.02 lower (better)
        mean, lo, hi = paired_bootstrap_delta(loss_a, loss_b, seed=1)
        assert mean == pytest.approx(-0.02, abs=1e-9)
        assert hi < 0  # CI strictly below zero

    def test_pairing_cancels_shared_variance(self) -> None:
        # huge shared match-to-match variance, tiny consistent edge: a paired
        # test sees the edge, marginal CIs would be swamped
        rng = np.random.default_rng(2)
        shared = rng.uniform(0.0, 0.5, 5000)  # match difficulty
        loss_a = shared + rng.normal(0, 0.001, 5000)
        loss_b = shared + 0.005 + rng.normal(0, 0.001, 5000)
        _, lo, hi = paired_bootstrap_delta(loss_a, loss_b, seed=3)
        assert lo < -0.005 < hi or hi < 0  # detects A better despite shared noise
        assert hi < 0

    def test_no_real_difference_ci_straddles_zero(self) -> None:
        rng = np.random.default_rng(4)
        loss_a = rng.uniform(0.1, 0.3, 3000)
        loss_b = rng.uniform(0.1, 0.3, 3000)
        _, lo, hi = paired_bootstrap_delta(loss_a, loss_b, seed=5)
        assert lo < 0 < hi

    def test_deterministic(self) -> None:
        rng = np.random.default_rng(6)
        a, b = rng.uniform(size=500), rng.uniform(size=500)
        assert paired_bootstrap_delta(a, b, seed=7) == paired_bootstrap_delta(a, b, seed=7)

    def test_misaligned_raises(self) -> None:
        with pytest.raises(ValueError, match="aligned"):
            paired_bootstrap_delta(np.zeros(5), np.zeros(6), seed=0)


class TestDieboldMariano:
    def test_sign_and_significance(self) -> None:
        rng = np.random.default_rng(8)
        loss_b = rng.uniform(0.1, 0.3, 1000)
        loss_a = loss_b - 0.01
        stat, p = diebold_mariano(loss_a, loss_b)
        assert stat < 0  # A better => negative
        assert p < 0.01

    def test_identical_losses(self) -> None:
        x = np.array([0.2, 0.3, 0.1])
        stat, p = diebold_mariano(x, x)
        assert stat == 0.0 and p == 1.0

    def test_symmetry(self) -> None:
        rng = np.random.default_rng(9)
        a, b = rng.uniform(size=400), rng.uniform(size=400)
        s1, p1 = diebold_mariano(a, b)
        s2, p2 = diebold_mariano(b, a)
        assert s1 == pytest.approx(-s2)
        assert p1 == pytest.approx(p2)


class TestCompare:
    def test_winner_and_significance(self) -> None:
        rng = np.random.default_rng(10)
        loss_b = rng.uniform(0.1, 0.3, 4000)
        loss_a = loss_b - 0.015
        result = compare("dixon_coles", loss_a, "elo", loss_b, metric="rps", seed=11)
        assert result.winner == "dixon_coles"
        assert result.significant
        assert result.mean_delta < 0
        assert result.dm_stat < 0 and result.dm_pvalue < 0.01
        assert result.n == 4000

    def test_no_winner_when_tied(self) -> None:
        rng = np.random.default_rng(12)
        loss_a = rng.uniform(0.1, 0.3, 2000)
        loss_b = loss_a + rng.normal(0, 0.05, 2000)  # symmetric noise, no edge
        result = compare("a", loss_a, "b", loss_b, metric="rps", seed=13)
        assert result.winner is None
        assert not result.significant
