"""Phase 4: hierarchical Bayesian Poisson (PyMC) with partial pooling.

Latent attack/defence as random effects, decay-weighted likelihood, full
posterior uncertainty on every prediction. Convergence diagnostics (R-hat,
ESS, traces) are part of the deliverable, not an afterthought.

Requires the ``bayes`` extra (``uv sync --extra bayes``).
"""

import datetime as dt

import numpy as np
import numpy.typing as npt
import pandas as pd

from wc2026.models.base import Fixture, OutcomeProbs, ScorelineDist


class BayesPoissonForecaster:
    """PyMC hierarchical Poisson model (implements ``PosteriorForecaster``)."""

    name = "bayes_poisson"

    def fit(self, history: pd.DataFrame, as_of: dt.date) -> None:
        """Sample the posterior with NUTS; stores the InferenceData for diagnostics."""
        raise NotImplementedError("ships in Phase 4")

    def predict(self, fixture: Fixture) -> OutcomeProbs:
        """Posterior-mean 1X2."""
        raise NotImplementedError("ships in Phase 4")

    def predict_scoreline(self, fixture: Fixture) -> ScorelineDist:
        """Posterior-predictive scoreline grid."""
        raise NotImplementedError("ships in Phase 4")

    def predict_posterior(self, fixture: Fixture, n_draws: int) -> npt.NDArray[np.float64]:
        """(n_draws, 3) outcome probabilities across posterior draws."""
        raise NotImplementedError("ships in Phase 4")
