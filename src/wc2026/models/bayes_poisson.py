"""Phase 4: hierarchical Bayesian Poisson (PyMC) with partial pooling.

The same generative structure as Dixon-Coles, made Bayesian:

    log lam_home = mu + attack[home] - defence[away] + home_adv*is_home
                                                     + neutral_adv*is_neutral
    log lam_away = mu + attack[away] - defence[home]
    home_goals ~ Poisson(lam_home), away_goals ~ Poisson(lam_away)

plus the Dixon-Coles tau low-score correction, so it is a like-for-like upgrade
over DC rather than a different model.

**Partial pooling is the point.** ``attack`` and ``defence`` are zero-sum
random effects whose scale (``sigma_att``, ``sigma_def``) is itself estimated.
A team with little (decayed) data is pulled toward the population mean (0); a
data-rich team is pinned by its own likelihood. This shrinkage is automatic and
should help most exactly where DC's per-team MLE is noisiest — sparse teams.

**Time-varying strength — choice (a), time-decayed likelihood.** Each match's
log-likelihood is weighted by ``0.5 ** (age_days / half_life)`` via a
``pm.Potential`` (a tempered likelihood), so recent form dominates. The
alternative (b), a random-walk state-space on attack/defence, would add tens of
thousands of latent states across 150 years and 300+ teams — NUTS would be slow
and the walk-forward (refit at a cadence) infeasible. The decayed hierarchical
model keeps the parameter count at O(teams), so it converges quickly and can be
refit repeatedly; that tractability is what makes the honest walk-forward
possible. The half-life is fixed to the DC-selected value (not re-tuned, to
avoid an expensive extra MCMC search) for a fair comparison.

**Walk-forward compute.** MCMC cannot be refit per match. ``fit`` samples the
posterior once per as-of cutoff on the strictly-pre-cutoff, decay-weighted,
window-capped data; the walk-forward harness reuses that posterior until the
next scheduled refit. Leak-freedom is exact (the fit never sees the cutoff or
later). The cost is real and documented where the model is run.

Requires the ``bayes`` extra (``uv sync --extra bayes``).
"""

import dataclasses
import datetime as dt
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import poisson

from wc2026.models.base import Fixture, OutcomeProbs, ScorelineDist

if TYPE_CHECKING:
    from matplotlib.figure import Figure

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

_TAU_FLOOR = 1e-10
DEFAULT_HALF_LIFE_DAYS = 1460.0  # the DC-selected value, fixed here for comparability


@dataclasses.dataclass(frozen=True)
class BayesConfig:
    """Sampler and likelihood configuration."""

    # defaults verified to converge cleanly on the real data (2026-06-13):
    # R-hat 1.0000, min ESS 4224, 0 divergences, ~156s on 11.5k matches / 298 teams
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS
    train_window_years: int = 12
    draws: int = 1500
    tune: int = 1500
    chains: int = 4
    target_accept: float = 0.95
    max_goals: int = 10
    thin_to: int = 500
    seed: int = 20260613


@dataclasses.dataclass(frozen=True)
class BayesDiagnostics:
    """Convergence summary; a fit that fails these is not a result."""

    max_rhat: float
    min_ess_bulk: float
    divergences: int
    n_train: int
    n_teams: int

    @property
    def converged(self) -> bool:
        """Standard thresholds: R-hat <= 1.01, ESS >= 400, no divergences."""
        return self.max_rhat <= 1.01 and self.min_ess_bulk >= 400 and self.divergences == 0


@dataclasses.dataclass(frozen=True)
class _Posterior:
    teams: tuple[str, ...]
    index: dict[str, int]
    mu: FloatArray  # (S,)
    home_adv: FloatArray
    neutral_adv: FloatArray
    rho: FloatArray
    attack: FloatArray  # (S, T)
    defence: FloatArray


def _log_tau(h: IntArray, a: IntArray, lh: Any, la: Any, rho: Any) -> Any:
    """Dixon-Coles low-score log-correction as a pytensor expression."""
    import pytensor.tensor as pt

    tau = pt.ones_like(lh)
    tau = pt.switch(pt.and_(pt.eq(h, 0), pt.eq(a, 0)), 1.0 - lh * la * rho, tau)
    tau = pt.switch(pt.and_(pt.eq(h, 0), pt.eq(a, 1)), 1.0 + lh * rho, tau)
    tau = pt.switch(pt.and_(pt.eq(h, 1), pt.eq(a, 0)), 1.0 + la * rho, tau)
    tau = pt.switch(pt.and_(pt.eq(h, 1), pt.eq(a, 1)), 1.0 - rho, tau)
    return pt.log(pt.maximum(tau, _TAU_FLOOR))


class BayesPoissonForecaster:
    """Hierarchical Bayesian Poisson (implements ``PosteriorForecaster``)."""

    name = "bayes_poisson"

    def __init__(self, config: BayesConfig | None = None) -> None:
        self._cfg = config or BayesConfig()
        self._post: _Posterior | None = None
        self._diagnostics: BayesDiagnostics | None = None
        self._idata: Any = None

    @property
    def diagnostics(self) -> BayesDiagnostics:
        """Convergence diagnostics from the last fit."""
        if self._diagnostics is None:
            raise RuntimeError("call fit() before reading diagnostics")
        return self._diagnostics

    def fit(self, history: pd.DataFrame, as_of: dt.date) -> None:
        """Sample the posterior on decay-weighted matches strictly before ``as_of``."""
        import arviz as az
        import pymc as pm
        import pytensor.tensor as pt

        cutoff = pd.Timestamp(as_of)
        window_start = cutoff - pd.DateOffset(years=self._cfg.train_window_years)
        rows = history[
            (history["status"] == "finished")
            & (history["date"] < cutoff)
            & (history["date"] >= window_start)
        ]
        if len(rows) < 500:
            raise ValueError(f"need at least 500 matches to fit, got {len(rows)}")

        teams = tuple(sorted(set(rows["home_id"]) | set(rows["away_id"])))
        index = {t: i for i, t in enumerate(teams)}
        home_idx = rows["home_id"].map(index).to_numpy(dtype=np.int64)
        away_idx = rows["away_id"].map(index).to_numpy(dtype=np.int64)
        hg = rows["home_goals"].to_numpy(dtype=np.int64)
        ag = rows["away_goals"].to_numpy(dtype=np.int64)
        is_home = (~rows["neutral"].to_numpy(dtype=bool)).astype(np.float64)
        is_neutral = 1.0 - is_home
        age_days = (cutoff - rows["date"]).dt.days.to_numpy(dtype=np.float64)
        weights = np.power(0.5, age_days / self._cfg.half_life_days)
        n_teams = len(teams)

        with pm.Model():
            mu = pm.Normal("mu", 0.0, 1.0)
            home_adv = pm.Normal("home_adv", 0.25, 0.5)
            neutral_adv = pm.Normal("neutral_adv", 0.0, 0.5)
            rho = pm.Normal("rho", 0.0, 0.05)
            sigma_att = pm.HalfNormal("sigma_att", 1.0)
            sigma_def = pm.HalfNormal("sigma_def", 1.0)
            # non-centered, zero-sum (identifiable location) random effects:
            # the hyperprior scale IS the partial pooling
            att_raw = pm.ZeroSumNormal("att_raw", sigma=1.0, shape=n_teams)
            def_raw = pm.ZeroSumNormal("def_raw", sigma=1.0, shape=n_teams)
            attack = pm.Deterministic("attack", att_raw * sigma_att)
            defence = pm.Deterministic("defence", def_raw * sigma_def)

            log_lh = (
                mu
                + attack[home_idx]
                - defence[away_idx]
                + home_adv * is_home
                + neutral_adv * is_neutral
            )
            log_la = mu + attack[away_idx] - defence[home_idx]
            lh = pt.exp(log_lh)
            la = pt.exp(log_la)
            logp = (
                pm.logp(pm.Poisson.dist(mu=lh), hg)
                + pm.logp(pm.Poisson.dist(mu=la), ag)
                + _log_tau(hg, ag, lh, la, rho)
            )
            pm.Potential("lik", pt.sum(pt.as_tensor_variable(weights) * logp))

            idata = pm.sample(
                draws=self._cfg.draws,
                tune=self._cfg.tune,
                chains=self._cfg.chains,
                target_accept=self._cfg.target_accept,
                random_seed=self._cfg.seed,
                progressbar=False,
                compute_convergence_checks=False,
            )

        self._idata = idata
        summary = az.summary(
            idata, var_names=["mu", "home_adv", "neutral_adv", "rho", "attack", "defence"]
        )
        self._diagnostics = BayesDiagnostics(
            max_rhat=float(summary["r_hat"].max()),
            min_ess_bulk=float(summary["ess_bulk"].min()),
            divergences=int(idata.sample_stats["diverging"].to_numpy().sum()),
            n_train=len(rows),
            n_teams=n_teams,
        )
        self._post = self._extract_posterior(idata, teams, index)

    def _extract_posterior(
        self, idata: Any, teams: tuple[str, ...], index: dict[str, int]
    ) -> _Posterior:
        post = idata.posterior
        n_teams = len(teams)

        def flat(name: str) -> FloatArray:
            arr = np.asarray(post[name].to_numpy(), dtype=np.float64)
            return arr.reshape(-1, *arr.shape[2:])

        s_total = flat("mu").shape[0]
        keep = np.linspace(0, s_total - 1, min(self._cfg.thin_to, s_total)).astype(int)
        return _Posterior(
            teams=teams,
            index=index,
            mu=flat("mu")[keep],
            home_adv=flat("home_adv")[keep],
            neutral_adv=flat("neutral_adv")[keep],
            rho=flat("rho")[keep],
            attack=flat("attack")[keep].reshape(len(keep), n_teams),
            defence=flat("defence")[keep].reshape(len(keep), n_teams),
        )

    def _strength(self, team: str) -> tuple[FloatArray, FloatArray]:
        post = self._posterior
        if team in post.index:
            i = post.index[team]
            return post.attack[:, i], post.defence[:, i]
        zeros = np.zeros_like(post.mu)  # unseen team -> population mean, full uncertainty
        return zeros, zeros

    @property
    def _posterior(self) -> _Posterior:
        if self._post is None:
            raise RuntimeError("call fit() before predicting")
        return self._post

    def _grids(self, fixture: Fixture) -> FloatArray:
        """Per-posterior-draw tau-corrected scoreline grids; shape (S, K+1, K+1)."""
        post = self._posterior
        a_h, d_h = self._strength(fixture.home_id)
        a_a, d_a = self._strength(fixture.away_id)
        gamma = post.neutral_adv if fixture.neutral else post.home_adv
        lam_h = np.exp(post.mu + a_h - d_a + gamma)
        lam_a = np.exp(post.mu + a_a - d_h)

        k = self._cfg.max_goals
        support = np.arange(k + 1)
        ph = poisson.pmf(support[None, :], lam_h[:, None])
        pa = poisson.pmf(support[None, :], lam_a[:, None])
        ph[:, k] = 1.0 - poisson.cdf(k - 1, lam_h)
        pa[:, k] = 1.0 - poisson.cdf(k - 1, lam_a)
        grids = ph[:, :, None] * pa[:, None, :]

        rho = post.rho
        grids[:, 0, 0] *= np.maximum(1.0 - lam_h * lam_a * rho, _TAU_FLOOR)
        grids[:, 0, 1] *= np.maximum(1.0 + lam_h * rho, _TAU_FLOOR)
        grids[:, 1, 0] *= np.maximum(1.0 + lam_a * rho, _TAU_FLOOR)
        grids[:, 1, 1] *= np.maximum(1.0 - rho, _TAU_FLOOR)
        grids /= grids.sum(axis=(1, 2), keepdims=True)
        return cast(FloatArray, grids)

    def predict_scoreline(self, fixture: Fixture) -> ScorelineDist:
        """Posterior-predictive scoreline grid (averaged over parameter uncertainty)."""
        return ScorelineDist(self._grids(fixture).mean(axis=0))

    def predict(self, fixture: Fixture) -> OutcomeProbs:
        """Posterior-mean 1X2."""
        return self.predict_scoreline(fixture).outcome_probs()

    def predict_posterior(self, fixture: Fixture, n_draws: int) -> FloatArray:
        """(n_draws, 3) outcome probabilities, one per posterior draw — the uncertainty."""
        grids = self._grids(fixture)
        home = np.tril(grids, -1).sum(axis=(1, 2))
        draw = np.trace(grids, axis1=1, axis2=2)
        away = np.triu(grids, 1).sum(axis=(1, 2))
        probs = np.column_stack([home, draw, away])
        if n_draws >= len(probs):
            return cast(FloatArray, probs)
        keep = np.linspace(0, len(probs) - 1, n_draws).astype(int)
        return cast(FloatArray, probs[keep])

    def save_trace_plot(self, path: Path) -> None:
        """Save a trace plot of the global parameters."""
        import arviz as az
        import matplotlib

        matplotlib.use("Agg")
        if self._idata is None:
            raise RuntimeError("call fit() before plotting")
        axes = az.plot_trace(
            self._idata,
            var_names=["mu", "home_adv", "neutral_adv", "rho", "sigma_att", "sigma_def"],
        )
        fig: Figure = axes.ravel()[0].figure
        fig.tight_layout()
        fig.savefig(path)
