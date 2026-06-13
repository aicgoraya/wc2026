"""Phase 2: Dixon-Coles bivariate Poisson with time decay.

Model (Dixon & Coles 1997): independent Poisson goal counts with
``log lam_home = c + attack_h - defence_a + gamma * is_true_home`` and
``log lam_away = c + attack_a - defence_h``, corrected for low-score
dependence by the tau factor on the (0,0), (0,1), (1,0), (1,1) cells:

    tau(0,0) = 1 - lam_h*lam_a*rho    tau(0,1) = 1 + lam_h*rho
    tau(1,0) = 1 + lam_a*rho          tau(1,1) = 1 - rho

Fitting is weighted MLE with exponential time decay
``w = 0.5 ** (age_days / half_life_days)`` and an analytic gradient
(gradient-checked in tests). Three deliberate choices:

- **Sparse teams**: an L2 penalty shrinks attack/defence toward 0 (the
  average team). The penalty is fixed while the decayed effective sample
  grows with data, so a 3-cap team is pulled hard toward average while a
  200-cap team is barely touched. Unseen teams predict as exactly average.
- **Identifiability**: optimization is unconstrained (the penalty pins the
  location softly); after fitting, attack/defence are centered to mean zero
  EXACTLY and the intercept absorbs the shift — a pure reparameterization
  that leaves every lambda unchanged.
- **Venue semantics**: identical to the Elo baseline — ``gamma`` applies
  only when ``neutral`` is False (true home games, incl. WC hosts).

The decay half-life is a hyperparameter selected leak-free on an inner
validation window that precedes the reported test period (see
``pipeline.tune``); ``DEFAULT_HALF_LIFE_DAYS`` records the frozen choice.
"""

import dataclasses
import datetime as dt

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from wc2026.models.base import Fixture, OutcomeProbs, ScorelineDist

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

DEFAULT_HALF_LIFE_DAYS = 1460.0
"""Frozen by the inner-window selection (pipeline.tune); see RESULTS.md."""

RHO_BOUND = 0.2
_LOG_LAMBDA_CLIP = (-10.0, 6.0)  # keeps exp() finite during line searches
_TAU_FLOOR = 1e-10


@dataclasses.dataclass(frozen=True)
class DCParams:
    """Fitted parameters in canonical (sum-to-zero) form."""

    teams: tuple[str, ...]
    intercept: float
    home_adv: float
    rho: float
    attack: FloatArray
    defence: FloatArray

    def index(self) -> dict[str, int]:
        """team_id -> row index."""
        return {team: i for i, team in enumerate(self.teams)}


def _tau_log_and_grads(
    goals_h: IntArray, goals_a: IntArray, lam_h: FloatArray, lam_a: FloatArray, rho: float
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Log tau and its partials wrt (log lam_h, log lam_a, rho), per match.

    Zero everywhere except the four low-score cells.
    """
    n = len(goals_h)
    tau = np.ones(n)
    d_u = np.zeros(n)  # d log tau / d log lam_h
    d_v = np.zeros(n)  # d log tau / d log lam_a
    d_rho = np.zeros(n)

    low = (goals_h <= 1) & (goals_a <= 1)
    cell = {
        (0, 0): low & (goals_h == 0) & (goals_a == 0),
        (0, 1): low & (goals_h == 0) & (goals_a == 1),
        (1, 0): low & (goals_h == 1) & (goals_a == 0),
        (1, 1): low & (goals_h == 1) & (goals_a == 1),
    }
    m = cell[(0, 0)]
    tau[m] = 1.0 - lam_h[m] * lam_a[m] * rho
    m = cell[(0, 1)]
    tau[m] = 1.0 + lam_h[m] * rho
    m = cell[(1, 0)]
    tau[m] = 1.0 + lam_a[m] * rho
    m = cell[(1, 1)]
    tau[m] = 1.0 - rho

    tau = np.maximum(tau, _TAU_FLOOR)

    m = cell[(0, 0)]
    d_u[m] = -lam_h[m] * lam_a[m] * rho / tau[m]
    d_v[m] = d_u[m]
    d_rho[m] = -lam_h[m] * lam_a[m] / tau[m]
    m = cell[(0, 1)]
    d_u[m] = lam_h[m] * rho / tau[m]
    d_rho[m] = lam_h[m] / tau[m]
    m = cell[(1, 0)]
    d_v[m] = lam_a[m] * rho / tau[m]
    d_rho[m] = lam_a[m] / tau[m]
    m = cell[(1, 1)]
    d_rho[m] = -1.0 / tau[m]

    return np.log(tau), d_u, d_v, d_rho


@dataclasses.dataclass(frozen=True)
class _FitData:
    """Index-encoded training matches."""

    home_idx: IntArray
    away_idx: IntArray
    goals_h: IntArray
    goals_a: IntArray
    is_home: FloatArray  # 1.0 when not neutral
    weights: FloatArray
    n_teams: int


def _nll_and_grad(theta: FloatArray, data: _FitData, l2: float) -> tuple[float, FloatArray]:
    """Penalized weighted negative log likelihood and its gradient."""
    t = data.n_teams
    intercept, home_adv, rho = theta[0], theta[1], theta[2]
    attack = theta[3 : 3 + t]
    defence = theta[3 + t :]

    u = intercept + attack[data.home_idx] - defence[data.away_idx] + home_adv * data.is_home
    v = intercept + attack[data.away_idx] - defence[data.home_idx]
    u = np.clip(u, *_LOG_LAMBDA_CLIP)
    v = np.clip(v, *_LOG_LAMBDA_CLIP)
    lam_h, lam_a = np.exp(u), np.exp(v)

    log_tau, dtau_u, dtau_v, dtau_rho = _tau_log_and_grads(
        data.goals_h, data.goals_a, lam_h, lam_a, rho
    )
    loglik = (
        data.goals_h * u - lam_h + data.goals_a * v - lam_a + log_tau
    )  # match-constant terms dropped
    nll = -float(np.dot(data.weights, loglik))
    nll += l2 * (float(np.dot(attack, attack)) + float(np.dot(defence, defence)))

    g_u = data.weights * (data.goals_h - lam_h + dtau_u)
    g_v = data.weights * (data.goals_a - lam_a + dtau_v)

    grad = np.empty_like(theta)
    grad[0] = -float(np.sum(g_u + g_v))
    grad[1] = -float(np.dot(g_u, data.is_home))
    grad[2] = -float(np.dot(data.weights, dtau_rho))
    grad_attack = -(
        np.bincount(data.home_idx, weights=g_u, minlength=t)
        + np.bincount(data.away_idx, weights=g_v, minlength=t)
    )
    grad_defence = np.bincount(data.away_idx, weights=g_u, minlength=t) + np.bincount(
        data.home_idx, weights=g_v, minlength=t
    )
    grad[3 : 3 + t] = grad_attack + 2.0 * l2 * attack
    grad[3 + t :] = grad_defence + 2.0 * l2 * defence
    return nll, grad


class DixonColesForecaster:
    """Dixon-Coles MLE model (implements ``ScorelineForecaster``)."""

    name = "dixon_coles"

    def __init__(
        self,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
        max_goals: int = 10,
        l2: float = 5.0,
        train_window_years: int = 25,
    ) -> None:
        self._half_life_days = half_life_days
        self._max_goals = max_goals
        self._l2 = l2
        self._train_window_years = train_window_years
        self._params: DCParams | None = None

    @property
    def params(self) -> DCParams:
        """Fitted parameters (canonical form); raises before ``fit``."""
        if self._params is None:
            raise RuntimeError("call fit() before reading params")
        return self._params

    def fit(self, history: pd.DataFrame, as_of: dt.date) -> None:
        """Weighted MLE on finished matches strictly before ``as_of``.

        Warm-starts from the previous fit when one exists (the walk-forward
        harness refits the same instance every cadence step).
        """
        cutoff = pd.Timestamp(as_of)
        window_start = cutoff - pd.DateOffset(years=self._train_window_years)
        rows = history[
            (history["status"] == "finished")
            & (history["date"] < cutoff)
            & (history["date"] >= window_start)
        ]
        if len(rows) < 500:
            raise ValueError(f"need at least 500 matches to fit Dixon-Coles, got {len(rows)}")

        teams = tuple(sorted(set(rows["home_id"]) | set(rows["away_id"])))
        index = {team: i for i, team in enumerate(teams)}
        age_days = (cutoff - rows["date"]).dt.days.to_numpy(dtype=np.float64)
        data = _FitData(
            home_idx=rows["home_id"].map(index).to_numpy(dtype=np.int64),
            away_idx=rows["away_id"].map(index).to_numpy(dtype=np.int64),
            goals_h=rows["home_goals"].to_numpy(dtype=np.int64),
            goals_a=rows["away_goals"].to_numpy(dtype=np.int64),
            is_home=(~rows["neutral"].to_numpy(dtype=bool)).astype(np.float64),
            weights=np.power(0.5, age_days / self._half_life_days),
            n_teams=len(teams),
        )

        theta0 = self._warm_start(teams)
        bounds = [(None, None), (None, None), (-RHO_BOUND, RHO_BOUND)]
        bounds += [(None, None)] * (2 * len(teams))
        result = minimize(
            _nll_and_grad,
            theta0,
            args=(data, self._l2),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500},
        )
        if not result.success and "ABNORMAL" in str(result.message).upper():
            raise RuntimeError(f"Dixon-Coles fit failed: {result.message}")

        t = len(teams)
        attack = result.x[3 : 3 + t]
        defence = result.x[3 + t :]
        # exact sum-to-zero canonicalization; intercept absorbs the shift so
        # every lambda is unchanged (mean_a enters both u and v symmetrically)
        mean_a, mean_d = float(attack.mean()), float(defence.mean())
        self._params = DCParams(
            teams=teams,
            intercept=float(result.x[0]) + mean_a - mean_d,
            home_adv=float(result.x[1]),
            rho=float(result.x[2]),
            attack=attack - mean_a,
            defence=defence - mean_d,
        )

    def _warm_start(self, teams: tuple[str, ...]) -> FloatArray:
        theta = np.zeros(3 + 2 * len(teams))
        theta[0] = np.log(1.3)  # ~mean international goals per side
        theta[1] = 0.3
        theta[2] = -0.05
        if self._params is not None:
            old = self._params.index()
            for i, team in enumerate(teams):
                if team in old:
                    j = old[team]
                    theta[3 + i] = self._params.attack[j]
                    theta[3 + len(teams) + i] = self._params.defence[j]
            theta[0] = self._params.intercept
            theta[1] = self._params.home_adv
            theta[2] = self._params.rho
        return theta

    def _lambdas(self, fixture: Fixture) -> tuple[float, float]:
        params = self.params
        index = params.index()
        attack = {t: params.attack[i] for t, i in index.items()}
        defence = {t: params.defence[i] for t, i in index.items()}
        a_h, d_h = attack.get(fixture.home_id, 0.0), defence.get(fixture.home_id, 0.0)
        a_a, d_a = attack.get(fixture.away_id, 0.0), defence.get(fixture.away_id, 0.0)
        gamma = 0.0 if fixture.neutral else params.home_adv
        u = np.clip(params.intercept + a_h - d_a + gamma, *_LOG_LAMBDA_CLIP)
        v = np.clip(params.intercept + a_a - d_h, *_LOG_LAMBDA_CLIP)
        return float(np.exp(u)), float(np.exp(v))

    def predict_scoreline(self, fixture: Fixture) -> ScorelineDist:
        """Tau-corrected joint goals grid with the tail folded at ``max_goals``."""
        lam_h, lam_a = self._lambdas(fixture)
        k = self._max_goals
        support = np.arange(k + 1)
        p_h = poisson.pmf(support, lam_h)
        p_a = poisson.pmf(support, lam_a)
        p_h[k] = 1.0 - poisson.cdf(k - 1, lam_h)  # fold the tail into the cap
        p_a[k] = 1.0 - poisson.cdf(k - 1, lam_a)
        grid = np.outer(p_h, p_a)

        rho = self.params.rho
        grid[0, 0] *= max(1.0 - lam_h * lam_a * rho, _TAU_FLOOR)
        grid[0, 1] *= max(1.0 + lam_h * rho, _TAU_FLOOR)
        grid[1, 0] *= max(1.0 + lam_a * rho, _TAU_FLOOR)
        grid[1, 1] *= max(1.0 - rho, _TAU_FLOOR)
        return ScorelineDist(grid / grid.sum())

    def predict(self, fixture: Fixture) -> OutcomeProbs:
        """1X2 by integrating the scoreline grid."""
        return self.predict_scoreline(fixture).outcome_probs()
