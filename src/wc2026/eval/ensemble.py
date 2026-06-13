"""Probability blending — where model combination earns its keep.

A convex blend ``p = Σ w_k p_k`` of several models' 1X2 probabilities, with the
weights fit to minimise out-of-sample log loss. Weight fitting is the only new
place leakage can hide, so it is split in time: weights are fit on an EARLIER
window of the base models' (already out-of-sample) predictions and the blend is
evaluated on a strictly LATER window. The paired test then asks whether the
blend beats the best single model on that held-out later window.

Base predictions must themselves be walk-forward / leak-free; this module only
fits the combination weights.
"""

import dataclasses
import datetime as dt
from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize

from wc2026.eval import scoring
from wc2026.eval.compare import compare

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def fit_blend_weights(probs: Sequence[FloatArray], outcomes: IntArray) -> FloatArray:
    """Convex weights (sum to 1, non-negative) minimising blend log loss.

    ``probs`` is a list of (n, 3) probability arrays, one per base model,
    aligned on the same matches; ``outcomes`` is (n,) in {0,1,2}.
    """
    stack = np.stack([np.asarray(p, dtype=np.float64) for p in probs], axis=0)  # (m, n, 3)
    m, n, _ = stack.shape
    idx = np.arange(n)
    out = np.asarray(outcomes)

    def neg_loglik(w: FloatArray) -> float:
        blend = np.tensordot(w, stack, axes=(0, 0))  # (n, 3)
        picked = blend[idx, out]
        return float(-np.log(np.clip(picked, 1e-12, 1.0)).mean())

    start = np.full(m, 1.0 / m)
    result = minimize(
        neg_loglik,
        start,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * m,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 500, "ftol": 1e-9},
    )
    weights: FloatArray = np.clip(result.x, 0.0, None)
    normed: FloatArray = weights / weights.sum()
    return normed


def blend(probs: Sequence[FloatArray], weights: FloatArray) -> FloatArray:
    """Weighted convex combination of (n, 3) probability arrays."""
    stack = np.stack([np.asarray(p, dtype=np.float64) for p in probs], axis=0)
    result: FloatArray = np.tensordot(weights, stack, axes=(0, 0))
    return result


@dataclasses.dataclass(frozen=True)
class EnsembleResult:
    """Outputs of the time-split ensemble evaluation."""

    models: tuple[str, ...]
    weights: dict[str, float]
    split_date: dt.date
    n_train: int
    n_test: int
    scoreboard: pd.DataFrame  # RPS on the held-out test window: each base + blend
    paired: pd.DataFrame  # blend vs each base on the test window

    @property
    def blend_beats_best_single(self) -> bool:
        """True if the blend's test RPS is the lowest and significantly so vs best base."""
        best_base = self.scoreboard[self.scoreboard["model"] != "blend"]["rps"].min()
        blend_rps = self.scoreboard.loc[self.scoreboard["model"] == "blend", "rps"].item()
        if blend_rps >= best_base:
            return False
        return bool((self.paired["verdict"].str.startswith("blend")).any())


@dataclasses.dataclass(frozen=True)
class WalkForwardEnsembleResult:
    """Rolling-weights ensemble: refit at each cadence, aggregate OOS predictions."""

    models: tuple[str, ...]
    n_oos: int
    scoreboard: pd.DataFrame  # RPS over the aggregated OOS window: each base + blend
    paired: pd.DataFrame  # blend vs each base on the aggregated OOS predictions
    weights_trajectory: pd.DataFrame  # weights at each refit cutoff
    per_period: pd.DataFrame  # blend-minus-DC mean ΔRPS per rolled block (stability)


def walk_forward_ensemble(
    model_rows: Mapping[str, pd.DataFrame],
    *,
    cadence_days: int,
    min_train: int,
    seed: int,
    baseline: str = "dixon_coles",
) -> WalkForwardEnsembleResult:
    """Refit blend weights at each cadence on expanding pre-cutoff data; roll forward.

    At every cutoff the weights are fit on the base models' (already OOS)
    predictions for matches STRICTLY before the cutoff, then applied to the next
    block of matches. The blend predictions are therefore out-of-sample for the
    weights too. ``min_train`` is the smallest training set before the first
    block is scored. Returns the aggregated OOS scoreboard, the paired blend-vs-
    base tests, the weight trajectory, and a per-block blend-minus-baseline
    ΔRPS series so the edge's stability across time is visible.
    """
    names = tuple(model_rows)
    frames = {
        name: (df.set_index("match_id") if "match_id" in df.columns else df)
        for name, df in model_rows.items()
    }
    shared = frames[names[0]].index
    for name in names[1:]:
        shared = shared.intersection(frames[name].index)
    aligned = {name: frames[name].loc[shared].sort_values("date") for name in names}
    order = aligned[names[0]].index
    dates = aligned[names[0]]["date"].to_numpy()
    outcomes = aligned[names[0]]["outcome"].to_numpy(dtype=np.int64)
    probs = {
        n: aligned[n].loc[order][["p_home", "p_draw", "p_away"]].to_numpy(np.float64) for n in names
    }

    n = len(order)
    blend_oos = np.full((n, 3), np.nan)
    traj: list[dict[str, object]] = []
    cadence = np.timedelta64(cadence_days, "D")

    start = min_train
    cutoff = dates[start]
    while start < n:
        block = (dates >= cutoff) & (dates < cutoff + cadence)
        if not block.any():
            cutoff = cutoff + cadence
            if cutoff > dates[-1]:
                break
            continue
        train = dates < cutoff
        if train.sum() >= min_train:
            weights = fit_blend_weights([probs[nm][train] for nm in names], outcomes[train])
            blend_oos[block] = blend([probs[nm][block] for nm in names], weights)
            traj.append(
                {
                    "cutoff": pd.Timestamp(cutoff),
                    "n_train": int(train.sum()),
                    "n_block": int(block.sum()),
                }
                | {nm: float(w) for nm, w in zip(names, weights, strict=True)}
            )
        cutoff = cutoff + cadence
        start = int(np.searchsorted(dates, cutoff))

    scored = ~np.isnan(blend_oos[:, 0])
    out = outcomes[scored]
    blend_p = blend_oos[scored]
    base_p = {nm: probs[nm][scored] for nm in names}

    rows = [{"model": nm, "rps": float(scoring.rps(base_p[nm], out).mean())} for nm in names]
    rows.append({"model": "blend", "rps": float(scoring.rps(blend_p, out).mean())})
    scoreboard = pd.DataFrame(rows).sort_values("rps", ignore_index=True)

    blend_rps = scoring.rps(blend_p, out)
    paired_rows = []
    for nm in names:
        cmp = compare("blend", blend_rps, nm, scoring.rps(base_p[nm], out), metric="rps", seed=seed)
        paired_rows.append(
            {
                "comparison": f"blend - {nm}",
                "n": cmp.n,
                "mean_dRPS": cmp.mean_delta,
                "ci_lo": cmp.ci_lo,
                "ci_hi": cmp.ci_hi,
                "p": cmp.dm_pvalue,
                "verdict": (
                    f"{cmp.winner} better (p={cmp.dm_pvalue:.1e})"
                    if cmp.winner
                    else "no significant difference"
                ),
            }
        )

    # per-calendar-year blend-minus-baseline ΔRPS, to show stability vs concentration
    scored_dates = pd.to_datetime(dates[scored])
    delta = scoring.rps(blend_p, out) - scoring.rps(base_p[baseline], out)
    per_period = (
        pd.DataFrame({"year": scored_dates.year, "blend_minus_base_rps": delta})
        .groupby("year")
        .agg(mean_dRPS=("blend_minus_base_rps", "mean"), n=("blend_minus_base_rps", "size"))
        .reset_index()
    )

    return WalkForwardEnsembleResult(
        models=names,
        n_oos=int(scored.sum()),
        scoreboard=scoreboard,
        paired=pd.DataFrame(paired_rows),
        weights_trajectory=pd.DataFrame(traj),
        per_period=per_period,
    )


def evaluate_ensemble(
    model_rows: Mapping[str, pd.DataFrame],
    split_date: dt.date,
    *,
    seed: int,
) -> EnsembleResult:
    """Fit blend weights on pre-split matches; score the blend on post-split.

    ``model_rows`` maps model name -> its walk-forward frame (indexed by
    match_id, with p_home/p_draw/p_away, outcome, date). All models are aligned
    on shared matches.
    """
    names = tuple(model_rows)
    frames = {
        name: df.set_index("match_id") if "match_id" in df.columns else df
        for name, df in model_rows.items()
    }
    shared = frames[names[0]].index
    for name in names[1:]:
        shared = shared.intersection(frames[name].index)
    aligned = {name: frames[name].loc[shared].sort_index() for name in names}

    dates = aligned[names[0]]["date"]
    train_mask = (dates < pd.Timestamp(split_date)).to_numpy()
    test_mask = ~train_mask
    outcomes = aligned[names[0]]["outcome"].to_numpy(dtype=np.int64)

    def probs_of(name: str) -> FloatArray:
        return aligned[name][["p_home", "p_draw", "p_away"]].to_numpy(dtype=np.float64)

    train_probs = [probs_of(n)[train_mask] for n in names]
    weights = fit_blend_weights(train_probs, outcomes[train_mask])

    test_probs = [probs_of(n)[test_mask] for n in names]
    test_out = outcomes[test_mask]
    blend_test = blend(test_probs, weights)

    rows = []
    for name, p in zip(names, test_probs, strict=True):
        rows.append({"model": name, "rps": float(scoring.rps(p, test_out).mean())})
    rows.append({"model": "blend", "rps": float(scoring.rps(blend_test, test_out).mean())})
    scoreboard = pd.DataFrame(rows).sort_values("rps", ignore_index=True)

    paired_rows = []
    blend_rps = scoring.rps(blend_test, test_out)
    for name, p in zip(names, test_probs, strict=True):
        cmp = compare("blend", blend_rps, name, scoring.rps(p, test_out), metric="rps", seed=seed)
        paired_rows.append(
            {
                "comparison": f"blend - {name}",
                "n": cmp.n,
                "mean_dRPS": cmp.mean_delta,
                "ci_lo": cmp.ci_lo,
                "ci_hi": cmp.ci_hi,
                "p": cmp.dm_pvalue,
                "verdict": (
                    f"{cmp.winner} better (p={cmp.dm_pvalue:.1e})"
                    if cmp.winner
                    else "no significant difference"
                ),
            }
        )

    return EnsembleResult(
        models=names,
        weights={n: float(w) for n, w in zip(names, weights, strict=True)},
        split_date=split_date,
        n_train=int(train_mask.sum()),
        n_test=int(test_mask.sum()),
        scoreboard=scoreboard,
        paired=pd.DataFrame(paired_rows),
    )
