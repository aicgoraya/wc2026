"""A deterministic SYNTHETIC artifact for tests, CI and container smoke checks.

The matches are invented (five fictional teams, results drawn from a seeded
RNG), so the bundle's predictions mean nothing about real football. The
manifest and ``evaluation.json`` are both marked ``synthetic: true`` and the
API repeats that flag in every response.
"""

import datetime as dt
from pathlib import Path

import numpy as np

from wc2026.data.schema import Match, MatchStatus, matches_to_frame
from wc2026.serving.artifacts import Manifest

SYNTHETIC_TEAMS: dict[str, float] = {
    "alpha": 1.0,
    "bravo": 0.5,
    "charlie": 0.0,
    "delta": -0.5,
    "echo": -1.0,
}
SYNTHETIC_CUTOFF = dt.date(2020, 1, 1)
_GBM_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "n_estimators": 30,
    "learning_rate": 0.1,
    "num_leaves": 8,
    "min_child_samples": 20,
    "verbose": -1,
    "seed": 0,
    "deterministic": True,
    "force_row_wise": True,
    "n_jobs": 1,
}


def synthetic_matches(n_rounds: int = 60, seed: int = 0) -> list[Match]:
    """A seeded round-robin among the fictional teams, one match per day."""
    rng = np.random.default_rng(seed)
    teams = list(SYNTHETIC_TEAMS)
    matches: list[Match] = []
    day = SYNTHETIC_CUTOFF - dt.timedelta(days=n_rounds * len(teams) * (len(teams) - 1) + 1)
    for _ in range(n_rounds):
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                day += dt.timedelta(days=1)
                edge = SYNTHETIC_TEAMS[home] - SYNTHETIC_TEAMS[away]
                logits = np.array([0.3 + edge, 0.0, 0.3 - edge])
                outcome = int(rng.choice(3, p=np.exp(logits) / np.exp(logits).sum()))
                home_goals, away_goals = {0: (2, 0), 1: (1, 1), 2: (0, 2)}[outcome]
                matches.append(
                    Match(
                        match_id=f"syn{len(matches)}",
                        date=day,
                        home_id=home,
                        away_id=away,
                        home_goals=home_goals,
                        away_goals=away_goals,
                        neutral=len(matches) % 3 == 0,
                        tournament="friendly",
                        status=MatchStatus.FINISHED,
                    )
                )
    return matches


def build_synthetic_bundle(
    bundle_dir: Path, *, version: str = "synthetic-fixture", with_evaluation: bool = True
) -> Manifest:
    """Train tiny models on the synthetic matches and write a bundle."""
    from wc2026.pipeline.export import historical_evaluation, train_and_write

    if bundle_dir.exists():
        raise FileExistsError(f"{bundle_dir} already exists; artifact versions are immutable")
    frame = matches_to_frame(synthetic_matches())
    evaluation = None
    if with_evaluation:
        window = (
            SYNTHETIC_CUTOFF - dt.timedelta(days=500),
            SYNTHETIC_CUTOFF - dt.timedelta(days=1),
        )
        evaluation = {
            "schema_version": 1,
            "synthetic": True,
            "note": "SYNTHETIC TEST FIXTURE - invented teams and results; not real model"
            " performance.",
            "metric": "rps",
            "historical": historical_evaluation(
                frame, window=window, cadence_days=250, min_train=100, seed=0
            ),
            "market": {"available": False, "n": 0, "reason": "synthetic fixture has no odds"},
        }
    return train_and_write(
        frame,
        bundle_dir,
        version=version,
        cutoff=SYNTHETIC_CUTOFF,
        synthetic=True,
        evaluation=evaluation,
        gbm_params=_GBM_PARAMS,
        training_data={"sources": "synthetic (wc2026.serving.fixture)", "seed": 0},
        n_probes=4,
    )
