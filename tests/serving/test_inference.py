import datetime as dt
import itertools
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wc2026.data.schema import MATCH_COLUMNS, matches_to_frame
from wc2026.features.build import (
    FEATURE_COLUMNS,
    build_feature_matrix,
    fixture_features,
    team_states_as_of,
)
from wc2026.pipeline import export
from wc2026.serving import inference
from wc2026.serving.artifacts import ArtifactCorruptError, sha256_file
from wc2026.serving.fixture import SYNTHETIC_CUTOFF, SYNTHETIC_TEAMS, synthetic_matches
from wc2026.serving.inference import (
    UnknownTeamError,
    UnsupportedDateError,
    load_artifact,
)

IN_WINDOW = SYNTHETIC_CUTOFF + dt.timedelta(days=10)


def test_every_pairing_yields_a_valid_distribution(master_bundle: Path) -> None:
    predictor = load_artifact(master_bundle).predictor
    for home, away in itertools.permutations(SYNTHETIC_TEAMS, 2):
        for neutral in (True, False):
            result = predictor.predict(
                home, away, IN_WINDOW, neutral=neutral, tournament="friendly"
            )
            for probs in (result.blend, result.dixon_coles, result.gbm):
                assert all(math.isfinite(p) and 0.0 <= p <= 1.0 for p in probs)
                assert sum(probs) == pytest.approx(1.0, abs=1e-9)


def test_blend_is_the_weighted_mix_of_its_components(master_bundle: Path) -> None:
    predictor = load_artifact(master_bundle).predictor
    r = predictor.predict("alpha", "echo", IN_WINDOW, neutral=True, tournament="friendly")
    w = predictor.blend_weights
    expected = w["dixon_coles"] * r.dixon_coles.as_array() + w["gbm"] * r.gbm.as_array()
    assert r.blend.as_array() == pytest.approx(expected)
    assert sum(w.values()) == pytest.approx(1.0)


def test_model_learned_the_synthetic_strength_order(master_bundle: Path) -> None:
    """Sanity: the strongest fictional team is favoured over the weakest."""
    predictor = load_artifact(master_bundle).predictor
    r = predictor.predict("alpha", "echo", IN_WINDOW, neutral=True, tournament="friendly")
    assert r.blend.home > 0.5 > r.blend.away


def test_served_features_equal_the_training_matrix_row() -> None:
    """Online features (saved state) == what build_feature_matrix computes offline."""
    finished = matches_to_frame(synthetic_matches())
    fixture_date = SYNTHETIC_CUTOFF + dt.timedelta(days=7)
    scheduled: dict[str, object] = dict.fromkeys(MATCH_COLUMNS)
    scheduled.update(
        match_id="future",
        date=pd.Timestamp(fixture_date),
        home_id="bravo",
        away_id="delta",
        neutral=False,
        tournament="fifa_world_cup",
        status="scheduled",
        went_to_shootout=False,
    )
    with_fixture = pd.concat(
        [finished, pd.DataFrame([scheduled]).astype(finished.dtypes.to_dict())], ignore_index=True
    )
    offline = build_feature_matrix(with_fixture).loc["future"]

    states = team_states_as_of(finished, SYNTHETIC_CUTOFF)
    restored = {
        team: type(state).from_dict(json.loads(json.dumps(state.to_dict())))
        for team, state in states.items()
    }
    online = fixture_features(
        restored["bravo"],
        restored["delta"],
        fixture_date,
        neutral=False,
        tournament="fifa_world_cup",
    )
    for column in FEATURE_COLUMNS:
        assert online[column] == pytest.approx(float(offline[column]), abs=1e-12), column


def test_state_ignores_matches_on_or_after_the_cutoff() -> None:
    frame = matches_to_frame(synthetic_matches())
    early = dt.date(2019, 6, 1)
    full = team_states_as_of(frame, early)
    truncated = team_states_as_of(frame[frame["date"] < pd.Timestamp(early)], early)
    assert {t: s.to_dict() for t, s in full.items()} == {
        t: s.to_dict() for t, s in truncated.items()
    }


def test_unknown_team_is_named(master_bundle: Path) -> None:
    predictor = load_artifact(master_bundle).predictor
    with pytest.raises(UnknownTeamError) as err:
        predictor.predict("alpha", "atlantis", IN_WINDOW, neutral=True, tournament="friendly")
    assert err.value.teams == ["atlantis"]


def test_supported_date_window_is_inclusive_and_enforced(master_bundle: Path) -> None:
    predictor = load_artifact(master_bundle).predictor
    m = predictor.manifest
    for ok in (m.training_cutoff, m.max_prediction_date):
        predictor.predict("alpha", "bravo", ok, neutral=True, tournament="friendly")
    for bad in (
        m.training_cutoff - dt.timedelta(days=1),
        m.max_prediction_date + dt.timedelta(days=1),
    ):
        with pytest.raises(UnsupportedDateError):
            predictor.predict("alpha", "bravo", bad, neutral=True, tournament="friendly")


def test_concurrent_predictions_match_serial_ones(master_bundle: Path) -> None:
    predictor = load_artifact(master_bundle).predictor
    pairs = list(itertools.permutations(SYNTHETIC_TEAMS, 2)) * 10

    def run(pair: tuple[str, str]) -> tuple[float, ...]:
        r = predictor.predict(pair[0], pair[1], IN_WINDOW, neutral=True, tournament="friendly")
        return tuple(r.blend)

    serial = [run(p) for p in pairs]
    with ThreadPoolExecutor(max_workers=8) as pool:
        threaded = list(pool.map(run, pairs))
    assert threaded == serial


def test_non_lightgbm_model_file_is_rejected_even_with_a_valid_hash(bundle: Path) -> None:
    path = bundle / "gbm.txt"
    path.write_text("this is not a lightgbm model\n")
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["files"]["gbm.txt"] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ArtifactCorruptError):
        load_artifact(bundle)


def test_export_refuses_a_bundle_whose_served_output_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The export's parity gate: serving-side drift fails the export."""
    real = inference.fixture_features

    def drifted(*args: object, **kwargs: object) -> dict[str, float]:
        feats = real(*args, **kwargs)  # type: ignore[arg-type]
        feats["elo_diff"] += 400.0
        return feats

    monkeypatch.setattr(inference, "fixture_features", drifted)
    with pytest.raises(export.ExportError, match="differs from the pipeline"):
        export.train_and_write(
            matches_to_frame(synthetic_matches()),
            tmp_path / "v1",
            version="v1",
            cutoff=SYNTHETIC_CUTOFF,
            synthetic=True,
            training_data={},
            gbm_params={
                "objective": "multiclass",
                "num_class": 3,
                "n_estimators": 20,
                "verbose": -1,
            },
        )


def test_synthetic_evaluation_is_labelled_and_json_safe(master_bundle: Path) -> None:
    evaluation = load_artifact(master_bundle).evaluation
    assert evaluation is not None
    assert evaluation["synthetic"] is True
    assert evaluation["market"]["available"] is False
    models = {row["model"]: row for row in evaluation["historical"]["models"]}
    assert set(models) == {"elo_baseline", "dixon_coles", "gbm"}
    assert all(row["n"] > 0 and np.isfinite(row["rps"]) for row in models.values())
    assert evaluation["historical"]["baseline"] is None
