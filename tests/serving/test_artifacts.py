import datetime as dt
import json
from pathlib import Path
from typing import Any

import pytest

from wc2026.features.build import TeamState
from wc2026.serving import artifacts
from wc2026.serving.artifacts import (
    ArtifactCorruptError,
    ArtifactIncompatibleError,
    ArtifactNotFoundError,
    read_bundle,
    validate_bundle,
)
from wc2026.serving.fixture import build_synthetic_bundle


def edit_manifest(bundle: Path, **changes: Any) -> None:
    path = bundle / "manifest.json"
    data = json.loads(path.read_text())
    data.update(changes)
    path.write_text(json.dumps(data))


def test_valid_bundle_passes_and_records_the_contract(bundle: Path) -> None:
    manifest = validate_bundle(bundle, expected_version="synthetic-fixture")
    assert manifest.synthetic is True
    assert manifest.training_cutoff == dt.date(2020, 1, 1)
    assert manifest.max_prediction_date > manifest.training_cutoff
    assert set(manifest.files) == {
        "dixon_coles.json",
        "gbm.txt",
        "team_state.json",
        "evaluation.json",
    }
    assert {"python", "numpy", "lightgbm"} <= set(manifest.runtime)


def test_bundle_contains_no_pickle(bundle: Path) -> None:
    for path in bundle.iterdir():
        assert path.suffix in {".json", ".txt"}
        assert not path.read_bytes().startswith(b"\x80")  # pickle protocol marker


def test_a_version_is_written_once(bundle: Path) -> None:
    with pytest.raises(FileExistsError):
        build_synthetic_bundle(bundle)


def test_missing_bundle_and_missing_file(bundle: Path, tmp_path: Path) -> None:
    with pytest.raises(ArtifactNotFoundError):
        validate_bundle(tmp_path / "nope")
    (bundle / "gbm.txt").unlink()
    with pytest.raises(ArtifactNotFoundError, match=r"gbm\.txt"):
        validate_bundle(bundle)


@pytest.mark.parametrize("name", ["gbm.txt", "dixon_coles.json", "team_state.json"])
def test_tampered_file_fails_its_hash(bundle: Path, name: str) -> None:
    path = bundle / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ArtifactCorruptError, match=name.replace(".", r"\.")):
        validate_bundle(bundle)


def test_truncated_manifest_is_corrupt(bundle: Path) -> None:
    (bundle / "manifest.json").write_text('{"schema_version": 1, "artifact_ver')
    with pytest.raises(ArtifactCorruptError):
        validate_bundle(bundle)


def test_manifest_missing_fields_is_corrupt(bundle: Path) -> None:
    (bundle / "manifest.json").write_text('{"schema_version": 1}')
    with pytest.raises(ArtifactCorruptError, match="invalid"):
        validate_bundle(bundle)


def test_unknown_manifest_schema_is_incompatible(bundle: Path) -> None:
    edit_manifest(bundle, schema_version=2)
    with pytest.raises(ArtifactIncompatibleError, match="schema_version"):
        validate_bundle(bundle)


def test_pinned_version_must_match(bundle: Path) -> None:
    with pytest.raises(ArtifactIncompatibleError, match="pinned"):
        validate_bundle(bundle, expected_version="some-other-version")


@pytest.mark.parametrize(
    "changes",
    [
        {"feature_schema_version": "0"},
        {"feature_columns": ["elo_diff"]},
    ],
)
def test_feature_schema_mismatch_is_incompatible(bundle: Path, changes: dict[str, Any]) -> None:
    edit_manifest(bundle, **changes)
    with pytest.raises(ArtifactIncompatibleError, match="feature schema"):
        validate_bundle(bundle)


def test_lightgbm_major_version_mismatch_is_incompatible(bundle: Path) -> None:
    edit_manifest(bundle, runtime={"lightgbm": "1.0.0"})
    with pytest.raises(ArtifactIncompatibleError, match="lightgbm"):
        validate_bundle(bundle)


def test_manifest_cannot_omit_a_required_file(bundle: Path) -> None:
    data = json.loads((bundle / "manifest.json").read_text())
    del data["files"]["team_state.json"]
    (bundle / "manifest.json").write_text(json.dumps(data))
    with pytest.raises(ArtifactIncompatibleError, match="required"):
        validate_bundle(bundle)


@pytest.mark.parametrize("name", ["../escape.json", "/etc/passwd", "manifest.json", "a/b.json"])
def test_manifest_file_names_cannot_escape_the_bundle(bundle: Path, name: str) -> None:
    data = json.loads((bundle / "manifest.json").read_text())
    data["files"][name] = {"sha256": "0" * 64, "bytes": 1}
    (bundle / "manifest.json").write_text(json.dumps(data))
    with pytest.raises(ArtifactCorruptError):
        validate_bundle(bundle)


def test_malformed_model_json_with_a_matching_hash_is_still_rejected(bundle: Path) -> None:
    """A publisher bug (valid hash, wrong content) must not load."""
    path = bundle / "dixon_coles.json"
    raw = json.loads(path.read_text())
    raw["attack"] = raw["attack"][:-1]
    path.write_text(json.dumps(raw))
    data = json.loads((bundle / "manifest.json").read_text())
    data["files"]["dixon_coles.json"] = {
        "sha256": artifacts.sha256_file(path),
        "bytes": path.stat().st_size,
    }
    (bundle / "manifest.json").write_text(json.dumps(data))
    with pytest.raises(ArtifactCorruptError, match="attack/defence"):
        read_bundle(bundle)


def test_team_state_round_trips_through_json() -> None:
    state = TeamState()
    state.rating = 1612.5
    state.rating_history.extend([1500.0, 1550.0, 1612.5])
    state.results.extend([(3, 2, 0), (1, 1, 1)])
    state.last_played = dt.date(2025, 3, 1)
    state.n_played = 2
    restored = TeamState.from_dict(json.loads(json.dumps(state.to_dict())))
    assert restored.to_dict() == state.to_dict()
    assert restored.momentum() == state.momentum()
    assert restored.rest_days(dt.date(2025, 3, 11)) == 10.0
