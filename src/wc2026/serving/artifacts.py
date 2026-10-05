"""The model-artifact contract: what a saved model bundle is and how it is checked.

A bundle is one directory (locally) or one S3 prefix holding::

    manifest.json     the contract: versions, training cutoff, file hashes
    dixon_coles.json  fitted Dixon-Coles parameters
    gbm.txt           the LightGBM booster in LightGBM's plain-text model format
    team_state.json   per-team feature state as of the training cutoff
    evaluation.json   (optional) stored walk-forward evaluation results

Nothing in a bundle is pickled: every file is JSON or LightGBM's own text
format, so loading a bundle cannot execute code. A bundle is still only as
trustworthy as its origin - the SHA-256 hashes in the manifest detect
corruption and partial uploads, not a malicious publisher - so only load
bundles produced by ``wc2026 export-artifact`` from a bucket you control.

Bundles are immutable and addressed by ``artifact_version``; the service is
pinned to one version and never picks "the latest upload".
"""

import dataclasses
import datetime as dt
import hashlib
import json
import platform
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from wc2026.features.build import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, TeamState
from wc2026.models.dixon_coles import DCParams

MANIFEST_NAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 1
DC_FILE = "dixon_coles.json"
GBM_FILE = "gbm.txt"
STATE_FILE = "team_state.json"
EVALUATION_FILE = "evaluation.json"
REQUIRED_FILES: tuple[str, ...] = (DC_FILE, GBM_FILE, STATE_FILE)

VERSION_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
_FILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ArtifactError(Exception):
    """Base class; ``code`` is a stable, non-sensitive identifier safe to expose."""

    code = "artifact_error"


class ArtifactNotFoundError(ArtifactError):
    """The bundle, or a file its manifest requires, does not exist."""

    code = "artifact_not_found"


class ArtifactCorruptError(ArtifactError):
    """A file is unreadable, malformed, or does not match its manifest hash."""

    code = "artifact_corrupt"


class ArtifactIncompatibleError(ArtifactError):
    """The bundle is intact but was built for a different schema, version or runtime."""

    code = "artifact_incompatible"


class ArtifactUnavailableError(ArtifactError):
    """The artifact store could not be reached or refused access."""

    code = "artifact_store_unavailable"


class ArtifactConfigError(ArtifactError):
    """The service's artifact settings are incomplete or contradictory."""

    code = "artifact_config_invalid"


class FileEntry(BaseModel):
    """Integrity record for one bundle file."""

    model_config = ConfigDict(extra="forbid")

    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0)


class Manifest(BaseModel):
    """``manifest.json`` - the artifact contract."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    artifact_version: str = Field(pattern=VERSION_PATTERN)
    created_utc: dt.datetime
    synthetic: bool
    """True for deterministic test fixtures. The API repeats this flag on every
    prediction so fixture output can never pass as real model output."""
    training_cutoff: dt.date
    """Models and team state use only matches strictly before this date."""
    max_prediction_date: dt.date
    """Last fixture date the service will predict with this artifact."""
    feature_schema_version: str
    feature_columns: list[str]
    blend_weights: dict[str, float]
    dixon_coles: dict[str, float]
    gbm_params: dict[str, Any]
    training_data: dict[str, Any]
    runtime: dict[str, str]
    files: dict[str, FileEntry]

    @field_validator("files")
    @classmethod
    def _safe_names(cls, files: dict[str, FileEntry]) -> dict[str, FileEntry]:
        for name in files:
            if not _FILE_NAME.fullmatch(name) or name == MANIFEST_NAME:
                raise ValueError(f"unsafe or reserved file name in manifest: {name!r}")
        return files

    @field_validator("blend_weights")
    @classmethod
    def _valid_weights(cls, weights: dict[str, float]) -> dict[str, float]:
        if set(weights) != {"dixon_coles", "gbm"}:
            raise ValueError("blend_weights must have exactly the keys dixon_coles and gbm")
        if any(not np.isfinite(w) or w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
            raise ValueError("blend_weights must be finite, non-negative and not all zero")
        return weights


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file, streamed."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_versions() -> dict[str, str]:
    """Versions of the libraries that read/write the model files."""
    import lightgbm
    import scipy

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "lightgbm": lightgbm.__version__,
    }


def dc_params_to_json(params: DCParams) -> dict[str, Any]:
    """Dixon-Coles parameters as a JSON-safe mapping."""
    return {
        "teams": list(params.teams),
        "intercept": params.intercept,
        "home_adv": params.home_adv,
        "neutral_adv": params.neutral_adv,
        "rho": params.rho,
        "attack": [float(v) for v in params.attack],
        "defence": [float(v) for v in params.defence],
    }


def dc_params_from_json(raw: Mapping[str, Any]) -> DCParams:
    """Inverse of ``dc_params_to_json``; raises ``ArtifactCorruptError`` if malformed."""
    try:
        teams = tuple(str(t) for t in raw["teams"])
        attack = np.asarray(raw["attack"], dtype=np.float64)
        defence = np.asarray(raw["defence"], dtype=np.float64)
        scalars = [float(raw[k]) for k in ("intercept", "home_adv", "neutral_adv", "rho")]
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactCorruptError(f"{DC_FILE}: malformed parameters ({exc})") from exc
    if not teams or len(set(teams)) != len(teams):
        raise ArtifactCorruptError(f"{DC_FILE}: team list is empty or has duplicates")
    if attack.shape != (len(teams),) or defence.shape != (len(teams),):
        raise ArtifactCorruptError(f"{DC_FILE}: attack/defence do not match the team list")
    if not (
        np.isfinite(attack).all() and np.isfinite(defence).all() and np.isfinite(scalars).all()
    ):
        raise ArtifactCorruptError(f"{DC_FILE}: non-finite parameter")
    return DCParams(teams, scalars[0], scalars[1], scalars[2], scalars[3], attack, defence)


def write_bundle(
    bundle_dir: Path,
    *,
    version: str,
    synthetic: bool,
    training_cutoff: dt.date,
    max_prediction_date: dt.date,
    dc_params: DCParams,
    dc_hyperparams: Mapping[str, float],
    gbm_model_string: str,
    gbm_params: Mapping[str, Any],
    team_states: Mapping[str, TeamState],
    blend_weights: Mapping[str, float],
    training_data: Mapping[str, Any],
    evaluation: Mapping[str, Any] | None = None,
    created_utc: dt.datetime | None = None,
) -> Manifest:
    """Write one immutable bundle directory and return its manifest.

    Refuses to write into an existing directory: a version is written once.
    """
    if not re.fullmatch(VERSION_PATTERN, version):
        raise ValueError(f"invalid artifact version {version!r} (allowed: {VERSION_PATTERN})")
    if max_prediction_date < training_cutoff:
        raise ValueError("max_prediction_date is before training_cutoff")
    bundle_dir.mkdir(parents=True, exist_ok=False)

    (bundle_dir / DC_FILE).write_text(json.dumps(dc_params_to_json(dc_params)))
    (bundle_dir / GBM_FILE).write_text(gbm_model_string)
    states = {team: state.to_dict() for team, state in sorted(team_states.items())}
    (bundle_dir / STATE_FILE).write_text(json.dumps(states))
    names = list(REQUIRED_FILES)
    if evaluation is not None:
        (bundle_dir / EVALUATION_FILE).write_text(json.dumps(evaluation, indent=2))
        names.append(EVALUATION_FILE)

    manifest = Manifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        artifact_version=version,
        created_utc=created_utc or dt.datetime.now(dt.UTC),
        synthetic=synthetic,
        training_cutoff=training_cutoff,
        max_prediction_date=max_prediction_date,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_columns=list(FEATURE_COLUMNS),
        blend_weights=dict(blend_weights),
        dixon_coles=dict(dc_hyperparams),
        gbm_params=dict(gbm_params),
        training_data=dict(training_data),
        runtime=runtime_versions(),
        files={
            name: FileEntry(
                sha256=sha256_file(bundle_dir / name), bytes=(bundle_dir / name).stat().st_size
            )
            for name in names
        },
    )
    (bundle_dir / MANIFEST_NAME).write_text(manifest.model_dump_json(indent=2))
    return manifest


def parse_manifest(raw: bytes | str) -> Manifest:
    """Parse and schema-check manifest content."""
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ArtifactCorruptError(f"{MANIFEST_NAME}: not valid JSON") from exc
    if isinstance(data, dict) and data.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ArtifactIncompatibleError(
            f"{MANIFEST_NAME}: schema_version {data.get('schema_version')!r} is not supported"
            f" (this build reads {MANIFEST_SCHEMA_VERSION})"
        )
    try:
        return Manifest.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:5]
        )
        raise ArtifactCorruptError(f"{MANIFEST_NAME}: invalid ({problems})") from exc


def validate_bundle(bundle_dir: Path, *, expected_version: str | None = None) -> Manifest:
    """Check a bundle directory against the contract without loading the models.

    Verifies the manifest schema, the pinned version, feature-schema and runtime
    compatibility, and the presence, size and SHA-256 of every listed file.
    """
    manifest_path = bundle_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ArtifactNotFoundError(f"no {MANIFEST_NAME} in {bundle_dir}")
    manifest = parse_manifest(manifest_path.read_bytes())
    check_compatible(manifest, expected_version=expected_version)
    for name, entry in manifest.files.items():
        path = bundle_dir / name
        if not path.is_file():
            raise ArtifactNotFoundError(f"{name} is listed in the manifest but missing")
        if path.stat().st_size != entry.bytes or sha256_file(path) != entry.sha256:
            raise ArtifactCorruptError(f"{name} does not match its manifest size/SHA-256")
    return manifest


def check_compatible(manifest: Manifest, *, expected_version: str | None = None) -> None:
    """Raise ``ArtifactIncompatibleError`` unless this build can serve the manifest."""
    if expected_version is not None and manifest.artifact_version != expected_version:
        raise ArtifactIncompatibleError(
            f"pinned artifact version is {expected_version!r} but the bundle is"
            f" {manifest.artifact_version!r}"
        )
    missing = [name for name in REQUIRED_FILES if name not in manifest.files]
    if missing:
        raise ArtifactIncompatibleError(f"manifest does not list required files: {missing}")
    if manifest.feature_schema_version != FEATURE_SCHEMA_VERSION or tuple(
        manifest.feature_columns
    ) != tuple(FEATURE_COLUMNS):
        raise ArtifactIncompatibleError(
            f"feature schema mismatch: bundle has version {manifest.feature_schema_version!r}"
            f" with {len(manifest.feature_columns)} columns, this build expects version"
            f" {FEATURE_SCHEMA_VERSION!r}"
        )
    built_with = manifest.runtime.get("lightgbm", "")
    running = runtime_versions()["lightgbm"]
    if built_with.split(".")[0] != running.split(".")[0]:
        raise ArtifactIncompatibleError(
            f"bundle was built with lightgbm {built_with or 'unknown'}, runtime has {running}"
        )


@dataclasses.dataclass(frozen=True)
class RawBundle:
    """A validated bundle's parsed contents, before model objects are built."""

    manifest: Manifest
    dc_params: DCParams
    gbm_model_string: str
    team_states: dict[str, TeamState]
    evaluation: dict[str, Any] | None


def read_bundle(bundle_dir: Path, *, expected_version: str | None = None) -> RawBundle:
    """Validate a bundle directory and parse its files."""
    manifest = validate_bundle(bundle_dir, expected_version=expected_version)
    try:
        dc_raw = json.loads((bundle_dir / DC_FILE).read_text())
        states_raw = json.loads((bundle_dir / STATE_FILE).read_text())
        states = {str(team): TeamState.from_dict(raw) for team, raw in states_raw.items()}
        evaluation = (
            json.loads((bundle_dir / EVALUATION_FILE).read_text())
            if EVALUATION_FILE in manifest.files
            else None
        )
    except (ValueError, KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ArtifactCorruptError(f"bundle file is malformed: {exc.__class__.__name__}") from exc
    if evaluation is not None and not isinstance(evaluation, dict):
        raise ArtifactCorruptError(f"{EVALUATION_FILE}: expected a JSON object")
    return RawBundle(
        manifest=manifest,
        dc_params=dc_params_from_json(dc_raw),
        gbm_model_string=(bundle_dir / GBM_FILE).read_text(),
        team_states=states,
        evaluation=evaluation,
    )
