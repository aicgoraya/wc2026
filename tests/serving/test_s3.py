import io
import json
from pathlib import Path
from typing import Any

import pytest
from botocore.stub import Stubber
from fastapi.testclient import TestClient

from wc2026.serving import s3
from wc2026.serving.app import create_app
from wc2026.serving.artifacts import (
    ArtifactCorruptError,
    ArtifactError,
    ArtifactIncompatibleError,
    ArtifactNotFoundError,
    ArtifactUnavailableError,
)
from wc2026.serving.inference import load_artifact
from wc2026.serving.settings import ServingSettings


def s3_client() -> Any:
    """A real boto3 S3 client with dummy credentials; only ever used with a Stubber."""
    import boto3

    return boto3.client(
        "s3",
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )


def body(data: bytes) -> Any:
    from botocore.response import StreamingBody

    return StreamingBody(io.BytesIO(data), len(data))


BUCKET = "test-bucket"
VERSION = "synthetic-fixture"
PREFIX = "artifacts"


def key(name: str) -> str:
    return f"{PREFIX}/{VERSION}/{name}"


def file_names(bundle: Path) -> list[str]:
    return list(json.loads((bundle / "manifest.json").read_text())["files"])


def stub_download(stub: Stubber, bundle: Path, *, replace: dict[str, bytes] | None = None) -> None:
    """Queue the exact GetObject calls a full download must make, in order."""
    replace = replace or {}
    for name in ["manifest.json", *file_names(bundle)]:
        data = replace.get(name, (bundle / name).read_bytes())
        stub.add_response("get_object", {"Body": body(data)}, {"Bucket": BUCKET, "Key": key(name)})


def download(client: Any, cache: Path, timeout_s: float = 30.0) -> Path:
    return s3.download_bundle(
        client, bucket=BUCKET, prefix=PREFIX, version=VERSION, cache_root=cache, timeout_s=timeout_s
    )


def test_download_fetches_the_pinned_version_and_it_loads(
    master_bundle: Path, tmp_path: Path
) -> None:
    client = s3_client()
    with Stubber(client) as stub:
        stub_download(stub, master_bundle)
        local = download(client, tmp_path)
        stub.assert_no_pending_responses()
    assert local == tmp_path / VERSION
    artifact = load_artifact(local, expected_version=VERSION)
    assert artifact.manifest.artifact_version == VERSION
    assert not list(local.glob("*.part"))


def test_second_download_reuses_verified_cached_files(master_bundle: Path, tmp_path: Path) -> None:
    client = s3_client()
    with Stubber(client) as stub:
        stub_download(stub, master_bundle)
        download(client, tmp_path)
        # only the manifest may be fetched again; any other call would fail the stub
        stub.add_response(
            "get_object",
            {"Body": body((master_bundle / "manifest.json").read_bytes())},
            {"Bucket": BUCKET, "Key": key("manifest.json")},
        )
        download(client, tmp_path)
        stub.assert_no_pending_responses()


def test_missing_version_is_not_found(tmp_path: Path) -> None:
    client = s3_client()
    with Stubber(client) as stub:
        stub.add_client_error("get_object", service_error_code="NoSuchKey", http_status_code=404)
        with pytest.raises(ArtifactNotFoundError, match=f"s3://{BUCKET}/{PREFIX}/{VERSION}"):
            download(client, tmp_path)


def test_access_denied_points_at_permissions(tmp_path: Path) -> None:
    client = s3_client()
    with Stubber(client) as stub:
        stub.add_client_error("get_object", service_error_code="AccessDenied", http_status_code=403)
        with pytest.raises(ArtifactUnavailableError, match="task role"):
            download(client, tmp_path)


def test_other_s3_failures_are_unavailable_not_crashes(tmp_path: Path) -> None:
    client = s3_client()
    with Stubber(client) as stub:
        stub.add_client_error("get_object", service_error_code="SlowDown", http_status_code=503)
        with pytest.raises(ArtifactUnavailableError, match="SlowDown"):
            download(client, tmp_path)


def test_corrupted_object_fails_the_hash_check(master_bundle: Path, tmp_path: Path) -> None:
    client = s3_client()
    with Stubber(client) as stub:
        stub_download(stub, master_bundle, replace={"gbm.txt": b"truncated upload"})
        with pytest.raises(ArtifactCorruptError, match=r"gbm\.txt"):
            download(client, tmp_path)


def test_prefix_holding_a_different_version_is_refused(master_bundle: Path, tmp_path: Path) -> None:
    manifest = json.loads((master_bundle / "manifest.json").read_text())
    manifest["artifact_version"] = "someone-overwrote-this"
    client = s3_client()
    with Stubber(client) as stub:
        stub.add_response(
            "get_object",
            {"Body": body(json.dumps(manifest).encode())},
            {"Bucket": BUCKET, "Key": key("manifest.json")},
        )
        with pytest.raises(ArtifactIncompatibleError, match="pinned"):
            download(client, tmp_path)
        stub.assert_no_pending_responses()  # no model file was fetched


def test_download_budget_is_enforced(master_bundle: Path, tmp_path: Path) -> None:
    client = s3_client()
    with Stubber(client) as stub:
        stub.add_response(
            "get_object",
            {"Body": body((master_bundle / "manifest.json").read_bytes())},
            {"Bucket": BUCKET, "Key": key("manifest.json")},
        )
        with pytest.raises(ArtifactUnavailableError, match="budget"):
            download(client, tmp_path, timeout_s=-1.0)


def test_client_has_bounded_timeouts_and_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    config = s3.make_client().meta.config
    assert config.connect_timeout == s3.CONNECT_TIMEOUT_S
    assert config.read_timeout == s3.READ_TIMEOUT_S
    assert config.retries == {"total_max_attempts": s3.MAX_ATTEMPTS, "mode": "standard"}


def test_service_starts_from_s3_and_becomes_ready(
    master_bundle: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = s3_client()
    monkeypatch.setattr(s3, "make_client", lambda: client)
    settings = ServingSettings(
        artifact_source="s3",
        artifact_bucket=BUCKET,
        artifact_prefix=PREFIX,
        artifact_version=VERSION,
        artifact_cache_dir=tmp_path,
    )
    with Stubber(client) as stub:
        stub_download(stub, master_bundle)
        with TestClient(create_app(settings)) as http:
            assert http.get("/ready").json()["model"]["artifact_version"] == VERSION
            for _ in range(3):  # requests never go back to S3 (the stub has nothing queued)
                resp = http.post(
                    "/predict",
                    json={"home_team": "alpha", "away_team": "echo", "date": "2020-01-10"},
                )
                assert resp.status_code == 200
        stub.assert_no_pending_responses()


def test_service_is_not_ready_when_s3_denies_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = s3_client()
    monkeypatch.setattr(s3, "make_client", lambda: client)
    settings = ServingSettings(
        artifact_source="s3",
        artifact_bucket=BUCKET,
        artifact_version=VERSION,
        artifact_cache_dir=tmp_path,
    )
    with Stubber(client) as stub:
        stub.add_client_error("get_object", service_error_code="AccessDenied", http_status_code=403)
        with TestClient(create_app(settings)) as http:
            assert http.get("/health").status_code == 200
            ready = http.get("/ready")
            assert ready.status_code == 503
            assert ready.json()["reason"] == "artifact_store_unavailable"


def test_upload_writes_model_files_first_and_the_manifest_last(master_bundle: Path) -> None:
    client = s3_client()
    with Stubber(client) as stub:
        stub.add_client_error(
            "head_object",
            service_error_code="404",
            http_status_code=404,
            expected_params={"Bucket": BUCKET, "Key": key("manifest.json")},
        )
        for name in [*file_names(master_bundle), "manifest.json"]:
            stub.add_response(
                "put_object",
                {},
                {
                    "Bucket": BUCKET,
                    "Key": key(name),
                    "Body": (master_bundle / name).read_bytes(),
                    "ServerSideEncryption": "AES256",
                },
            )
        manifest, uri = s3.upload_bundle(client, master_bundle, bucket=BUCKET, prefix=PREFIX)
        stub.assert_no_pending_responses()
    assert manifest.artifact_version == VERSION
    assert uri == f"s3://{BUCKET}/{PREFIX}/{VERSION}"


def test_upload_never_overwrites_an_existing_version(master_bundle: Path) -> None:
    client = s3_client()
    with Stubber(client) as stub:
        stub.add_response("head_object", {}, {"Bucket": BUCKET, "Key": key("manifest.json")})
        with pytest.raises(ArtifactError, match="immutable"):
            s3.upload_bundle(client, master_bundle, bucket=BUCKET, prefix=PREFIX)
        stub.assert_no_pending_responses()  # nothing was put


def test_upload_validates_before_any_network_call(bundle: Path) -> None:
    (bundle / "gbm.txt").write_text("tampered")
    client = s3_client()
    with Stubber(client) as stub:  # no responses queued: any call would raise
        with pytest.raises(ArtifactCorruptError):
            s3.upload_bundle(client, bundle, bucket=BUCKET, prefix=PREFIX)
        stub.assert_no_pending_responses()
