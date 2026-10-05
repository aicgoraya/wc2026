"""S3 transport for artifact bundles: pinned-version download, validated upload.

Layout: ``s3://<bucket>/<prefix>/<artifact_version>/<file>``. Credentials come
from the standard AWS provider chain (task role on ECS, profile/SSO/env vars
locally); this module never reads or stores a credential.
"""

import os
import time
from pathlib import Path
from typing import Any

from wc2026.serving.artifacts import (
    MANIFEST_NAME,
    ArtifactError,
    ArtifactNotFoundError,
    ArtifactUnavailableError,
    Manifest,
    check_compatible,
    parse_manifest,
    sha256_file,
    validate_bundle,
)

CONNECT_TIMEOUT_S = 3
READ_TIMEOUT_S = 10
MAX_ATTEMPTS = 3  # total tries per request, including the first


def make_client() -> Any:
    """An S3 client with bounded timeouts and retries (botocore 'standard' mode)."""
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        config=Config(
            connect_timeout=CONNECT_TIMEOUT_S,
            read_timeout=READ_TIMEOUT_S,
            retries={"total_max_attempts": MAX_ATTEMPTS, "mode": "standard"},
        ),
    )


def _key(prefix: str, version: str, name: str) -> str:
    return "/".join(part for part in (prefix.strip("/"), version, name) if part)


def _translate(exc: Exception, bucket: str, key: str) -> ArtifactError:
    """Map a boto error to an artifact error with a useful, credential-free message."""
    from botocore.exceptions import ClientError

    where = f"s3://{bucket}/{key}"
    if isinstance(exc, ClientError):
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound", "NoSuchBucket"}:
            return ArtifactNotFoundError(f"{where} does not exist ({code})")
        if code in {"403", "AccessDenied", "AllAccessDisabled"}:
            return ArtifactUnavailableError(
                f"access denied reading {where}; check the task role's S3 permissions"
            )
        return ArtifactUnavailableError(f"S3 error {code or 'unknown'} for {where}")
    return ArtifactUnavailableError(f"could not reach S3 for {where}: {exc.__class__.__name__}")


def _get_bytes(client: Any, bucket: str, key: str) -> bytes:
    from botocore.exceptions import BotoCoreError, ClientError

    try:
        body: bytes = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    except (BotoCoreError, ClientError) as exc:
        raise _translate(exc, bucket, key) from exc
    return body


def download_bundle(
    client: Any,
    *,
    bucket: str,
    prefix: str,
    version: str,
    cache_root: Path,
    timeout_s: float,
) -> Path:
    """Fetch one pinned bundle version into ``cache_root/<version>`` and validate it.

    The manifest is fetched first and decides which files to fetch. Files
    already cached with the right SHA-256 are reused. The overall wall-clock
    budget is ``timeout_s`` (checked between files; a single in-flight request
    is additionally bounded by the client's own timeouts and retries).
    """
    deadline = time.monotonic() + timeout_s
    manifest_bytes = _get_bytes(client, bucket, _key(prefix, version, MANIFEST_NAME))
    manifest = parse_manifest(manifest_bytes)
    check_compatible(manifest, expected_version=version)

    target = cache_root / version
    target.mkdir(parents=True, exist_ok=True)
    for name, entry in manifest.files.items():
        path = target / name
        if path.is_file() and sha256_file(path) == entry.sha256:
            continue
        if time.monotonic() > deadline:
            raise ArtifactUnavailableError(
                f"artifact download exceeded its {timeout_s:.0f}s budget before {name}"
            )
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(_get_bytes(client, bucket, _key(prefix, version, name)))
        os.replace(tmp, path)
    (target / MANIFEST_NAME).write_bytes(manifest_bytes)
    validate_bundle(target, expected_version=version)
    return target


def upload_bundle(
    client: Any, bundle_dir: Path, *, bucket: str, prefix: str
) -> tuple[Manifest, str]:
    """Validate a local bundle and upload it to its versioned prefix.

    Refuses to overwrite an existing version. Model files go first and the
    manifest last, so an interrupted upload leaves no loadable bundle behind.
    Returns the manifest and the ``s3://`` URI of the version prefix.
    """
    from botocore.exceptions import BotoCoreError, ClientError

    manifest = validate_bundle(bundle_dir)
    version = manifest.artifact_version
    manifest_key = _key(prefix, version, MANIFEST_NAME)
    try:
        client.head_object(Bucket=bucket, Key=manifest_key)
    except ClientError as exc:
        if str(exc.response.get("Error", {}).get("Code", "")) not in {
            "404",
            "NoSuchKey",
            "NotFound",
        }:
            raise _translate(exc, bucket, manifest_key) from exc
    except BotoCoreError as exc:
        raise _translate(exc, bucket, manifest_key) from exc
    else:
        raise ArtifactError(
            f"s3://{bucket}/{_key(prefix, version, '')} already holds version {version!r};"
            " versions are immutable - export under a new version instead"
        )

    for name in [*manifest.files, MANIFEST_NAME]:
        key = _key(prefix, version, name)
        try:
            client.put_object(
                Bucket=bucket,
                Key=key,
                Body=(bundle_dir / name).read_bytes(),
                ServerSideEncryption="AES256",
            )
        except (BotoCoreError, ClientError) as exc:
            raise _translate(exc, bucket, key) from exc
    return manifest, f"s3://{bucket}/{_key(prefix, version, '')}"
