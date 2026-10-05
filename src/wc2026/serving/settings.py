"""Serving configuration, read from ``WC2026_*`` environment variables."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class ServingSettings(BaseSettings):
    """Where the model artifact comes from and how the service behaves.

    ``artifact_source="local"`` reads a bundle directory (development, tests,
    CI). ``artifact_source="s3"`` downloads one pinned version from S3 at
    startup using the normal AWS credential chain; nothing here ever holds a
    credential.
    """

    model_config = SettingsConfigDict(env_prefix="WC2026_", extra="ignore")

    artifact_source: Literal["local", "s3"] = "local"
    artifact_dir: Path | None = None
    """Local source: the bundle directory (the one containing ``manifest.json``)."""
    artifact_bucket: str | None = None
    artifact_prefix: str = "artifacts"
    artifact_version: str | None = None
    """Required for S3 (the pinned version). Optional for local, where it is
    checked against the manifest when set."""
    artifact_cache_dir: Path = Path("/tmp/wc2026-artifacts")
    artifact_load_timeout_s: float = 60.0
    """Overall budget for the startup S3 download; exceeded -> not ready."""
    max_body_bytes: int = 16_384
    log_level: str = "INFO"
