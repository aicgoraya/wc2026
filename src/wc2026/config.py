"""Runtime configuration, loaded from the environment / a local ``.env`` file."""

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process-wide settings; secrets stay out of the repo via ``.env`` (gitignored)."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    football_data_token: SecretStr | None = None
    odds_api_key: SecretStr | None = None
    data_root: Path = Path("data")
    default_seed: int = 20260611


def get_settings() -> Settings:
    """Build a fresh ``Settings`` from the current environment."""
    return Settings()
