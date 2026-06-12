from pathlib import Path

import pytest
from typer.testing import CliRunner

from wc2026.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def no_ambient_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    monkeypatch.delenv("FOOTBALL_DATA_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)  # no .env in cwd


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("snapshot-odds", "snapshot-fixtures", "refresh", "simulate"):
        assert command in result.output


def test_snapshot_odds_without_key_exits_2() -> None:
    result = runner.invoke(app, ["snapshot-odds"])
    assert result.exit_code == 2
    assert "ODDS_API_KEY" in result.output


def test_snapshot_fixtures_without_token_exits_2() -> None:
    result = runner.invoke(app, ["snapshot-fixtures"])
    assert result.exit_code == 2
    assert "FOOTBALL_DATA_TOKEN" in result.output


def test_unimplemented_command_exits_1() -> None:
    result = runner.invoke(app, ["simulate"])
    assert result.exit_code == 1
