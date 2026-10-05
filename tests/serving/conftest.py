import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("lightgbm")
pytest.importorskip("boto3")

from fastapi.testclient import TestClient

from wc2026.serving.app import create_app
from wc2026.serving.fixture import build_synthetic_bundle
from wc2026.serving.settings import ServingSettings


@pytest.fixture(scope="session")
def master_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The synthetic bundle, built once (read-only: copy it to mutate)."""
    path = tmp_path_factory.mktemp("artifact") / "synthetic-fixture"
    build_synthetic_bundle(path)
    return path


@pytest.fixture
def bundle(master_bundle: Path, tmp_path: Path) -> Path:
    """A private, mutable copy of the synthetic bundle."""
    target = tmp_path / "synthetic-fixture"
    shutil.copytree(master_bundle, target)
    return target


@pytest.fixture
def client(master_bundle: Path) -> Iterator[TestClient]:
    """A started service loaded from the local synthetic bundle."""
    app = create_app(ServingSettings(artifact_source="local", artifact_dir=master_bundle))
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
