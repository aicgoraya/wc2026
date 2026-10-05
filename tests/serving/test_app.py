import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from wc2026.serving.app import create_app
from wc2026.serving.artifacts import ArtifactNotFoundError
from wc2026.serving.fixture import build_synthetic_bundle
from wc2026.serving.settings import ServingSettings

VALID = {"home_team": "alpha", "away_team": "echo", "date": "2020-01-10"}


def log_lines(capsys: pytest.CaptureFixture[str]) -> list[dict[str, Any]]:
    out = capsys.readouterr().out
    return [json.loads(line) for line in out.splitlines() if line.startswith("{")]


def unloaded_client(settings: ServingSettings, **kwargs: Any) -> TestClient:
    return TestClient(create_app(settings, **kwargs), raise_server_exceptions=False)


# --- liveness vs readiness -------------------------------------------------


def test_ready_reports_the_loaded_artifact(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    resp = client.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["model"]["artifact_version"] == "synthetic-fixture"
    assert body["model"]["synthetic"] is True
    assert body["teams"] == 5
    assert body["evaluation_available"] is True


def test_alive_but_not_ready_when_the_artifact_is_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = ServingSettings(artifact_source="local", artifact_dir=tmp_path / "missing")
    with unloaded_client(settings) as c:
        assert c.get("/health").status_code == 200
        ready = c.get("/ready")
        assert ready.status_code == 503
        assert ready.json() == {"status": "not_ready", "reason": "artifact_not_found"}
        predict = c.post("/predict", json=VALID)
        assert predict.status_code == 503
        assert predict.json()["error"]["code"] == "model_unavailable"
        assert c.get("/models/comparison").status_code == 503
        assert c.get("/teams").status_code == 503
    events = {line["event"]: line for line in log_lines(capsys)}
    assert events["artifact_load_failed"]["reason"] == "artifact_not_found"
    assert events["artifact_load_failed"]["level"] == "ERROR"


def test_corrupt_artifact_never_becomes_ready(bundle: Path) -> None:
    (bundle / "gbm.txt").write_text("garbage")
    with unloaded_client(ServingSettings(artifact_dir=bundle)) as c:
        assert c.get("/ready").json()["reason"] == "artifact_corrupt"
        assert c.get("/health").status_code == 200


def test_pinned_version_mismatch_is_not_ready(master_bundle: Path) -> None:
    settings = ServingSettings(artifact_dir=master_bundle, artifact_version="v-other")
    with unloaded_client(settings) as c:
        assert c.get("/ready").json()["reason"] == "artifact_incompatible"


@pytest.mark.parametrize(
    "settings",
    [
        ServingSettings(artifact_source="local"),
        ServingSettings(artifact_source="s3", artifact_bucket="b"),
        ServingSettings(artifact_source="s3", artifact_version="v1"),
    ],
)
def test_incomplete_configuration_is_not_ready(settings: ServingSettings) -> None:
    with unloaded_client(settings) as c:
        assert c.get("/ready").json()["reason"] == "artifact_config_invalid"


def test_unexpected_loader_crash_is_contained(capsys: pytest.CaptureFixture[str]) -> None:
    def loader(_: ServingSettings) -> Any:
        raise RuntimeError("boom: s3cr3t-internal-detail")

    with unloaded_client(ServingSettings(), loader=loader) as c:
        ready = c.get("/ready")
        assert ready.status_code == 503
        assert "s3cr3t" not in ready.text
        assert ready.json()["reason"] == "artifact_load_error"
    failed = next(line for line in log_lines(capsys) if line["event"] == "artifact_load_failed")
    assert "s3cr3t-internal-detail" in failed["exception"]  # full context stays in server logs


def test_ready_does_not_reload_or_touch_the_store(master_bundle: Path) -> None:
    calls = 0

    def loader(settings: ServingSettings) -> Any:
        nonlocal calls
        calls += 1
        from wc2026.serving.inference import load_artifact

        return load_artifact(master_bundle)

    with unloaded_client(ServingSettings(), loader=loader) as c:
        for _ in range(5):
            assert c.get("/ready").status_code == 200
            assert c.post("/predict", json=VALID).status_code == 200
    assert calls == 1


# --- /predict --------------------------------------------------------------


def test_predict_returns_a_normalised_distribution_and_the_model_version(
    client: TestClient,
) -> None:
    resp = client.post("/predict", json={**VALID, "neutral": False, "competition": "uefa_euro"})
    assert resp.status_code == 200
    body = resp.json()
    probs = body["probabilities"]
    assert set(probs) == {"home_win", "draw", "away_win"}
    assert all(0.0 <= p <= 1.0 for p in probs.values())
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-9)
    for component in ("dixon_coles", "gbm"):
        assert sum(body["components"][component].values()) == pytest.approx(1.0, abs=1e-6)
    assert body["model"] == {
        "artifact_version": "synthetic-fixture",
        "training_cutoff": "2020-01-01",
        "max_prediction_date": "2020-06-29",
        "feature_schema_version": "1",
        "synthetic": True,
    }
    assert body["fixture"] == {
        "home_team": "alpha",
        "away_team": "echo",
        "date": "2020-01-10",
        "neutral": False,
        "competition": "uefa_euro",
    }
    assert body["request_id"] == resp.headers["x-request-id"]


def test_home_advantage_input_changes_the_prediction(client: TestClient) -> None:
    neutral = client.post("/predict", json={**VALID, "neutral": True}).json()
    at_home = client.post("/predict", json={**VALID, "neutral": False}).json()
    assert neutral["probabilities"] != at_home["probabilities"]


def test_display_names_are_normalised_to_team_ids(client: TestClient) -> None:
    resp = client.post("/predict", json={**VALID, "home_team": " Alpha "})
    assert resp.status_code == 200
    assert resp.json()["fixture"]["home_team"] == "alpha"


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({**VALID, "away_team": "alpha"}, "body"),
        ({**VALID, "date": "10/01/2020"}, "date"),
        ({**VALID, "date": "2020-13-45"}, "date"),
        ({**VALID, "neutral": "sometimes"}, "neutral"),
        ({**VALID, "competition": "World Cup!"}, "competition"),
        ({**VALID, "home_team": ""}, "home_team"),
        ({**VALID, "home_team": "!!!"}, "home_team"),
        ({**VALID, "home_team": "x" * 65}, "home_team"),
        ({**VALID, "unexpected": 1}, "unexpected"),
        ({"home_team": "alpha"}, "away_team"),
        ([VALID, VALID], "body"),
    ],
)
def test_malformed_requests_get_422_naming_the_field(
    client: TestClient, payload: Any, field: str
) -> None:
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
    error = resp.json()["error"]
    assert error["code"] == "validation_error"
    assert field in {d["field"] for d in error["details"]}
    assert set(error["details"][0]) == {"field", "issue"}  # the input itself is not echoed


def test_unknown_team_is_a_documented_error(client: TestClient) -> None:
    resp = client.post("/predict", json={**VALID, "away_team": "Brazil"})
    assert resp.status_code == 422
    error = resp.json()["error"]
    assert error["code"] == "unknown_team"
    assert error["details"] == [{"field": "team", "issue": "unknown team id: brazil"}]


@pytest.mark.parametrize("date", ["2019-12-31", "2020-06-30", "2031-01-01"])
def test_dates_outside_the_supported_window_are_rejected(client: TestClient, date: str) -> None:
    resp = client.post("/predict", json={**VALID, "date": date})
    assert resp.status_code == 422
    error = resp.json()["error"]
    assert error["code"] == "unsupported_date"
    assert "2020-01-01" in error["message"] and "2020-06-29" in error["message"]


def test_oversized_bodies_are_rejected_before_parsing(client: TestClient) -> None:
    resp = client.post("/predict", json={**VALID, "home_team": "a" * 20_000})
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "payload_too_large"

    def chunks() -> Any:  # no Content-Length: the stream itself is bounded
        for _ in range(40):
            yield b"x" * 1024

    streamed = client.post(
        "/predict", content=chunks(), headers={"content-type": "application/json"}
    )
    assert streamed.status_code == 413


def test_unknown_route_and_wrong_method_use_the_error_envelope(client: TestClient) -> None:
    missing = client.get("/nope")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "not_found"
    assert client.get("/predict").json()["error"]["code"] == "method_not_allowed"


def test_internal_errors_are_generic_to_the_client_and_detailed_in_logs(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    predictor = client.app.state.service.artifact.predictor  # type: ignore[attr-defined]

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("db password is hunter2")

    monkeypatch.setattr(predictor, "predict", explode)
    capsys.readouterr()
    resp = client.post("/predict", json=VALID)
    assert resp.status_code == 500
    assert resp.json() == {
        "error": {"code": "internal_error", "message": "internal server error"},
        "request_id": resp.headers["x-request-id"],
    }
    assert "hunter2" not in resp.text
    (line,) = [entry for entry in log_lines(capsys) if entry["event"] == "request"]
    assert line["level"] == "ERROR"
    assert line["status"] == 500
    assert line["error_type"] == "RuntimeError"
    assert "hunter2" in line["exception"] and "Traceback" in line["exception"]
    assert line["request_id"] == resp.headers["x-request-id"]


# --- /models/comparison ----------------------------------------------------


def test_comparison_serves_the_stored_evaluation_verbatim(
    client: TestClient, master_bundle: Path
) -> None:
    resp = client.get("/models/comparison")
    assert resp.status_code == 200
    body = resp.json()
    stored = json.loads((master_bundle / "evaluation.json").read_text())
    assert body["evaluation"] == stored
    assert body["model"]["artifact_version"] == "synthetic-fixture"
    assert body["evaluation"]["synthetic"] is True


def test_comparison_is_unavailable_rather_than_invented(tmp_path: Path) -> None:
    bare = tmp_path / "no-eval"
    build_synthetic_bundle(bare, version="no-eval", with_evaluation=False)
    with unloaded_client(ServingSettings(artifact_dir=bare)) as c:
        ready = c.get("/ready").json()
        assert ready["status"] == "ready" and ready["evaluation_available"] is False
        resp = c.get("/models/comparison")
        assert resp.status_code == 503
        assert resp.json()["error"]["code"] == "evaluation_unavailable"
        assert c.post("/predict", json=VALID).status_code == 200


def test_tampered_evaluation_blocks_the_whole_artifact(master_bundle: Path, tmp_path: Path) -> None:
    target = tmp_path / "b"
    shutil.copytree(master_bundle, target)
    doctored = json.loads((target / "evaluation.json").read_text())
    doctored["historical"]["models"][0]["rps"] = 0.01
    (target / "evaluation.json").write_text(json.dumps(doctored))
    with unloaded_client(ServingSettings(artifact_dir=target)) as c:
        assert c.get("/ready").json()["reason"] == "artifact_corrupt"


# --- logging ---------------------------------------------------------------


def test_one_structured_line_per_request_without_the_body(
    client: TestClient, capsys: pytest.CaptureFixture[str]
) -> None:
    capsys.readouterr()
    client.post("/predict", json=VALID)
    client.post("/predict", json={**VALID, "away_team": "zzsecretteamzz"})
    client.get("/health")
    out = capsys.readouterr().out
    lines = [json.loads(line) for line in out.splitlines()]
    requests = [line for line in lines if line["event"] == "request"]
    assert len(requests) == 3
    ok, rejected, health = requests
    assert ok["method"] == "POST" and ok["route"] == "/predict" and ok["status"] == 200
    assert ok["artifact_version"] == "synthetic-fixture"
    assert isinstance(ok["duration_ms"], float) and ok["duration_ms"] >= 0
    assert len(ok["request_id"]) == 32
    assert rejected["status"] == 422 and rejected["error_code"] == "unknown_team"
    assert health["route"] == "/health" and health["status"] == 200
    assert "zzsecretteamzz" not in out  # request bodies never reach the logs
    assert all({"ts", "level", "logger"} <= set(line) for line in lines)


def test_request_id_is_propagated_or_replaced(client: TestClient) -> None:
    kept = client.get("/health", headers={"X-Request-ID": "trace-abc-12345"})
    assert kept.headers["x-request-id"] == "trace-abc-12345"
    replaced = client.get("/health", headers={"X-Request-ID": 'bad id" injected\\'})
    assert replaced.headers["x-request-id"] != 'bad id" injected\\'
    assert len(replaced.headers["x-request-id"]) == 32


def test_startup_log_records_what_was_loaded(
    master_bundle: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with unloaded_client(ServingSettings(artifact_dir=master_bundle)):
        pass
    events = {line["event"]: line for line in log_lines(capsys)}
    assert events["artifact_load_start"]["source"] == "local"
    loaded = events["artifact_loaded"]
    assert loaded["artifact_version"] == "synthetic-fixture"
    assert loaded["synthetic"] is True and loaded["teams"] == 5
    assert "shutdown" in events


# --- docs ------------------------------------------------------------------


def test_openapi_documents_routes_examples_and_errors(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    assert {"/predict", "/models/comparison", "/health", "/ready", "/teams"} <= set(spec["paths"])
    predict = spec["paths"]["/predict"]["post"]
    assert {"200", "413", "422", "503"} <= set(predict["responses"])
    assert spec["components"]["schemas"]["PredictRequest"]["examples"][0]["home_team"] == "brazil"
    assert "ErrorResponse" in spec["components"]["schemas"]
    assert client.get("/docs").status_code == 200


def test_loader_failure_type_maps_to_its_reason_code() -> None:
    def loader(_: ServingSettings) -> Any:
        raise ArtifactNotFoundError("s3://bucket/key does not exist")

    with unloaded_client(ServingSettings(), loader=loader) as c:
        body = c.get("/ready").json()
        assert body == {"status": "not_ready", "reason": "artifact_not_found"}
