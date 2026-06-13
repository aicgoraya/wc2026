import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # required by starlette's TestClient
from fastapi.testclient import TestClient

from wc2026.dashboard.app import app

SAMPLE = {
    "generated_utc": "2026-06-13T12:00:00+00:00",
    "blend_weights": {"dixon_coles": 0.67, "gbm": 0.33},
    "upcoming": [
        {
            "match_id": "fd_1",
            "date": "2026-06-14",
            "home": "brazil",
            "away": "morocco",
            "elo": {"home": 0.47, "draw": 0.29, "away": 0.24},
            "dixon_coles": {"home": 0.55, "draw": 0.29, "away": 0.17},
            "gbm": {"home": 0.5, "draw": 0.3, "away": 0.2},
            "blend": {"home": 0.53, "draw": 0.29, "away": 0.18},
            "market": {"home": 0.57, "draw": 0.26, "away": 0.17},
        }
    ],
    "win_cup": [
        {
            "team": "brazil",
            "group": "C",
            "reach_r16": 75.0,
            "reach_qf": 57.0,
            "reach_sf": 40.0,
            "reach_final": 26.0,
            "champion": 15.7,
        }
    ],
    "live_vs_market": {"n": 0, "note": "no completed match has a line yet", "scoreboard": []},
    "model_board": [{"model": "dixon_coles", "rps": 0.1675}, {"model": "gbm", "rps": 0.1705}],
    "calibration": {"ece": 0.0106, "bins": [{"p_pred": 0.5, "p_emp": 0.49, "n": 100}]},
}


def test_api_503_without_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WC2026_DASHBOARD_JSON", str(tmp_path / "missing.json"))
    resp = TestClient(app).get("/api/data")
    assert resp.status_code == 503
    assert "refresh" in resp.json()["error"]


def test_api_serves_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "dashboard.json"
    path.write_text(json.dumps(SAMPLE))
    monkeypatch.setenv("WC2026_DASHBOARD_JSON", str(path))
    resp = TestClient(app).get("/api/data")
    assert resp.status_code == 200
    body = resp.json()
    assert body["win_cup"][0]["team"] == "brazil"
    assert body["blend_weights"]["dixon_coles"] == 0.67


def test_index_serves_html() -> None:
    resp = TestClient(app).get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "World Cup 2026" in resp.text
