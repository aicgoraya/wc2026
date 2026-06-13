"""FastAPI app serving the pre-built dashboard payload + a static page.

The payload is computed by ``wc2026 refresh`` (``dashboard/data.py``) and cached
to JSON; this app just serves it, so requests are instant. The JSON path is
configurable via ``WC2026_DASHBOARD_JSON`` (default ``results/dashboard.json``)
so the served data and tests can point at different files.
"""

import json
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

_STATIC = Path(__file__).parent / "static"

app = FastAPI(title="WC2026 Probabilistic Forecaster", docs_url="/api/docs")


def _payload_path() -> Path:
    return Path(os.environ.get("WC2026_DASHBOARD_JSON", "results/dashboard.json"))


@app.get("/api/data")
def data() -> JSONResponse:
    """The dashboard payload, or a 503 telling the user to run a refresh."""
    path = _payload_path()
    if not path.exists():
        return JSONResponse(
            {"error": "dashboard not built yet — run `wc2026 refresh`"}, status_code=503
        )
    return JSONResponse(json.loads(path.read_text()))


@app.get("/")
def index() -> FileResponse:
    """The single-page dashboard."""
    return FileResponse(_STATIC / "index.html")
