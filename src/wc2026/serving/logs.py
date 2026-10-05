"""Structured (JSON-lines) logging and the single request-logging middleware.

Everything goes to stdout, one JSON object per line; on ECS the ``awslogs``
driver ships that stream to CloudWatch unchanged, so CloudWatch Logs Insights
can filter on the fields directly. These are logs, not metrics: latency and
error rates are derived by querying them (see the README).

One log line is written per request by ``RequestLogMiddleware`` - uvicorn's own
access log is disabled by ``wc2026 serve`` so requests are not logged twice.
Request bodies, query strings and headers are never logged.
"""

import contextvars
import datetime as dt
import json
import logging
import re
import sys
import time
import uuid
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

LOGGER_NAME = "wc2026.serving"
logger = logging.getLogger(LOGGER_NAME)

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
AsgiApp = Callable[[Scope, Receive, Send], Awaitable[None]]

_REQUEST_ID_OK = re.compile(r"^[A-Za-z0-9._-]{8,64}$")

# Per-request fields. Handlers add to the dict (e.g. the artifact version); the
# dict object is shared with the worker thread a sync handler runs in.
request_context: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "wc2026_request_context", default=None
)


def current_request_id() -> str | None:
    """The request id of the request being handled, if any."""
    ctx = request_context.get()
    return None if ctx is None else str(ctx.get("request_id"))


def add_log_fields(**fields: Any) -> None:
    """Attach extra fields to the current request's log line."""
    ctx = request_context.get()
    if ctx is not None:
        ctx.update(fields)


class JsonFormatter(logging.Formatter):
    """One JSON object per record: ts, level, logger, event, plus ``fields``."""

    def format(self, record: logging.LogRecord) -> str:
        """Render the record as a single JSON line."""
        payload: dict[str, Any] = {
            "ts": dt.datetime.fromtimestamp(record.created, dt.UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class _StdoutHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """Writes to whatever ``sys.stdout`` is at emit time (so redirection works)."""

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stdout
        super().emit(record)


def configure_logging(level: str = "INFO") -> None:
    """Route the service logger (and uvicorn's error logger) to JSON on stdout."""
    handler = _StdoutHandler()
    handler.setFormatter(JsonFormatter())
    for name in (LOGGER_NAME, "uvicorn.error"):
        target = logging.getLogger(name)
        target.handlers = [handler]
        target.setLevel(level.upper())
        target.propagate = False


def log_event(level: int, event: str, **fields: Any) -> None:
    """Emit one structured event."""
    logger.log(level, event, extra={"fields": fields})


class _BodyTooLargeError(Exception):
    pass


def _json_response(status: int, body: dict[str, Any], request_id: str) -> list[Message]:
    raw = json.dumps(body).encode()
    return [
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(raw)).encode()),
                (b"x-request-id", request_id.encode()),
            ],
        },
        {"type": "http.response.body", "body": raw},
    ]


class RequestLogMiddleware:
    """Request id, body-size bound, one log line per request, last-resort error handling.

    Pure ASGI (not ``BaseHTTPMiddleware``) so it sees the final status code and
    any unhandled exception exactly once. An unhandled exception is logged here
    with its traceback and answered with a generic 500 - the exception text is
    never sent to the client.
    """

    def __init__(self, app: AsgiApp, *, max_body_bytes: int) -> None:
        self._app = app
        self._max_body = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one ASGI connection."""
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        inbound = headers.get("x-request-id", "")
        request_id = inbound if _REQUEST_ID_OK.fullmatch(inbound) else uuid.uuid4().hex
        ctx: dict[str, Any] = {"request_id": request_id}
        token = request_context.set(ctx)
        started = time.perf_counter()
        status = 500
        response_started = False
        received = 0
        overflow = False

        async def limited_receive() -> Message:
            nonlocal received, overflow
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self._max_body:
                    overflow = True
                    raise _BodyTooLargeError
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal status, response_started
            if overflow:
                # the framework turns our receive error into its own 400; drop
                # that and answer 413 ourselves once the app returns
                return
            if message["type"] == "http.response.start":
                status = int(message["status"])
                response_started = True
                message.setdefault("headers", [])
                message["headers"] = [*message["headers"], (b"x-request-id", request_id.encode())]
            await send(message)

        async def reply(code: int, error_code: str, text: str) -> None:
            nonlocal status
            status = code
            if response_started:
                return
            body = {"error": {"code": error_code, "message": text}, "request_id": request_id}
            for message in _json_response(code, body, request_id):
                await send(message)

        error: BaseException | None = None
        try:
            declared = headers.get("content-length", "")
            if declared.isdigit() and int(declared) > self._max_body:
                raise _BodyTooLargeError
            await self._app(scope, limited_receive, tracking_send)
            if overflow:
                raise _BodyTooLargeError
        except _BodyTooLargeError:
            await reply(413, "payload_too_large", f"request body exceeds {self._max_body} bytes")
        except Exception as exc:
            error = exc
            await reply(500, "internal_error", "internal server error")
        finally:
            route = scope.get("route")
            fields = {
                "method": scope.get("method"),
                "route": getattr(route, "path", None) or "unmatched",
                "status": status,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                **ctx,
            }
            if error is not None:
                fields["error_type"] = error.__class__.__name__
                logger.error(
                    "request",
                    extra={"fields": fields},
                    exc_info=(type(error), error, error.__traceback__),
                )
            else:
                level = logging.ERROR if status >= 500 else logging.INFO
                logger.log(level, "request", extra={"fields": fields})
            request_context.reset(token)
