"""Local web interface: Python's standard-library HTTP server on 127.0.0.1, no external resources.

Security: binds to the loopback interface only; rejects requests whose Host header is not this machine
(DNS-rebinding protection) and cross-site POSTs (Origin check). The page loads nothing from the internet.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from src.chatbot.chunks import CHUNKS
from src.chatbot.engine import ChatEngine, SessionStore
from src.chatbot.llm import Cancelled

log = logging.getLogger("imsdb.chatbot.server")

STATIC_DIR = Path(__file__).parent / "static"
MAX_BODY = 16_384
MAX_QUESTION = 2_000
ATTRIBUTION = ("Screenplay text retrieved from The Internet Movie Script Database (IMSDb), https://imsdb.com. "
               "All rights belong to the respective copyright holders (studios and writers).")


class EngineHolder:
    """Loads the models in the background so the page is available immediately.

    Attributes:
        engine: The engine once loaded, else None.
        error: Why loading failed (``"ExcType: message"``), else None.
        ready: Set when loading finished, successfully or not.
        started: Time the holder was created (Unix seconds).
    """

    def __init__(self, factory: Callable[[], ChatEngine]) -> None:
        """Create the holder; nothing is loaded until ``start``.

        Args:
            factory: Builds the chat engine (loads the catalogue and models; slow).
        """
        self._factory = factory
        self.engine: ChatEngine | None = None
        self.error: str | None = None
        self.ready = threading.Event()
        self.started = time.time()

    def start(self) -> None:
        """Start loading in a daemon thread named ``model-loader``; returns immediately."""
        threading.Thread(target=self._load, name="model-loader", daemon=True).start()

    def _load(self) -> None:
        """Build the engine and warm its prompt caches; record any failure in ``error``. Always sets ``ready``."""
        try:
            engine = self._factory()
            warm = getattr(engine, "warm_up", None)
            if warm:
                warm()
            self.engine = engine
        except Exception as exc:  # noqa: BLE001 - surfaced to the user through /api/health
            log.exception("model loading failed")
            self.error = f"{exc.__class__.__name__}: {exc}"
        finally:
            self.ready.set()

    def status(self) -> dict[str, Any]:
        """Loading state for ``/api/health``.

        Returns:
            ``{"state": "error", "error"}``, ``{"state": "loading", "seconds"}`` (since start) or
            ``{"state": "ready", "models", "films"}`` (model load info and catalogue size).
        """
        if self.error:
            return {"state": "error", "error": self.error}
        if self.engine is None:
            return {"state": "loading", "seconds": round(time.time() - self.started)}
        return {"state": "ready", "models": self.engine.models.info,
                "films": len(self.engine.catalog.films)}


class ChatHTTPServer(ThreadingHTTPServer):
    """Threaded HTTP server (one daemon thread per request) that carries the engine holder and sessions."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], holder: EngineHolder, sessions: SessionStore) -> None:
        """Bind the server and attach shared state for ``ChatHandler``.

        Args:
            address: ``(host, port)`` to bind.
            holder: The background-loaded engine.
            sessions: The chat session store.
        """
        super().__init__(address, ChatHandler)
        self.holder = holder
        self.sessions = sessions

    def allowed_hosts(self) -> set[str]:
        """Host header values accepted (DNS-rebinding protection), also used for the Origin check.

        Returns:
            ``127.0.0.1:<port>``, ``localhost:<port>`` and ``[::1]:<port>`` for the bound port.
        """
        port = self.server_address[1]
        return {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}


class ChatHandler(BaseHTTPRequestHandler):
    """Request handler: serves the page, ``/api/health``, ``/api/passage``, ``/api/reset`` and ``/api/chat`` (SSE)."""

    server: ChatHTTPServer
    protocol_version = "HTTP/1.1"
    server_version = "ScriptBot/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # route through logging, not stderr
        """Log an access line at debug level instead of writing it to stderr.

        Args:
            fmt: printf-style format from ``http.server``.
            *args: Values for the format.
        """
        log.debug("%s - %s", self.address_string(), fmt % args)

    # ------------------------------------------------------------------ helpers
    def _host_ok(self) -> bool:
        """Check the Host header against the server's loopback names.

        Returns:
            True when the Host header is allowed.
        """
        return self.headers.get("Host", "") in self.server.allowed_hosts()

    def _origin_ok(self) -> bool:
        """Reject cross-site requests: the Origin header, if sent, must be this server over ``http://``.

        Returns:
            True when there is no Origin header or it is one of the allowed hosts.
        """
        origin = self.headers.get("Origin")
        return origin is None or origin.removeprefix("http://") in self.server.allowed_hosts()

    def _send(self, status: int, body: bytes, content_type: str, extra: dict[str, str] | None = None) -> None:
        """Write a complete response with no-store, nosniff and no-referrer headers.

        Args:
            status: HTTP status code.
            body: Response body.
            content_type: Content-Type header value.
            extra: Additional headers, or None.
        """
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data: Any) -> None:
        """Send a UTF-8 JSON response (non-ASCII kept; unknown types converted with ``str``).

        Args:
            status: HTTP status code.
            data: JSON-serializable value.
        """
        self._send(status, json.dumps(data, ensure_ascii=False, default=str).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _read_json(self) -> dict[str, Any] | None:
        """Read and parse the request body as a JSON object.

        Returns:
            The parsed object, or None when Content-Length is missing, invalid, 0 or above ``MAX_BODY``, the
            body is not valid UTF-8 JSON, or it is not an object.
        """
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return None
        if length <= 0 or length > MAX_BODY:
            return None
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    # ------------------------------------------------------------------ routes
    def do_GET(self) -> None:  # noqa: N802 - http.server API
        """Handle GET requests (after the Host check, else 403).

        Routes: ``/`` and ``/index.html`` serve the page with a strict Content-Security-Policy;
        ``/api/health`` returns the loading state plus the IMSDb attribution; ``/api/passage?id=<chunk_id>``
        returns one ``chat_chunks`` document (``text``, ``film``, ``year``, ``scene_number``, ``heading``), 503
        while loading, 404 if unknown. Anything else is 404.
        """
        if not self._host_ok():
            return self._json(HTTPStatus.FORBIDDEN, {"error": "forbidden host"})
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            body = (STATIC_DIR / "index.html").read_bytes()
            csp = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                   "connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'")
            return self._send(HTTPStatus.OK, body, "text/html; charset=utf-8", {"Content-Security-Policy": csp})
        if url.path == "/api/health":
            return self._json(HTTPStatus.OK, {**self.server.holder.status(), "attribution": ATTRIBUTION})
        if url.path == "/api/passage":
            chunk_id = (parse_qs(url.query).get("id") or [""])[0]
            engine = self.server.holder.engine
            if engine is None:
                return self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "loading"})
            doc = engine.db[CHUNKS].find_one({"_id": chunk_id}, {"text": 1, "film": 1, "year": 1,
                                                                 "scene_number": 1, "heading": 1})
            if doc is None:
                return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return self._json(HTTPStatus.OK, doc)
        return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        """Handle POST requests with a JSON body (after the Host and Origin checks, else 403).

        Routes: ``/api/reset`` ``{"session_id"}`` forgets a session; ``/api/chat`` ``{"question", "session_id"}``
        validates the question (non-empty, at most ``MAX_QUESTION`` characters, else 400) and streams the
        answer. A missing or invalid body is 400, an unknown path 404.
        """
        if not self._host_ok() or not self._origin_ok():
            return self._json(HTTPStatus.FORBIDDEN, {"error": "forbidden"})
        url = urlparse(self.path)
        data = self._read_json()
        if data is None:
            return self._json(HTTPStatus.BAD_REQUEST, {"error": "expected a JSON body"})
        if url.path == "/api/reset":
            self.server.sessions.reset(str(data.get("session_id", "")))
            return self._json(HTTPStatus.OK, {"ok": True})
        if url.path != "/api/chat":
            return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        question = str(data.get("question", "")).strip()
        if not question:
            return self._json(HTTPStatus.BAD_REQUEST, {"error": "empty question"})
        if len(question) > MAX_QUESTION:
            return self._json(HTTPStatus.BAD_REQUEST, {"error": f"question longer than {MAX_QUESTION} characters"})
        self._stream_answer(question, str(data.get("session_id") or ""))

    def _stream_answer(self, question: str, session_id: str) -> None:
        """Answer a question as a Server-Sent Events stream, then close the connection.

        Events: ``session`` ``{"session_id"}`` first, ``status`` ``{"stage": "loading"}`` while the models load,
        then the engine's events (``status``, ``plan``, ``sources``, ``token``, ``done``), or ``error``
        ``{"message"}`` on failure. Waits for model loading if needed. If writing to the client fails (tab
        closed, Stop pressed), the cancel event is set and generation stops.

        Args:
            question: The validated question.
            session_id: Client's session id, or "" to start a new session.
        """
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        cancel = threading.Event()
        session = self.server.sessions.get(session_id or None)

        def emit(event: str, payload: dict[str, Any]) -> None:
            """Write one SSE event and flush it to the client.

            Args:
                event: Event name.
                payload: JSON-serializable event data.

            Raises:
                Cancelled: The client disconnected (also sets ``cancel``).
            """
            try:
                self.wfile.write(f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                                 .encode("utf-8"))
                self.wfile.flush()
            except OSError:  # the browser went away (closed tab or pressed Stop)
                cancel.set()
                raise Cancelled() from None

        holder = self.server.holder
        try:
            emit("session", {"session_id": session.id})
            if not holder.ready.is_set():
                emit("status", {"stage": "loading"})
                holder.ready.wait()
            if holder.engine is None:
                emit("error", {"message": holder.error or "the models could not be loaded"})
                return
            holder.engine.ask(question, session, emit, cancel)
        except Cancelled:
            log.info("generation cancelled by the client")
        except Exception as exc:  # noqa: BLE001 - report to the page instead of dropping the connection
            log.exception("chat request failed")
            try:
                emit("error", {"message": f"{exc.__class__.__name__}: {exc}"})
            except Cancelled:
                pass


def make_server(host: str, port: int, holder: EngineHolder, sessions: SessionStore | None = None) -> ChatHTTPServer:
    """Create (and bind) the chat server; the caller runs ``serve_forever``.

    Args:
        host: Interface to bind (127.0.0.1 normally).
        port: TCP port.
        holder: The background-loaded engine.
        sessions: Session store, or None for a new one.

    Returns:
        The bound server.
    """
    return ChatHTTPServer((host, port), holder, sessions or SessionStore())
