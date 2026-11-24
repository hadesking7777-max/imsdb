"""Shared test fixtures. Tests never access the live IMSDb site."""

from __future__ import annotations

import os
import threading
import uuid
from pathlib import Path
from typing import Iterator

import httpx
import pytest

from src.config import Settings, load_settings
from src.crawler.fetcher import Fetcher, RateLimiter
from src.crawler.url_manager import normalize_url
from src.database.connection import Mongo
from src.database.indexes import ensure_schema
from src.errors import DatabaseUnavailableError

FIXTURES = Path(__file__).parent / "fixtures"
BASE = "https://imsdb.com"


def fixture_text(name: str) -> str:
    """Read a file from ``tests/fixtures`` as UTF-8 text.

    Args:
        name: File name inside the fixtures folder, e.g. ``"all_scripts.html"``.

    Returns:
        The file's decoded contents.
    """
    return (FIXTURES / name).read_text(encoding="utf-8")


def fixture_bytes(name: str) -> bytes:
    """Read a file from ``tests/fixtures`` as raw bytes.

    Args:
        name: File name inside the fixtures folder.

    Returns:
        The file's exact bytes (used as fake HTTP response bodies).
    """
    return (FIXTURES / name).read_bytes()


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent a developer's .env / environment from leaking into tests.

    Runs automatically for every test and removes all ``MONGODB_*``, ``IMSDB_*`` and ``MONGOD_*`` variables.

    Args:
        monkeypatch: Removes the variables for the duration of the test only.
    """
    for key in list(os.environ):
        if key.startswith(("MONGODB_", "IMSDB_", "MONGOD_")):
            monkeypatch.delenv(key, raising=False)


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    """Build test settings that keep all files under a temporary folder and use a throwaway database.

    The defaults make tests fast: a short MongoDB timeout, tiny backoff delays, few retries and a random
    ``imsdb_test_<hex>`` database name. No ``.env`` file is read.

    Args:
        tmp_path: Folder that receives the ``data`` and ``logs`` sub-folders.
        **overrides: Setting values that replace the test defaults (any ``load_settings`` keyword).

    Returns:
        Validated settings whose data/log directories already exist.
    """
    values: dict[str, object] = {
        "data_dir": tmp_path / "data",
        "log_dir": tmp_path / "logs",
        "mongodb_db": f"imsdb_test_{uuid.uuid4().hex[:10]}",
        "mongodb_timeout_ms": 1500,
        "backoff_base": 0.01,
        "backoff_max": 0.05,
        "max_retries": 2,
        "concurrency": 2,
        "lease_seconds": 60,
        "max_queue_attempts": 3,
    }
    values.update(overrides)
    settings = load_settings(env_file=tmp_path / "absent.env", **values)
    settings.ensure_directories()
    return settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Default test settings (see ``make_settings``).

    Args:
        tmp_path: pytest's per-test temporary folder.

    Returns:
        Settings pointing at ``tmp_path`` and a unique throwaway database name.
    """
    return make_settings(tmp_path)


@pytest.fixture
def mongo(settings: Settings) -> Iterator[Mongo]:
    """A connection to the local MongoDB with the project schema applied to a throwaway database.

    The test is skipped when no local server is reachable. The database is dropped afterwards.

    Args:
        settings: Test settings that name the throwaway database.

    Yields:
        The open ``Mongo`` connection.
    """
    try:
        connection = Mongo.connect(settings)
    except DatabaseUnavailableError:
        pytest.skip("local MongoDB server not reachable on the configured URI")
    ensure_schema(connection.db)
    try:
        yield connection
    finally:
        connection.client.drop_database(connection.db.name)
        connection.close()


# ====================================================================== fake IMSDb site
def _u(path: str, *, exact: bool = False) -> str:
    """Turn a site path into the canonical ``https://imsdb.com`` URL used as a key by ``FakeSite``.

    Args:
        path: A path such as ``"/genre/Drama"`` (or a full URL).
        exact: Keep a trailing slash instead of stripping it (needed for ``/privacy/``).

    Returns:
        The normalized absolute URL.
    """
    url = normalize_url(path, base=BASE + "/", strip_trailing_slash=not exact)
    assert url is not None
    return url


class FakeSite:
    """A tiny in-memory IMSDb built from the HTML fixtures.

    Pages are keyed by normalized URL. ``robots.txt`` returns 404 (allow all) unless a test overrides it.
    Every request URL is recorded in ``requests`` so tests can assert what was (not) fetched.
    """

    def __init__(self) -> None:
        """Register the fixture pages: home, listings, five detail pages, three scripts and policy pages.

        The ``/privacy`` 301 redirect to ``http://imsdb.com/privacy/`` copies the live site's behaviour.
        """
        self.pages: dict[str, tuple[int, bytes, dict[str, str]]] = {}
        self.sequences: dict[str, list[tuple[int, bytes, dict[str, str]]]] = {}
        self.requests: list[str] = []
        self._lock = threading.Lock()
        html = {"content-type": "text/html; charset=UTF-8"}
        self.add("/robots.txt", b"Not Found", status=404)
        routes = {
            "/": "home.html",
            "/all-scripts.html": "all_scripts.html",
            "/alphabetical/L": "alpha_L.html",
            "/alphabetical/P": "alpha_P.html",
            "/genre/Drama": "genre_drama.html",
            "/Movie Scripts/Lighthouse Keeper, The Script.html": "detail_lighthouse.html",
            "/Movie Scripts/Paper Moons Script.html": "detail_paper_moons.html",
            "/Movie Scripts/Quiet Orchard Script.html": "detail_quiet_orchard.html",
            "/Movie Scripts/Night Shift at Dawn Script.html": "detail_night_shift.html",
            "/Movie Scripts/Ember Road Script.html": "detail_ember_road.html",
            "/scripts/Lighthouse-Keeper,-The.html": "script_lighthouse.html",
            "/scripts/Paper-Moons.html": "script_flush.html",
            "/scripts/Quiet-Orchard.html": "script_br.html",
        }
        for path, name in routes.items():
            self.add(path, fixture_bytes(name), headers=html)
        disclaimer = (b"<html><body><table><tr><td><h1>Disclaimer</h1><p>Section 107. Limitations on exclusive "
                      b"rights: Fair use. All scripts on this site are educational purposes only.</p></td></tr>"
                      b"</table></body></html>")
        self.add("/disclaimer.html", disclaimer, headers=html)
        self.add("/contact.html", b"<html><body><h1>Contact IMSDb</h1></body></html>", headers=html)
        self.add("/submit.html", b"<html><body><h1>Submit</h1></body></html>", headers=html)
        # Mirrors the live site: /privacy -> 301 http://imsdb.com/privacy/ (trailing slash, plain http).
        self.add("/privacy", b"", status=301, headers={"location": "http://imsdb.com/privacy/"})
        self.add("/privacy/", b"<html><body><h1>Privacy policy</h1><p>We use cookies.</p></body></html>",
                 headers=html, exact=True)

    def add(self, path: str, body: bytes | str, *, status: int = 200, headers: dict[str, str] | None = None,
            exact: bool = False) -> None:
        """Register (or replace) the response served for one path.

        Args:
            path: Site path or URL to serve.
            body: Response body; ``str`` is encoded as UTF-8.
            status: HTTP status code to return.
            headers: Response headers; defaults to an HTML ``content-type``.
            exact: Keep a trailing slash in the key (see ``_u``).
        """
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.pages[_u(path, exact=exact)] = (status, data, headers or {"content-type": "text/html; charset=UTF-8"})

    def script(self, path: str, responses: list[tuple[int, bytes | str, dict[str, str] | None]]) -> None:
        """Serve ``responses`` in order for ``path`` before falling back to ``pages``.

        Args:
            path: Site path whose next requests get the queued responses.
            responses: ``(status, body, headers)`` tuples consumed one per request; ``headers`` may be ``None``.
        """
        self.sequences[_u(path)] = [(s, b.encode() if isinstance(b, str) else b, h or {}) for s, b, h in responses]

    def count(self, path: str) -> int:
        """Count how many times a path has been requested.

        Args:
            path: Site path or URL (normalized before comparison).

        Returns:
            The number of recorded requests for that URL.
        """
        url = _u(path)
        return sum(1 for r in self.requests if r == url)

    def handler(self, request: httpx.Request) -> httpx.Response:
        """Answer one HTTP request from the queued responses or the registered pages (thread-safe).

        A page registered with its exact URL (trailing slash kept) wins over the normalized form.

        Args:
            request: The request made by the code under test.

        Returns:
            The registered response, or a 404 HTML response for unknown URLs.
        """
        exact = normalize_url(str(request.url), strip_trailing_slash=False) or str(request.url)
        url = normalize_url(str(request.url)) or str(request.url)
        with self._lock:
            self.requests.append(url)
            queued = self.sequences.get(url)
            entry = queued.pop(0) if queued else self.pages.get(exact, self.pages.get(url))
        if entry is None:
            return httpx.Response(404, content=b"not found", headers={"content-type": "text/html"})
        status, body, headers = entry
        return httpx.Response(status, content=body, headers=headers)

    def transport(self) -> httpx.MockTransport:
        """Wrap ``handler`` as an httpx transport.

        Returns:
            A ``MockTransport`` to pass to ``Fetcher`` / ``httpx.Client``.
        """
        return httpx.MockTransport(self.handler)


@pytest.fixture
def site() -> FakeSite:
    """A fresh fake IMSDb for each test.

    Returns:
        A new ``FakeSite`` with the default fixture pages.
    """
    return FakeSite()


@pytest.fixture
def app(settings: Settings, mongo: Mongo, site: FakeSite) -> Iterator[object]:
    """An ingestion ``AppContext`` wired to the throwaway database and the fake site, with no rate limiting.

    Args:
        settings: Test settings.
        mongo: Connection to the throwaway database.
        site: The fake IMSDb that serves every request.

    Yields:
        The ``AppContext``; its fetcher is closed afterwards.
    """
    from src.services.ingestion import AppContext

    stop = threading.Event()
    fetcher = Fetcher(settings, rate_limiter=RateLimiter(0.0), stop_event=stop, transport=site.transport())
    ctx = AppContext(settings, mongo, stop_event=stop, fetcher=fetcher)
    try:
        yield ctx
    finally:
        fetcher.close()


def BASE_URL(path: str) -> str:  # noqa: N802 - helper used like a constant in tests
    """Return the canonical absolute IMSDb URL for a path (same as ``_u`` without ``exact``).

    Args:
        path: A site path such as ``"/genre/Drama"``.

    Returns:
        The normalized ``https://imsdb.com/...`` URL.
    """
    return _u(path)
