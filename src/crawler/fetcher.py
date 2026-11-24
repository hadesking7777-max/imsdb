"""Polite HTTP fetching.

* per-host minimum interval between requests (shared across worker threads)
* retries with exponential backoff + jitter for timeouts, connection errors, 408/429/5xx
* honours ``Retry-After`` (seconds or HTTP-date); very long values are not waited out inline
  but surfaced to the queue so the URL is rescheduled
* never bypasses access control: 401/403/407/451 raise a non-retryable *blocked* error
* redirects are followed manually and only within the allowed hosts
* response size is capped; waits are interruptible for graceful shutdown
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx

from src.config import Settings
from src.crawler.url_manager import is_imsdb_url, normalize_url
from src.errors import FetchCancelled, FetchError
from src.utils.clock import utcnow
from src.utils.logging import get_logger

log = get_logger("fetcher")

RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504, 522, 524})
BLOCKED_STATUS = frozenset({401, 403, 407, 451})
KEPT_HEADERS = ("content-type", "content-length", "content-encoding", "etag", "last-modified",
                "date", "server", "cache-control", "retry-after", "location", "vary", "age",
                "via", "x-robots-tag")
MAX_REDIRECTS = 5


@dataclass(slots=True)
class FetchResult:
    """Outcome of one successful (2xx or 304) fetch, after any redirects.

    Attributes:
        url: The URL originally requested.
        final_url: URL of the response that produced the body (last redirect hop).
        status_code: HTTP status of the final response (2xx, or 304 for a conditional hit).
        headers: Subset of response headers listed in ``KEPT_HEADERS``, keyed by lower-case name.
        body: Raw response body; empty for 304 responses.
        fetched_at: UTC time the response was received.
        elapsed_ms: Wall time of the successful attempt (including redirects and rate-limit waits), in ms.
        attempts: 1-based number of the attempt that succeeded.
        http_version: Protocol reported by HTTPX, e.g. ``"HTTP/1.1"``.
        redirects: Redirect targets followed, in order (empty when none).
    """

    url: str
    final_url: str
    status_code: int
    headers: dict[str, str]
    body: bytes
    fetched_at: datetime
    elapsed_ms: int
    attempts: int
    http_version: str
    redirects: list[str] = field(default_factory=list)

    @property
    def not_modified(self) -> bool:
        """Whether the server answered a conditional request with 304 Not Modified.

        Returns:
            True when ``status_code`` is 304 (the body is empty and the stored copy is current).
        """
        return self.status_code == 304

    @property
    def content_type(self) -> str | None:
        """The response's ``Content-Type`` header.

        Returns:
            The raw header value (may include a charset), or None when the server sent none.
        """
        return self.headers.get("content-type")


class RateLimiter:
    """Enforces a minimum interval between request *starts* per host.

    Thread-safe: worker threads share one instance, and each ``reserve`` call hands out the next free
    slot for the host, so concurrent workers are spaced out rather than firing together.
    """

    def __init__(self, min_interval: float, clock: Callable[[], float] = time.monotonic) -> None:
        """Create a limiter with no reservations yet.

        Args:
            min_interval: Minimum gap between two request starts to the same host, in seconds.
            clock: Monotonic time source in seconds; injectable for tests.
        """
        self.min_interval = min_interval
        self._clock = clock
        self._lock = threading.Lock()
        self._next_allowed: dict[str, float] = {}

    def set_interval(self, seconds: float) -> None:
        """Raise the minimum interval, e.g. to honour a robots.txt ``Crawl-delay``.

        The interval can only grow: a value lower than the current one is ignored.

        Args:
            seconds: Requested minimum interval between request starts, in seconds.
        """
        with self._lock:
            self.min_interval = max(self.min_interval, seconds)

    def reserve(self, host: str) -> float:
        """Reserve the next request slot for a host.

        The slot is booked immediately (the following slot moves ``min_interval`` later), so the
        caller must actually wait the returned time and then send its request.

        Args:
            host: Host name the request will go to.

        Returns:
            Seconds to wait before using the slot; 0.0 when the host is free now.
        """
        with self._lock:
            now = self._clock()
            slot = max(now, self._next_allowed.get(host, now))
            self._next_allowed[host] = slot + self.min_interval
            return slot - now

    def penalize(self, host: str, seconds: float) -> None:
        """Push the host's next slot at least ``seconds`` into the future (Retry-After).

        Never moves an already later slot earlier.

        Args:
            host: Host name that asked us to back off.
            seconds: Delay from now before the host may be contacted again, in seconds.
        """
        with self._lock:
            target = self._clock() + seconds
            if target > self._next_allowed.get(host, 0.0):
                self._next_allowed[host] = target


def parse_retry_after(value: str | None, now: datetime | None = None) -> float | None:
    """Parse a ``Retry-After`` header into a delay.

    Args:
        value: Header value: either a number of seconds or an HTTP-date; None or empty when absent.
        now: Reference time for HTTP-dates (UTC now when None). Naive dates are treated as UTC.

    Returns:
        Delay in seconds (never negative; past dates give 0.0), or None when the header is missing
        or unparseable.
    """
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - (now or datetime.now(UTC))).total_seconds())


class Fetcher:
    """Polite, retrying HTTP GET client for IMSDb (see the module docstring for the policy).

    Wraps one ``httpx.Client`` with automatic redirects disabled; use it as a context manager or call
    ``close``. Safe to share between worker threads.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        rate_limiter: RateLimiter | None = None,
        stop_event: threading.Event | None = None,
        transport: httpx.BaseTransport | None = None,
        rng: random.Random | None = None,
    ) -> None:
        """Build the HTTP client from the settings.

        Args:
            settings: Application settings (user agent, timeouts, retry/backoff limits, max response size,
                minimum request interval).
            rate_limiter: Shared per-host limiter; a new one using ``settings.min_request_interval`` when None.
            stop_event: Event that, once set, cancels waits and further attempts; a private one when None.
            transport: Custom HTTPX transport (e.g. ``httpx.MockTransport`` in tests); default network
                transport when None.
            rng: Random source for backoff jitter; injectable for deterministic tests.
        """
        self.settings = settings
        self.rate_limiter = rate_limiter or RateLimiter(settings.min_request_interval)
        self.stop_event = stop_event or threading.Event()
        self._rng = rng or random.Random()
        self._client = httpx.Client(
            headers={
                "User-Agent": settings.user_agent,
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5",
                "Accept-Language": "en;q=0.9",
            },
            timeout=httpx.Timeout(settings.request_timeout, connect=settings.connect_timeout),
            follow_redirects=False,
            transport=transport,
        )

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        """Close the underlying HTTPX client and its connection pool."""
        self._client.close()

    def __enter__(self) -> "Fetcher":
        """Enter a ``with`` block.

        Returns:
            This fetcher.
        """
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the client when the ``with`` block exits; exceptions are not suppressed.

        Args:
            *exc: Exception type, value and traceback (ignored).
        """
        self.close()

    # ------------------------------------------------------------------ helpers
    def _sleep(self, seconds: float) -> None:
        """Wait, waking early if shutdown is requested.

        Args:
            seconds: Time to wait in seconds; zero or negative returns immediately.

        Raises:
            FetchCancelled: The stop event was set before or during the wait.
        """
        if seconds <= 0:
            return
        if self.stop_event.wait(seconds):
            raise FetchCancelled("shutdown requested")

    def _backoff(self, attempt: int) -> float:
        """Compute the delay before the next retry (exponential backoff with equal jitter).

        Args:
            attempt: 1-based number of the attempt that just failed.

        Returns:
            Delay in seconds: ``min(backoff_max, backoff_base * 2**(attempt-1))`` scaled by a random
            factor in [0.5, 1.0).
        """
        base = min(self.settings.backoff_max, self.settings.backoff_base * (2 ** (attempt - 1)))
        return base * (0.5 + self._rng.random() / 2)  # "equal jitter"

    def _read_limited(self, response: httpx.Response) -> bytes:
        """Read a streamed response body, enforcing ``settings.max_response_bytes``.

        Rejects early on an oversized ``Content-Length`` and otherwise counts bytes while streaming.

        Args:
            response: An open streaming HTTPX response.

        Returns:
            The complete (decoded) body.

        Raises:
            FetchError: The body is, or declares itself, larger than the limit (not retryable).
        """
        limit = self.settings.max_response_bytes
        declared = response.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > limit:
            raise FetchError(f"response too large ({declared} bytes > {limit})",
                             url=str(response.url), status_code=response.status_code)
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_bytes():
            total += len(chunk)
            if total > limit:
                raise FetchError(f"response exceeded {limit} bytes", url=str(response.url),
                                 status_code=response.status_code)
            chunks.append(chunk)
        return b"".join(chunks)

    # ------------------------------------------------------------------ main entry
    def fetch(
        self,
        url: str,
        *,
        etag: str | None = None,
        last_modified: str | None = None,
        allow_url: Callable[[str], bool] | None = None,
    ) -> FetchResult:
        """Fetch ``url`` with retries, sending conditional headers when validators are known.

        Up to ``settings.max_retries + 1`` attempts. Only retryable errors (timeouts, transport errors,
        408/425/429/5xx) are retried, after an exponential-backoff delay or the server's ``Retry-After``
        if longer; a ``Retry-After`` also penalizes the host in the rate limiter. A ``Retry-After`` above
        ``settings.max_retry_after`` is not waited out: the error is re-raised so the queue reschedules.
        Each retry is logged at INFO level.

        Args:
            url: Normalized absolute URL to GET.
            etag: Stored ``ETag``, sent as ``If-None-Match``; None to skip.
            last_modified: Stored ``Last-Modified``, sent as ``If-Modified-Since``; None to skip.
            allow_url: Permission-gate callback that vets each canonical redirect target; None allows any
                IMSDb target.

        Returns:
            The result of the first successful attempt (2xx, or 304 with an empty body).

        Raises:
            FetchCancelled: Shutdown was requested before an attempt or during a wait.
            FetchError: The fetch failed for good; ``attempts`` is set to the number of attempts made,
                and ``blocked``/``retryable``/``retry_after`` tell the caller how to treat the URL.
        """
        conditional: dict[str, str] = {}
        if etag:
            conditional["If-None-Match"] = etag
        if last_modified:
            conditional["If-Modified-Since"] = last_modified

        max_attempts = self.settings.max_retries + 1
        last_error: FetchError | None = None
        for attempt in range(1, max_attempts + 1):
            if self.stop_event.is_set():
                raise FetchCancelled("shutdown requested")
            try:
                return self._fetch_once(url, conditional, attempt, allow_url)
            except FetchError as exc:
                exc.attempts = attempt
                if not exc.retryable or attempt >= max_attempts:
                    raise
                last_error = exc
                delay = self._backoff(attempt)
                if exc.retry_after is not None:
                    if exc.retry_after > self.settings.max_retry_after:
                        log.warning("Retry-After %.0fs exceeds limit; rescheduling %s",
                                    exc.retry_after, url)
                        raise
                    delay = max(delay, exc.retry_after)
                    host = httpx.URL(url).host
                    self.rate_limiter.penalize(host, exc.retry_after)
                log.info("retrying %s in %.1fs (attempt %d/%d): %s", url, delay, attempt,
                         max_attempts, exc)
                self._sleep(delay)
        assert last_error is not None  # pragma: no cover - loop always returns or raises
        raise last_error

    def _fetch_once(
        self,
        url: str,
        conditional: dict[str, str],
        attempt: int,
        allow_url: Callable[[str], bool] | None,
    ) -> FetchResult:
        """Make one attempt: GET ``url`` and follow up to ``MAX_REDIRECTS`` redirects by hand.

        Every hop waits for a rate-limiter slot first. Redirects must stay on IMSDb, pass ``allow_url``
        and must not loop.

        Args:
            url: URL originally requested.
            conditional: ``If-None-Match`` / ``If-Modified-Since`` headers to send (may be empty).
            attempt: 1-based attempt number, recorded in the result.
            allow_url: Permission-gate callback for canonical redirect targets, or None.

        Returns:
            The final 2xx or 304 response.

        Raises:
            FetchCancelled: Shutdown requested while waiting for a rate-limiter slot.
            FetchError: Retryable for timeouts, transport errors and ``RETRYABLE_STATUS`` (with
                ``retry_after`` parsed); blocked for ``BLOCKED_STATUS`` or a redirect refused by
                ``allow_url``; non-retryable for off-site redirects, loops, too many redirects, other
                statuses and oversized bodies.
        """
        current = url
        redirects: list[str] = []
        started = time.monotonic()
        for _hop in range(MAX_REDIRECTS + 1):
            host = httpx.URL(current).host
            self._sleep(self.rate_limiter.reserve(host))
            try:
                with self._client.stream("GET", current, headers=conditional) as response:
                    status = response.status_code
                    headers = {k: response.headers[k] for k in KEPT_HEADERS if k in response.headers}
                    if status in (301, 302, 303, 307, 308):
                        location = response.headers.get("location")
                        # Request the server's exact target (keeps a trailing slash); vet its canonical form.
                        target = normalize_url(location, base=current, strip_trailing_slash=False)
                        canonical = normalize_url(location, base=current)
                        if not target or not canonical or not is_imsdb_url(target):
                            raise FetchError(f"redirect to disallowed location {location!r}",
                                             url=url, status_code=status)
                        if allow_url and not allow_url(canonical):
                            raise FetchError(f"redirect target not permitted: {canonical}",
                                             url=url, status_code=status, blocked=True)
                        if target in redirects or target == current:
                            raise FetchError("redirect loop detected", url=url, status_code=status)
                        redirects.append(target)
                        current = target
                        continue
                    if status in BLOCKED_STATUS:
                        raise FetchError(f"access restricted (HTTP {status}); not bypassing",
                                         url=url, status_code=status, blocked=True)
                    if status in RETRYABLE_STATUS:
                        raise FetchError(f"HTTP {status}", url=url, status_code=status,
                                         retryable=True,
                                         retry_after=parse_retry_after(headers.get("retry-after")))
                    if status == 304:
                        body = b""
                    elif 200 <= status < 300:
                        body = self._read_limited(response)
                    else:
                        raise FetchError(f"HTTP {status}", url=url, status_code=status)
                    return FetchResult(
                        url=url,
                        final_url=str(response.url),
                        status_code=status,
                        headers=headers,
                        body=body,
                        fetched_at=utcnow(),
                        elapsed_ms=int((time.monotonic() - started) * 1000),
                        attempts=attempt,
                        http_version=response.http_version,
                        redirects=redirects,
                    )
            except httpx.TimeoutException as exc:
                raise FetchError(f"timeout: {exc.__class__.__name__}", url=url, retryable=True) from exc
            except httpx.TransportError as exc:
                raise FetchError(f"transport error: {exc}", url=url, retryable=True) from exc
        raise FetchError("too many redirects", url=url)
