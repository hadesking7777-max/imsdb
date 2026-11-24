"""Crawler tests: URLs, discovery, fetching/retries, rate limiting, robots, permission gate."""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest

from src.crawler.discovery import detail_items, listing_followups, make_queue_item, seed_items
from src.crawler.fetcher import Fetcher, RateLimiter, parse_retry_after
from src.crawler.url_manager import (
    PageType,
    classify_url,
    detail_title_from_url,
    is_crawlable,
    listing_label,
    normalize_url,
    url_id,
)
from src.errors import FetchCancelled, FetchError, PermissionDeniedError
from src.services.permissions import (
    PermissionGate,
    RobotsPolicy,
    assess_policy_text,
    default_imsdb_permission,
)
from tests.conftest import BASE_URL


# ====================================================================== URL normalization
@pytest.mark.parametrize("raw,expected", [
    ("/Movie Scripts/12 Monkeys Script.html", "https://imsdb.com/Movie%20Scripts/12%20Monkeys%20Script.html"),
    ("/Movie%20Scripts/12%20Monkeys%20Script.html", "https://imsdb.com/Movie%20Scripts/12%20Monkeys%20Script.html"),
    ("http://www.IMSDB.com/genre/Action/", "https://imsdb.com/genre/Action"),
    ("https://imsdb.com:443/alphabetical/A#top", "https://imsdb.com/alphabetical/A"),
    ("https://imsdb.com//scripts//Alien.html", "https://imsdb.com/scripts/Alien.html"),
    ("/writer.php?w=David Peoples", "https://imsdb.com/writer.php?w=David%20Peoples"),
    ("https://imsdb.com", "https://imsdb.com/"),
    ("javascript:void(0)", None),
    ("mailto:someone@example.org", None),
    ("#comments", None),
    ("", None),
])
def test_normalize_url(raw: str, expected: str | None) -> None:
    """URLs become absolute https URLs without ``www``, port, fragment or extra slashes, and are percent-encoded.

    Non-HTTP links (``javascript:``, ``mailto:``, bare fragments, empty) normalize to ``None``.

    Args:
        raw: The link as found in a page.
        expected: The canonical URL, or ``None`` for unusable links.
    """
    assert normalize_url(raw, base="https://imsdb.com/") == expected


def test_normalization_is_idempotent_and_ids_are_stable() -> None:
    """Normalizing a canonical URL changes nothing, and spelling variants get the same ``url_id``."""
    url = normalize_url("/Movie Scripts/Dark Knight, The Script.html", base="https://imsdb.com/")
    assert url is not None
    assert normalize_url(url) == url
    assert url_id(url) == url_id(normalize_url("https://www.imsdb.com/Movie%20Scripts/Dark%20Knight,%20The%20Script.html#x") or "")


@pytest.mark.parametrize("path,page_type", [
    ("/", PageType.HOME), ("/all-scripts.html", PageType.LISTING_ALL), ("/alphabetical/0", PageType.LISTING_ALPHA),
    ("/genre/Sci-Fi", PageType.LISTING_GENRE), ("/language/French", PageType.LISTING_LANGUAGE),
    ("/Movie Scripts/Alien Script.html", PageType.MOVIE_DETAIL), ("/scripts/Alien.html", PageType.SCRIPT_HTML),
    ("/scripts/Alien.pdf", PageType.SCRIPT_PDF), ("/writer.php?w=X", PageType.WRITER),
    ("/TV Transcripts/Lost.html", PageType.TV), ("/disclaimer.html", PageType.POLICY), ("/privacy", PageType.POLICY),
    ("/privacy.html", PageType.POLICY), ("/robots.txt", PageType.ROBOTS), ("/style.css", PageType.ASSET),
    ("/addcomment.php", PageType.FORM), ("/something/else", PageType.OTHER),
])
def test_classify_url(path: str, page_type: PageType) -> None:
    """Each IMSDb URL shape maps to the expected page type.

    Args:
        path: Site path to classify.
        page_type: Expected ``PageType``.
    """
    assert classify_url(BASE_URL(path)) == page_type


def test_external_urls_are_not_crawlable() -> None:
    """Other hosts and writer pages are not crawlable; script pages on IMSDb are."""
    assert classify_url("https://example.org/scripts/x.html") == PageType.EXTERNAL
    assert not is_crawlable("https://example.org/")
    assert not is_crawlable(BASE_URL("/writer.php?w=X"))
    assert is_crawlable(BASE_URL("/scripts/Alien.html"))


def test_title_and_label_helpers() -> None:
    """The detail-page URL yields the movie title and listing URLs yield ``kind:value`` labels."""
    assert detail_title_from_url(BASE_URL("/Movie Scripts/Dark Knight, The Script.html")) == "Dark Knight, The"
    assert listing_label(BASE_URL("/genre/Drama")) == "genre:Drama"
    assert listing_label(BASE_URL("/alphabetical/0")) == "alphabetical:0"


# ====================================================================== discovery
def test_seed_items() -> None:
    """Full discovery seeds the home page and the all-scripts list; quick discovery seeds only the list."""
    full = seed_items("https://imsdb.com/")
    assert [i.page_type for i in full] == ["home", "listing_all"]
    quick = seed_items("https://imsdb.com/", quick=True)
    assert [i.url for i in quick] == ["https://imsdb.com/all-scripts.html"]


def test_listing_followups_are_bounded_and_filtered() -> None:
    """Only genre/alphabetical listings are followed, and nothing is followed beyond the depth cap."""
    links = [BASE_URL("/genre/Drama"), BASE_URL("/alphabetical/A"), BASE_URL("/writer.php?w=x"), BASE_URL("/")]
    items = listing_followups(links, source_url=BASE_URL("/"), depth=0)
    assert [i.page_type for i in items] == ["listing_genre", "listing_alpha"]
    assert listing_followups(links, source_url=BASE_URL("/genre/Drama"), depth=2) == []  # depth cap prevents loops


def test_detail_items_keep_order_and_ignore_other_types() -> None:
    """Detail-page queue items keep their link position and movie id; non-detail links are skipped."""
    items = detail_items([(BASE_URL("/Movie Scripts/A Script.html"), "mv-a"), (BASE_URL("/scripts/A.html"), None),
                          (BASE_URL("/Movie Scripts/B Script.html"), None)], source_url=BASE_URL("/"), depth=0)
    assert [(i.position, i.movie_id) for i in items] == [(0, "mv-a"), (2, None)]
    assert all(i.page_type == "movie_detail" and i.priority == 50 for i in items)


def test_make_queue_item_rejects_unusable_urls() -> None:
    """Unusable URLs give ``None``; script pages get the ``script_html`` type and priority 100."""
    assert make_queue_item("javascript:alert(1)") is None
    item = make_queue_item("/scripts/X.html".join(["https://imsdb.com", ""]))
    assert item is not None and item.page_type == "script_html" and item.priority == 100


# ====================================================================== fetcher
def _fetcher(settings, handler, stop: threading.Event | None = None) -> Fetcher:
    """Build a ``Fetcher`` with no rate limiting that answers every request with ``handler``.

    Args:
        settings: Test settings.
        handler: Function ``request -> httpx.Response`` that plays the server.
        stop: Shutdown event; a new unset event when omitted.

    Returns:
        The fetcher (use it as a context manager so it is closed).
    """
    return Fetcher(settings, rate_limiter=RateLimiter(0.0), transport=httpx.MockTransport(handler),
                   stop_event=stop or threading.Event())


def test_fetch_success_records_metadata(settings) -> None:
    """A 200 response returns body, headers and attempt count, and the configured User-Agent is sent.

    Args:
        settings: Test settings (few retries, tiny backoff).
    """
    def handler(request: httpx.Request) -> httpx.Response:
        """Check the User-Agent header and return a small HTML page with an ETag.

        Args:
            request: The request made by the fetcher.

        Returns:
            A 200 HTML response.
        """
        assert request.headers["user-agent"] == settings.user_agent  # honest, configured identity
        return httpx.Response(200, content=b"<html>ok</html>", headers={"content-type": "text/html", "etag": '"abc"'})
    with _fetcher(settings, handler) as fetcher:
        result = fetcher.fetch("https://imsdb.com/")
    assert result.status_code == 200 and result.body == b"<html>ok</html>"
    assert result.headers["etag"] == '"abc"' and result.attempts == 1


def test_retry_on_server_errors_then_success(settings) -> None:
    """HTTP 503 responses are retried until the third attempt succeeds.

    Args:
        settings: Test settings (few retries, tiny backoff).
    """
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        """Count the call and fail the first two attempts.

        Args:
            request: The request made by the fetcher.

        Returns:
            503 for the first two calls, then 200.
        """
        calls["n"] += 1
        return httpx.Response(503 if calls["n"] < 3 else 200, content=b"x")
    with _fetcher(settings, handler) as fetcher:
        result = fetcher.fetch("https://imsdb.com/")
    assert calls["n"] == 3 and result.attempts == 3


def test_retries_exhausted_raise_retryable_error(settings) -> None:
    """Persistent 500s raise a retryable ``FetchError`` after ``max_retries + 1`` attempts.

    Args:
        settings: Test settings (few retries, tiny backoff).
    """
    with _fetcher(settings, lambda r: httpx.Response(500)) as fetcher:
        with pytest.raises(FetchError) as info:
            fetcher.fetch("https://imsdb.com/")
    assert info.value.retryable and info.value.status_code == 500
    assert info.value.attempts == settings.max_retries + 1


def test_transport_errors_are_retried(settings) -> None:
    """A connection timeout is retried like a server error.

    Args:
        settings: Test settings (few retries, tiny backoff).
    """
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        """Raise a connect timeout on the first call, then succeed.

        Args:
            request: The request made by the fetcher.

        Returns:
            A 200 response from the second call on.
        """
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectTimeout("slow", request=request)
        return httpx.Response(200, content=b"ok")
    with _fetcher(settings, handler) as fetcher:
        assert fetcher.fetch("https://imsdb.com/").status_code == 200


def test_retry_after_is_honoured(settings) -> None:
    """A 429 with ``Retry-After: 0`` is retried and the second attempt succeeds.

    Args:
        settings: Test settings (few retries, tiny backoff).
    """
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        """Rate-limit the first call, then succeed.

        Args:
            request: The request made by the fetcher.

        Returns:
            429 with ``Retry-After: 0`` first, then 200.
        """
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(200, content=b"ok")
    with _fetcher(settings, handler) as fetcher:
        assert fetcher.fetch("https://imsdb.com/").attempts == 2


def test_long_retry_after_is_surfaced_not_waited(settings) -> None:
    """A day-long ``Retry-After`` is raised immediately (with the value) instead of sleeping.

    Args:
        settings: Test settings (few retries, tiny backoff).
    """
    with _fetcher(settings, lambda r: httpx.Response(503, headers={"retry-after": "86400"})) as fetcher:
        with pytest.raises(FetchError) as info:
            fetcher.fetch("https://imsdb.com/")
    assert info.value.retry_after == 86400 and info.value.attempts == 1


def test_parse_retry_after_http_date() -> None:
    """``Retry-After`` accepts an HTTP date or seconds; garbage gives ``None``."""
    now = datetime(2026, 1, 1, tzinfo=UTC)
    assert parse_retry_after(format_datetime(now + timedelta(seconds=30), usegmt=True), now=now) == 30
    assert parse_retry_after("12") == 12
    assert parse_retry_after("garbage") is None


@pytest.mark.parametrize("status", [401, 403, 451])
def test_access_restrictions_are_never_retried(settings, status: int) -> None:
    """401/403/451 are reported as blocked, not retryable, after a single request.

    Args:
        settings: Test settings (few retries, tiny backoff).
        status: The access-restriction status code to return.
    """
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        """Count the call and refuse access.

        Args:
            request: The request made by the fetcher.

        Returns:
            A response with the parametrized ``status``.
        """
        calls["n"] += 1
        return httpx.Response(status)
    with _fetcher(settings, handler) as fetcher:
        with pytest.raises(FetchError) as info:
            fetcher.fetch("https://imsdb.com/")
    assert info.value.blocked and not info.value.retryable and calls["n"] == 1


def test_not_found_is_not_retried(settings) -> None:
    """A 404 raises a non-retryable ``FetchError``.

    Args:
        settings: Test settings (few retries, tiny backoff).
    """
    with _fetcher(settings, lambda r: httpx.Response(404)) as fetcher:
        with pytest.raises(FetchError) as info:
            fetcher.fetch("https://imsdb.com/missing")
    assert info.value.status_code == 404 and not info.value.retryable


def test_redirects_are_followed_only_within_scope(settings) -> None:
    """Redirects within IMSDb are followed; off-site or ``allow_url``-rejected targets raise ``FetchError``.

    Args:
        settings: Test settings (few retries, tiny backoff).
    """
    def handler(request: httpx.Request) -> httpx.Response:
        """Redirect ``/privacy`` within the site and ``/away`` off-site; serve everything else.

        Args:
            request: The request made by the fetcher.

        Returns:
            A 301/302 redirect or a 200 ``policy`` page.
        """
        if request.url.path == "/privacy":
            return httpx.Response(301, headers={"location": "/privacy.html"})
        if request.url.path == "/away":
            return httpx.Response(302, headers={"location": "https://example.org/"})
        return httpx.Response(200, content=b"policy")
    with _fetcher(settings, handler) as fetcher:
        result = fetcher.fetch("https://imsdb.com/privacy")
        assert result.final_url == "https://imsdb.com/privacy.html" and result.redirects
        with pytest.raises(FetchError):
            fetcher.fetch("https://imsdb.com/away")
        with pytest.raises(FetchError) as info:
            fetcher.fetch("https://imsdb.com/privacy", allow_url=lambda u: False)
        assert info.value.blocked


def test_redirect_to_trailing_slash_is_not_a_loop(settings) -> None:
    """Regression: IMSDb redirects /privacy -> http://imsdb.com/privacy/ (found during the live run).

    Args:
        settings: Test settings (few retries, tiny backoff).
    """
    def handler(request: httpx.Request) -> httpx.Response:
        """Redirect ``/privacy`` to the plain-http trailing-slash URL like the live site.

        Args:
            request: The request made by the fetcher.

        Returns:
            The 301 redirect, or the 200 page for ``/privacy/``.
        """
        if request.url.path == "/privacy":
            return httpx.Response(301, headers={"location": "http://imsdb.com/privacy/"})
        return httpx.Response(200, content=b"privacy")
    with _fetcher(settings, handler) as fetcher:
        result = fetcher.fetch("https://imsdb.com/privacy", allow_url=lambda u: u == "https://imsdb.com/privacy")
    assert result.body == b"privacy" and result.final_url == "https://imsdb.com/privacy/"


def test_policy_text_uses_content_heading_not_site_banner() -> None:
    """Policy text is taken from the content cell, not the site banner or sidebar movie titles."""
    from src.services.permissions import policy_main_text
    html = ("<html><body><table><tr><td><h1>The Internet Movie Script Database (IMSDb)</h1></td></tr>"
            "<tr><td>sidebar: Crawl, Spider-Man</td><td><h1>Disclaimer</h1><strong>Section 107. Fair use</strong>"
            "All scripts are for educational purposes.</td></tr></table></body></html>")
    text = policy_main_text(html)
    assert "Section 107" in text and "Spider-Man" not in text


def test_response_size_limit(tmp_path) -> None:
    """A response larger than ``max_response_bytes`` raises ``FetchError``.

    Args:
        tmp_path: Folder for the custom settings.
    """
    from tests.conftest import make_settings

    settings = make_settings(tmp_path, max_response_bytes=10_000)
    with _fetcher(settings, lambda r: httpx.Response(200, content=b"x" * 20_000)) as fetcher:
        with pytest.raises(FetchError, match="exceed|too large"):
            fetcher.fetch("https://imsdb.com/")


def test_shutdown_cancels_pending_fetch(settings) -> None:
    """A fetch started after shutdown was requested raises ``FetchCancelled``.

    Args:
        settings: Test settings (few retries, tiny backoff).
    """
    stop = threading.Event()
    stop.set()
    with _fetcher(settings, lambda r: httpx.Response(200), stop) as fetcher:
        with pytest.raises(FetchCancelled):
            fetcher.fetch("https://imsdb.com/")


def test_rate_limiter_spaces_requests_per_host() -> None:
    """The limiter spaces reservations by the interval per host, and a penalty delays that host only."""
    now = {"t": 100.0}
    limiter = RateLimiter(5.0, clock=lambda: now["t"])
    assert limiter.reserve("imsdb.com") == 0
    assert limiter.reserve("imsdb.com") == 5.0
    assert limiter.reserve("imsdb.com") == 10.0
    assert limiter.reserve("other.org") == 0  # independent per host
    limiter.penalize("other.org", 60)
    assert limiter.reserve("other.org") == 60


def test_minimum_interval_cannot_be_configured_below_floor(tmp_path) -> None:
    """A request interval below the safety floor is a configuration error.

    Args:
        tmp_path: Folder for the custom settings.
    """
    from src.errors import ConfigError
    from tests.conftest import make_settings

    with pytest.raises(ConfigError):
        make_settings(tmp_path, min_request_interval=0.1)


# ====================================================================== robots + permission gate
def _gate(level: str = "sample", robots: RobotsPolicy | None = None, collected: int = 0, limit: int = 10,
          **assessment: object) -> PermissionGate:
    """Build a permission gate from the default IMSDb permission record with test overrides.

    Args:
        level: Permission level (``denied``, ``discovery_only``, ``sample`` or ``bulk``).
        robots: Parsed robots.txt policy, or ``None`` when no policy is loaded.
        collected: Number of records already collected (counts against the sample quota).
        limit: Sample quota.
        **assessment: Policy-assessment flags to override, e.g. ``restriction_found=True``.

    Returns:
        The configured ``PermissionGate``.
    """
    record = default_imsdb_permission()
    record = record.model_copy(update={"level": level, "sample_limit": limit,
                                       "assessment": {**record.assessment, **assessment}})
    return PermissionGate(record, robots, "imsdb-local-research/1.0", lambda: collected)


def test_missing_robots_allows_everything() -> None:
    """A 404 robots.txt allows every URL."""
    robots = RobotsPolicy.from_response(404, "")
    assert robots.allow_all and robots.allowed("https://imsdb.com/scripts/x.html", "bot")


def test_robots_rules_and_crawl_delay() -> None:
    """``Disallow`` rules block matching paths only, and ``Crawl-delay`` is read."""
    robots = RobotsPolicy.from_response(200, "User-agent: *\nDisallow: /scripts/\nCrawl-delay: 7\n")
    assert not robots.allowed("https://imsdb.com/scripts/x.html", "imsdb-local-research")
    assert robots.allowed("https://imsdb.com/genre/Drama", "imsdb-local-research")
    assert robots.crawl_delay == 7.0


@pytest.mark.parametrize("status", [500, 503, None, 401, 403])
def test_unreachable_or_forbidden_robots_disallows_all(status: int | None) -> None:
    """A robots.txt that errors, is unreachable or is forbidden blocks everything.

    Args:
        status: The robots.txt HTTP status (``None`` = no response).
    """
    assert RobotsPolicy.from_response(status, "").disallow_all


def test_gate_levels() -> None:
    """Each permission level allows the right page types; policy pages are always allowed, off-site never."""
    detail = BASE_URL("/Movie Scripts/A Script.html")
    listing = BASE_URL("/genre/Drama")
    assert _gate("sample").check(detail).allowed
    assert _gate("discovery_only").check(listing).allowed
    assert not _gate("discovery_only").check(detail).allowed
    assert not _gate("denied").check(listing).allowed
    assert _gate("denied").check(BASE_URL("/disclaimer.html")).allowed  # policy inspection always allowed
    assert not _gate("bulk").check(BASE_URL("/writer.php?w=x")).allowed  # excluded page types
    assert not _gate("bulk").check("https://example.org/").allowed


def test_gate_applies_robots_on_top_of_level() -> None:
    """robots.txt can block a URL even at the ``bulk`` level, and the reason says so."""
    robots = RobotsPolicy.from_response(200, "User-agent: *\nDisallow: /\n")
    decision = _gate("bulk", robots).check(BASE_URL("/genre/Drama"))
    assert not decision.allowed and "robots.txt" in decision.reason


def test_policy_pages_obey_robots_but_robots_txt_is_always_readable() -> None:
    """Policy pages are allowed at every level but not against robots.txt; robots.txt itself is always allowed.
    Refusals that concern only one URL are marked ``url_specific`` (the crawler skips them); level refusals are not."""
    open_gate = _gate("discovery_only")
    assert open_gate.check(BASE_URL("/disclaimer"), "policy").allowed
    closed = _gate("bulk", RobotsPolicy.from_response(200, "User-agent: *\nDisallow: /\n"))
    refused = closed.check(BASE_URL("/disclaimer"), "policy")
    assert not refused.allowed and "robots.txt" in refused.reason and refused.url_specific
    assert closed.check(BASE_URL("/robots.txt"), "robots").allowed
    assert closed.check("https://example.org/x").url_specific
    level = _gate("discovery_only").check(BASE_URL("/scripts/Some-Film.html"), "script_html")
    assert not level.allowed and not level.url_specific


def test_bulk_crawl_requires_bulk_permission() -> None:
    """``crawl --all`` is refused below ``bulk`` and unlimited at ``bulk``."""
    with pytest.raises(PermissionDeniedError, match="bulk authorization"):
        _gate("sample").plan_crawl(limit=None, all_records=True)
    assert _gate("bulk").plan_crawl(limit=None, all_records=True)[0] is None


def test_sample_quota_clamps_and_exhausts() -> None:
    """A run is clamped to the remaining sample quota and refused once it is used up."""
    limit, quota, notices = _gate("sample", collected=7).plan_crawl(limit=10, all_records=False)
    assert (limit, quota) == (3, 3) and notices
    with pytest.raises(PermissionDeniedError, match="quota is used up"):
        _gate("sample", collected=10).plan_crawl(limit=1, all_records=False)


def test_restrictive_policy_blocks_record_collection() -> None:
    """Restrictive policy language blocks record pages even at ``bulk`` unless explicitly overridden."""
    gate = _gate("bulk", restriction_found=True)
    assert not gate.check(BASE_URL("/scripts/A.html")).allowed
    with pytest.raises(PermissionDeniedError):
        gate.plan_crawl(limit=1, all_records=False)
    assert _gate("bulk", restriction_found=True, restriction_overridden=True).check(BASE_URL("/scripts/A.html")).allowed


def test_policy_text_assessment() -> None:
    """Fair-use text is noted, prohibition text is a restriction, and movie titles like "Crawl" are not."""
    fair = assess_policy_text([{"url": "u", "text": "Section 107 fair use. All scripts are for educational purposes only."}])
    assert fair["fair_use_claim_found"] and not fair["restriction_found"] and not fair["bulk_authorized"]
    strict = assess_policy_text([{"url": "u", "text": "Scraping or automated access is strictly prohibited."}])
    assert strict["restriction_found"]
    movie_titles = assess_policy_text([{"url": "u", "text": "Latest: Crawl, Spider-Man, I, Robot"}])
    assert not movie_titles["restriction_found"]  # sidebar movie titles are not policy language


def test_double_encoded_imsdb_links_share_one_canonical_url() -> None:
    """Regression: IMSDb's alphabetical pages link "Kate %2526 Leopold" while others link "Kate & Leopold".

    All spellings normalize to one URL; a literal ``%25`` that is not a double-encoded escape survives.
    """
    a = normalize_url("/Movie Scripts/Kate %2526 Leopold Script.html", base="https://imsdb.com/")
    b = normalize_url("/Movie Scripts/Kate & Leopold Script.html", base="https://imsdb.com/")
    c = normalize_url("/Movie%20Scripts/Kate%20%26%20Leopold%20Script.html", base="https://imsdb.com/")
    assert a == b == c == "https://imsdb.com/Movie%20Scripts/Kate%20&%20Leopold%20Script.html"
    assert normalize_url("/Movie Scripts/100%25 Wolf Script.html", base="https://imsdb.com/") == \
        "https://imsdb.com/Movie%20Scripts/100%25%20Wolf%20Script.html"  # literal percent survives
