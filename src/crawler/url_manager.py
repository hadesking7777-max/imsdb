"""URL normalization, classification and scope control for IMSDb.

Only an explicit allow-list of page types is ever enqueued, which keeps the crawl finite:
listing pages -> movie detail pages -> screenplay pages. Forms, comment endpoints, search,
TV transcripts, writer pages and external hosts are never followed.
"""

from __future__ import annotations

import re
from enum import StrEnum
from urllib.parse import parse_qsl, quote, unquote, urlencode, urljoin, urlsplit, urlunsplit

from src.utils.hashing import short_hash

CANONICAL_HOST = "imsdb.com"
ALLOWED_HOSTS = frozenset({"imsdb.com", "www.imsdb.com"})
_PATH_SAFE = "/:@!$&'()*+,;=~-._"
_QUERY_SAFE = ":@!$'()*+,;/?~-._"


class PageType(StrEnum):
    """Kind of IMSDb page a URL points to, as decided by ``classify_url``.

    Only ``CRAWLABLE_TYPES`` (listings, movie detail and screenplay pages) are ever enqueued.

    Attributes:
        HOME: Site root or ``/index.html``.
        LISTING_ALL: ``/all-scripts.html``, every movie script on one page.
        LISTING_ALPHA: Per-letter listing, ``/alphabetical/<letter>``.
        LISTING_GENRE: Per-genre listing, ``/genre/<Genre>``.
        LISTING_LANGUAGE: Per-language listing, ``/language/<Language>``.
        MOVIE_DETAIL: Movie detail page, ``/Movie Scripts/<Title> Script.html``.
        SCRIPT_HTML: HTML screenplay page, ``/scripts/<Name>.html``.
        SCRIPT_PDF: PDF screenplay, ``/scripts/<Name>.pdf`` (not crawled).
        WRITER: Writer page, ``/writer.php`` (not crawled).
        TV: TV transcript pages (not crawled).
        POLICY: Disclaimer, contact, privacy, terms and similar pages (matched by ``POLICY_RE``).
        ROBOTS: ``/robots.txt``.
        ASSET: Stylesheets, scripts and images.
        FORM: Other ``.php`` endpoints (search, comments, forms).
        OTHER: Any other IMSDb path.
        EXTERNAL: A URL on a host other than IMSDb.
    """

    HOME = "home"
    LISTING_ALL = "listing_all"
    LISTING_ALPHA = "listing_alpha"
    LISTING_GENRE = "listing_genre"
    LISTING_LANGUAGE = "listing_language"
    MOVIE_DETAIL = "movie_detail"
    SCRIPT_HTML = "script_html"
    SCRIPT_PDF = "script_pdf"
    WRITER = "writer"
    TV = "tv"
    POLICY = "policy"
    ROBOTS = "robots"
    ASSET = "asset"
    FORM = "form"
    OTHER = "other"
    EXTERNAL = "external"


LISTING_TYPES = frozenset({PageType.HOME, PageType.LISTING_ALL, PageType.LISTING_ALPHA,
                           PageType.LISTING_GENRE, PageType.LISTING_LANGUAGE})
RECORD_TYPES = frozenset({PageType.MOVIE_DETAIL, PageType.SCRIPT_HTML})
CRAWLABLE_TYPES = LISTING_TYPES | RECORD_TYPES

# Queue priority: finish started records before starting new ones.
PRIORITY = {
    PageType.SCRIPT_HTML: 100,
    PageType.MOVIE_DETAIL: 50,
    PageType.HOME: 90,
    PageType.LISTING_ALL: 80,
    PageType.LISTING_ALPHA: 70,
    PageType.LISTING_GENRE: 70,
    PageType.LISTING_LANGUAGE: 70,
}

POLICY_RE = re.compile(
    r"^/(?:disclaimer|contact|privacy(?:-policy)?|submit|terms(?:-of-(?:use|service))?|tos|legal|copyright|about"
    r"|link%20to%20us)(?:\.html?|\.php)?/?$",
    re.IGNORECASE,
)

_RULES: list[tuple[re.Pattern[str], PageType]] = [
    (re.compile(r"^/$"), PageType.HOME),
    (re.compile(r"^/index\.html?$", re.I), PageType.HOME),
    (re.compile(r"^/robots\.txt$", re.I), PageType.ROBOTS),
    (re.compile(r"^/all-scripts\.html$", re.I), PageType.LISTING_ALL),
    (re.compile(r"^/alphabetical/[^/]+$", re.I), PageType.LISTING_ALPHA),
    (re.compile(r"^/genre/[^/]+$", re.I), PageType.LISTING_GENRE),
    (re.compile(r"^/language/[^/]+$", re.I), PageType.LISTING_LANGUAGE),
    (re.compile(r"^/Movie%20Scripts/.+\.html$", re.I), PageType.MOVIE_DETAIL),
    (re.compile(r"^/scripts/.+\.pdf$", re.I), PageType.SCRIPT_PDF),
    (re.compile(r"^/scripts/.+\.html?$", re.I), PageType.SCRIPT_HTML),
    (re.compile(r"^/writer\.php$", re.I), PageType.WRITER),
    (re.compile(r"^/(TV%20Transcripts|transcripts|TV)/", re.I), PageType.TV),
    (re.compile(r"\.(css|js|gif|jpe?g|png|ico|svg|webp)$", re.I), PageType.ASSET),
    (re.compile(r"\.php$", re.I), PageType.FORM),
]


def normalize_url(url: str | None, base: str | None = None, *, strip_trailing_slash: bool = True) -> str | None:
    """Return a canonical absolute URL, or ``None`` for non-HTTP / unusable links.

    * resolves relative references against ``base``
    * lower-cases scheme and host, maps ``www.imsdb.com`` -> ``imsdb.com`` and forces https there
    * drops fragments and default ports
    * re-encodes the path consistently (spaces -> ``%20``)
    * removes a trailing slash (except for the root) on IMSDb URLs, unless
      ``strip_trailing_slash=False`` (used for redirect hops, where the server's exact
      target must be requested to avoid ``/x`` -> ``/x/`` -> ``/x`` loops)
    * keeps query parameters in their original order, re-encoded consistently

    Args:
        url: Absolute or relative URL as found in a page; None is accepted and returns None.
        base: URL of the page the link was found on, used to resolve relative references; None when
            ``url`` is already absolute.
        strip_trailing_slash: Whether to drop a trailing slash from IMSDb paths.

    Returns:
        The canonical absolute URL, or None for empty, fragment-only, ``javascript:``/``mailto:``/
        ``tel:``/``data:`` links, non-HTTP schemes, missing hosts and invalid ports.
    """
    if url is None:
        return None
    url = url.strip()
    if not url or url.startswith(("#", "javascript:", "mailto:", "tel:", "data:")):
        return None
    if base:
        url = urljoin(base, url)
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"}:
        return None
    host = (parts.hostname or "").lower().rstrip(".")
    if not host:
        return None
    try:
        port = parts.port
    except ValueError:
        return None
    is_imsdb = host in ALLOWED_HOSTS
    if is_imsdb:
        host = CANONICAL_HOST
        scheme = "https"
        port = None
    netloc = host
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{host}:{port}"

    decoded = unquote(parts.path or "/")
    if is_imsdb:
        # IMSDb publishes some links double-encoded ("Kate %2526 Leopold" next to "Kate & Leopold");
        # decode until stable so both spellings map to one canonical URL.
        for _ in range(3):
            again = unquote(decoded)
            if again == decoded:
                break
            decoded = again
    path = quote(decoded, safe=_PATH_SAFE)
    if not path.startswith("/"):
        path = "/" + path
    path = re.sub(r"/{2,}", "/", path)
    if strip_trailing_slash and is_imsdb and len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/") or "/"

    query = ""
    if parts.query:
        pairs = parse_qsl(parts.query, keep_blank_values=True)
        query = urlencode(pairs, safe=_QUERY_SAFE, quote_via=quote)
    return urlunsplit((scheme, netloc, path, query, ""))


def host_of(url: str) -> str:
    """Extract the host name of a URL.

    Args:
        url: Absolute URL.

    Returns:
        The lower-cased host without port, or ``""`` when the URL has no host.
    """
    return (urlsplit(url).hostname or "").lower()


def is_imsdb_url(url: str) -> bool:
    """Check whether a URL points at IMSDb.

    Args:
        url: Absolute URL.

    Returns:
        True when the host is ``imsdb.com`` or ``www.imsdb.com``.
    """
    return host_of(url) in ALLOWED_HOSTS


def classify_url(url: str) -> PageType:
    """Classify a *normalized* URL.

    Policy pages are checked first, then the ``_RULES`` patterns in order (first match wins).

    Args:
        url: URL as returned by ``normalize_url`` (path percent-encoded, e.g. ``%20`` for spaces).

    Returns:
        ``EXTERNAL`` for non-IMSDb hosts, the matching page type, or ``OTHER`` when nothing matches.
    """
    if not is_imsdb_url(url):
        return PageType.EXTERNAL
    path = urlsplit(url).path
    if POLICY_RE.match(path):
        return PageType.POLICY
    for pattern, page_type in _RULES:
        if pattern.search(path):
            return page_type
    return PageType.OTHER


def is_crawlable(url: str) -> bool:
    """Check whether a URL is in crawl scope.

    Args:
        url: Normalized URL.

    Returns:
        True when the URL is a listing, movie detail or HTML screenplay page (``CRAWLABLE_TYPES``).
    """
    return classify_url(url) in CRAWLABLE_TYPES


def url_id(url: str) -> str:
    """Stable queue identifier for a normalized URL.

    Args:
        url: Normalized URL; the same URL always gives the same id.

    Returns:
        ``"u_"`` followed by a 32-character hash of the URL, used as the queue document ``_id``.
    """
    return "u_" + short_hash(url, length=32)


def detail_title_from_url(url: str) -> str | None:
    """Extract the movie title from a detail-page URL.

    Example: ``/Movie Scripts/Dark Knight, The Script.html`` -> ``Dark Knight, The``.

    Args:
        url: Movie detail URL (percent-encoded or not).

    Returns:
        The decoded title with the trailing `` Script`` removed, or None when the URL is not a
        detail page.
    """
    path = unquote(urlsplit(url).path)
    match = re.match(r"^/Movie Scripts/(?P<title>.+?)(?: Script)?\.html$", path, re.IGNORECASE)
    return match.group("title").strip() if match else None


def script_name_from_url(url: str) -> str | None:
    """Extract the screenplay file name from a script URL.

    Example: ``/scripts/Dark-Knight,-The.html`` -> ``Dark-Knight,-The``.

    Args:
        url: Screenplay URL ending in ``.html``, ``.htm`` or ``.pdf``.

    Returns:
        The decoded name without directory or extension, or None when the URL is not under ``/scripts/``.
    """
    path = unquote(urlsplit(url).path)
    match = re.match(r"^/scripts/(?P<name>.+)\.(?:html?|pdf)$", path, re.IGNORECASE)
    return match.group("name") if match else None


def listing_label(url: str) -> str:
    """Human label of a listing page, e.g. ``genre:Action`` or ``alphabetical:A``.

    Args:
        url: Normalized listing URL.

    Returns:
        ``alphabetical:<x>``, ``genre:<x>``, ``language:<x>`` or ``all-scripts``; for any other page the
        page type's value (e.g. ``home``).
    """
    page_type = classify_url(url)
    tail = unquote(urlsplit(url).path).rstrip("/").rsplit("/", 1)[-1]
    if page_type == PageType.LISTING_ALPHA:
        return f"alphabetical:{tail}"
    if page_type == PageType.LISTING_GENRE:
        return f"genre:{tail}"
    if page_type == PageType.LISTING_LANGUAGE:
        return f"language:{tail}"
    if page_type == PageType.LISTING_ALL:
        return "all-scripts"
    return page_type.value
