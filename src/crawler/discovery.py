"""Discovery: seed URLs and queue items derived from listing pages.

IMSDb navigation (verified 2026-11-17):

* ``/``                       home page with links to every listing page
* ``/all-scripts.html``       one page listing every movie script (A-Z)
* ``/alphabetical/<0|A..Z>``  per-letter listings
* ``/genre/<Genre>``          per-genre listings (gives genre membership)
* ``/language/<Language>``    international listings
* ``/Movie Scripts/<Title> Script.html``   movie detail page
* ``/scripts/<Name>.html``    screenplay page

There is no pagination: each listing is a single page. Listing pages only ever enqueue
other listing pages and detail pages, so the crawl graph is finite by construction.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from src.crawler.url_manager import (
    PRIORITY,
    PageType,
    classify_url,
    normalize_url,
    url_id,
)
from src.database.models import QueueItem
from src.utils.clock import utcnow

MAX_LISTING_DEPTH = 2  # home (0) -> listing (1) -> listing links found on listings (2)


def make_queue_item(url: str, *, discovered_from: str | None = None, depth: int = 0,
                    movie_id: str | None = None, position: int = 0,
                    now: datetime | None = None) -> QueueItem | None:
    """Build a pending queue item for a URL, due immediately.

    The URL is normalized, classified, and given the priority of its page type (0 when unlisted). Scope
    is not checked here; callers filter on ``page_type``.

    Args:
        url: Absolute URL to enqueue.
        discovered_from: URL of the page the link was found on, or None for seeds.
        depth: Crawl depth of the new item (0 for the home page).
        movie_id: Movie the page belongs to, when already known.
        position: Index of the link on the source page, used to keep listing order.
        now: Timestamp for ``discovered_at``/``next_attempt_at``; current UTC time when None.

    Returns:
        The queue item (not yet persisted), or None when the URL cannot be normalized.
    """
    normalized = normalize_url(url)
    if normalized is None:
        return None
    page_type = classify_url(normalized)
    now = now or utcnow()
    return QueueItem(
        _id=url_id(normalized),
        url=normalized,
        page_type=page_type.value,
        priority=PRIORITY.get(page_type, 0),
        depth=depth,
        position=position,
        discovered_from=discovered_from,
        discovered_at=now,
        next_attempt_at=now,
        movie_id=movie_id,
    )


def seed_items(base_url: str, *, quick: bool = False, now: datetime | None = None) -> list[QueueItem]:
    """Seeds for discovery. ``quick`` uses only the all-scripts page (one request).

    Args:
        base_url: Site root, e.g. ``https://imsdb.com/``.
        quick: Seed only ``all-scripts.html`` instead of the home page plus ``all-scripts.html``.
        now: Timestamp for the items; current UTC time when None.

    Returns:
        Queue items for the home page (depth 0) and/or the all-scripts page (depth 1).
    """
    seeds = ["all-scripts.html"] if quick else ["", "all-scripts.html"]
    items = [make_queue_item(normalize_url(path, base=base_url) or base_url, depth=0 if not path else 1, now=now)
             for path in seeds]
    return [item for item in items if item is not None]


def listing_followups(listing_links: Iterable[str], *, source_url: str, depth: int,
                      now: datetime | None = None) -> list[QueueItem]:
    """Queue items for listing pages linked from a listing page (bounded depth).

    Args:
        listing_links: Links found on the listing page, in page order.
        source_url: URL of the listing page they were found on.
        depth: Depth of the source page; children get ``depth + 1``.
        now: Timestamp for the items; current UTC time when None.

    Returns:
        Items for links that classify as listing pages (all-scripts, alphabetical, genre, language);
        empty when ``depth`` has reached ``MAX_LISTING_DEPTH``.
    """
    if depth >= MAX_LISTING_DEPTH:
        return []
    out: list[QueueItem] = []
    for position, link in enumerate(listing_links):
        item = make_queue_item(link, discovered_from=source_url, depth=depth + 1, position=position, now=now)
        if item and item.page_type in {PageType.LISTING_ALL, PageType.LISTING_ALPHA, PageType.LISTING_GENRE,
                                       PageType.LISTING_LANGUAGE}:
            out.append(item)
    return out


def detail_items(detail_urls: Iterable[tuple[str, str | None]], *, source_url: str, depth: int,
                 now: datetime | None = None) -> list[QueueItem]:
    """Queue items for movie detail pages: ``(url, movie_id)`` pairs.

    Args:
        detail_urls: ``(url, movie_id)`` pairs from a listing page, in page order; ``movie_id`` may be None.
        source_url: URL of the listing page they were found on.
        depth: Depth of the source page; children get ``depth + 1``.
        now: Timestamp for the items; current UTC time when None.

    Returns:
        Items for the URLs that classify as movie detail pages; anything else is dropped.
    """
    out: list[QueueItem] = []
    for position, (url, movie_id) in enumerate(detail_urls):
        item = make_queue_item(url, discovered_from=source_url, depth=depth + 1, movie_id=movie_id,
                               position=position, now=now)
        if item and item.page_type == PageType.MOVIE_DETAIL:
            out.append(item)
    return out
