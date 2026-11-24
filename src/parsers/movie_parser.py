"""Parsers for IMSDb listing pages and movie detail pages.

Listing entries look like::

    <p><a href="/Movie Scripts/12 Monkeys Script.html" title="12 Monkeys Script">12 Monkeys</a>
       (1994-06 Draft)<br><i>Written by David Peoples,Janet Peoples</i><br></p>

Detail pages contain a ``table.script-details`` whose cell holds ``<b>Label</b>`` blocks
(Writers, Genres, Script Date, Movie Release Date, ratings ...). Labels are parsed
generically so that fields not known today are preserved in ``extra_fields``.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from bs4 import BeautifulSoup, NavigableString, Tag

from src.crawler.url_manager import (
    LISTING_TYPES,
    PageType,
    classify_url,
    detail_title_from_url,
    normalize_url,
)
from src.utils.text_processing import collapse_whitespace

# Leading "(...)" after a listing link, e.g. "(1994-06 Draft)".
_DRAFT_RE = re.compile(r"^\((?P<inner>[^)]*)\)")
# "YYYY[-MM[-DD]] <label>" inside the draft parentheses.
_DRAFT_DATE_RE = re.compile(r"^(?P<date>\d{4}(?:-\d{2}(?:-\d{2})?)?)\s+(?P<label>.+)$")
# Star-rating image such as ".../images/rating/8-stars.gif" or "4.5-stars.gif".
_RATING_IMG_RE = re.compile(r"/(?P<n>\d+(?:\.\d+)?)-stars?\.gif", re.IGNORECASE)
_RATING_TEXT_RE = re.compile(r"(?P<n>\d+(?:\.\d+)?)\s*(?:out of|/)\s*10", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(18[89]\d|19\d\d|20\d\d)\b")
_IMDB_RE = re.compile(r"imdb\.com/title/(tt\d+)", re.IGNORECASE)
_NOT_AVAILABLE_RE = re.compile(r"\b(not available|unavailable|coming soon|no script)\b", re.I)

KNOWN_LABELS = {
    "imsdb opinion": "imsdb_opinion",
    "imsdb rating": "imsdb_rating",
    "average user rating": "average_user_rating",
    "writers": "writers",
    "writer": "writers",
    "genres": "genres",
    "genre": "genres",
    "script date": "script_date",
    "movie release date": "movie_release_date",
    "release date": "movie_release_date",
    "directors": "directors",
    "director": "directors",
}


def _soup(html: str) -> BeautifulSoup:
    """Parse HTML with the lenient lxml backend.

    Args:
        html: Decoded page HTML.

    Returns:
        The parsed document.
    """
    return BeautifulSoup(html, "lxml")


# ====================================================================== listing pages
@dataclass(slots=True)
class ListingEntry:
    """One movie entry (``<p>`` with a detail-page link) on a listing page.

    Attributes:
        title: Link text, or a title derived from the detail URL when the link text is empty.
        detail_url: Normalised absolute URL of the movie detail page.
        position: 0-based position among the page's (deduplicated) entries.
        draft_text: Text inside the parentheses after the link, e.g. ``"1994-06 Draft"``.
        draft_date: Leading ``YYYY[-MM[-DD]]`` of the draft text, or None.
        draft_label: Rest of the draft text, e.g. ``"Draft"``, or None.
        writers: Writer names from the ``<i>Written by ...</i>`` line.
        writers_text: The italic writers line as written.
        link_title_attr: The anchor's ``title`` attribute, or None.
        href: The link exactly as published (provenance for canonicalization).
    """

    title: str
    detail_url: str
    position: int
    draft_text: str | None = None
    draft_date: str | None = None
    draft_label: str | None = None
    writers: list[str] = field(default_factory=list)
    writers_text: str | None = None
    link_title_attr: str | None = None
    href: str | None = None  # the link exactly as published (provenance for canonicalization)

    def to_dict(self) -> dict[str, Any]:
        """Convert the entry to a plain dict for storage.

        Returns:
            All fields as a dict.
        """
        return asdict(self)


@dataclass(slots=True)
class ListingPage:
    """Everything extracted from one listing page (alphabetical, genre or similar index).

    Attributes:
        url: The page URL as passed to the parser.
        page_type: ``PageType`` value of the page URL.
        heading: Text of the listing's ``<h1>``, or None.
        entries: Movie entries in page order, without duplicates.
        listing_links: Other listing pages linked anywhere on the page (home page excluded), deduplicated.
        sidebar_movie_links: Detail-page links that are not entries (e.g. sidebar "top" lists).
        warnings: Problems found, e.g. duplicate entries or no entries at all.
    """

    url: str
    page_type: str
    heading: str | None
    entries: list[ListingEntry]
    listing_links: list[str]
    sidebar_movie_links: list[str]
    warnings: list[str] = field(default_factory=list)


def parse_draft(text: str | None) -> tuple[str | None, str | None, str | None]:
    """Parse listing draft info: ``"(1994-06 Draft)"`` -> ``("1994-06 Draft", "1994-06", "Draft")``.

    Only the first parenthesised group at the start of the text is used. ``"(Undated Draft)"`` gives
    ``("Undated Draft", None, "Draft")``; any other undated text becomes the label.

    Args:
        text: Text following the listing link, or None.

    Returns:
        ``(inner_text, date, label)``; all None when ``text`` is empty or does not start with ``(``.
    """
    if not text:
        return None, None, None
    match = _DRAFT_RE.match(text.strip())
    if not match:
        return None, None, None
    inner = collapse_whitespace(match.group("inner"))
    dated = _DRAFT_DATE_RE.match(inner)
    if dated:
        return inner, dated.group("date"), dated.group("label")
    if inner.lower().startswith("undated"):
        return inner, None, inner[len("undated"):].strip() or None
    return inner, None, inner or None


def split_people(text: str | None) -> list[str]:
    """Split a list of names on commas, `` & `` and `` and ``.

    Args:
        text: E.g. ``"David Peoples,Janet Peoples"`` or ``"Joel Coen & Ethan Coen"``, or None.

    Returns:
        Whitespace-normalised non-empty names in order; empty for empty input.
    """
    if not text:
        return []
    names = [collapse_whitespace(part) for part in re.split(r",|\s+&\s+|\s+and\s+", text)]
    return [name for name in names if name]


def _entry_paragraph_count(container: Tag | BeautifulSoup, base: str) -> int:
    """Count ``<p>`` elements under a container whose first link points to a movie detail page.

    Args:
        container: Element (or whole document) to search.
        base: Page URL used to resolve relative links.

    Returns:
        The number of listing-entry paragraphs.
    """
    count = 0
    for paragraph in container.find_all("p"):
        anchor = paragraph.find("a", href=True)
        target = normalize_url(anchor["href"], base=base) if anchor else None
        if target and classify_url(target) == PageType.MOVIE_DETAIL:
            count += 1
    return count


def parse_listing_page(html: str, url: str) -> ListingPage:
    """Parse an IMSDb listing page into movie entries and navigation links.

    Entries are read from the ``<p>`` paragraphs (see module docstring) inside the parent of the
    ``<h1>`` whose container holds the most entry paragraphs; without one, the whole document is
    scanned. Each entry's draft info is the text right after the link up to the first ``<br>``/``<i>``,
    and its writers come from the paragraph's ``<i>`` (``Written by`` prefix removed).

    Args:
        html: Decoded page HTML.
        url: Page URL, used to resolve relative links and to classify the page.

    Returns:
        The parsed page. Unparseable or empty pages give no entries and a ``"no listing entries found"``
        warning (except for the home page).
    """
    soup = _soup(html)
    warnings: list[str] = []
    # Live IMSDb pages start with a site-banner <h1>; the listing's own <h1> sits in the content cell.
    # Choose the heading whose container actually holds movie-entry paragraphs.
    best: tuple[int, Tag | None] = (0, None)
    for h1 in soup.find_all("h1"):
        if isinstance(h1.parent, Tag):
            count = _entry_paragraph_count(h1.parent, url)
            if count > best[0]:
                best = (count, h1)
    heading_tag = best[1]
    if heading_tag is None:
        headings = soup.find_all("h1")
        heading_tag = headings[-1] if headings else None
    heading = collapse_whitespace(heading_tag.get_text(" ")) if heading_tag else None
    container: Tag | BeautifulSoup = soup
    if best[1] is not None and isinstance(best[1].parent, Tag):
        container = best[1].parent
    elif _entry_paragraph_count(soup, url):
        warnings.append("listing entries found outside any <h1> container; scanned whole document")

    entries: list[ListingEntry] = []
    seen: set[str] = set()
    for paragraph in container.find_all("p"):
        anchor = paragraph.find("a", href=True)
        if anchor is None:
            continue
        detail_url = normalize_url(anchor["href"], base=url)
        if not detail_url or classify_url(detail_url) != PageType.MOVIE_DETAIL:
            continue
        if detail_url in seen:
            warnings.append(f"duplicate listing entry for {detail_url}")
            continue
        seen.add(detail_url)
        title = collapse_whitespace(anchor.get_text(" ")) or detail_title_from_url(detail_url) or ""
        # Text immediately following the anchor holds "(draft info)".
        trailing = []
        for sibling in anchor.next_siblings:
            if isinstance(sibling, Tag) and sibling.name in {"br", "i"}:
                break
            trailing.append(sibling.get_text(" ") if isinstance(sibling, Tag) else str(sibling))
        draft_text, draft_date, draft_label = parse_draft(collapse_whitespace("".join(trailing)))
        italic = paragraph.find("i")
        writers_text = collapse_whitespace(italic.get_text(" ")) if italic else None
        writers = []
        if writers_text:
            writers = split_people(re.sub(r"^written by\s*", "", writers_text, flags=re.I))
        entries.append(ListingEntry(
            title=title,
            detail_url=detail_url,
            position=len(entries),
            draft_text=draft_text,
            draft_date=draft_date,
            draft_label=draft_label,
            writers=writers,
            writers_text=writers_text,
            link_title_attr=anchor.get("title"),
            href=anchor["href"],
        ))

    listing_links: list[str] = []
    sidebar: list[str] = []
    for anchor in soup.find_all("a", href=True):
        target = normalize_url(anchor["href"], base=url)
        if not target:
            continue
        page_type = classify_url(target)
        if page_type in LISTING_TYPES and page_type != PageType.HOME and target not in listing_links:
            listing_links.append(target)
        elif page_type == PageType.MOVIE_DETAIL and target not in seen and target not in sidebar:
            sidebar.append(target)

    page_type = classify_url(normalize_url(url) or url)
    if not entries and page_type not in {PageType.HOME}:
        warnings.append("no listing entries found")
    return ListingPage(url=url, page_type=page_type.value, heading=heading, entries=entries,
                       listing_links=listing_links, sidebar_movie_links=sidebar, warnings=warnings)


# ====================================================================== detail pages
@dataclass(slots=True)
class PersonLink:
    """A person (writer, director) named on a detail page.

    Attributes:
        name: Display name.
        url: Normalised link to the person's page, or None when only plain text was given.
    """

    name: str
    url: str | None = None


@dataclass(slots=True)
class MovieDetail:
    """Metadata extracted from a movie detail page (``table.script-details``).

    Attributes:
        url: The page URL as passed to the parser.
        title: Movie title from the ``<h1>`` (trailing "Script" removed) or, failing that, from the URL.
        writers: Writers, preferably from ``writer.php`` links.
        genres: Genre names, preferably from ``/genre/`` links.
        directors: Directors (links or plain names).
        script_date: Text of the "Script Date" field, e.g. ``"June 1994"``.
        movie_release_date: Text of the "Movie Release Date" field.
        release_year: Year found in ``movie_release_date`` (1880-2099), or None.
        imsdb_opinion: The IMSDb opinion text; None when absent or "None available".
        imsdb_rating: IMSDb rating (out of 10) from a star image or "x out of 10" text, or None.
        imsdb_rating_text: Star image URL or the raw rating text.
        average_user_rating: Average user rating (out of 10), or None.
        average_user_rating_text: Star image URL or the raw rating text.
        poster_url: Normalised poster image URL, or None.
        script_url: Normalised URL of the script (HTML or PDF), or None.
        script_link_text: Text of the script link, e.g. ``'Read "12 Monkeys" Script'``.
        script_availability: ``"html"``, ``"pdf"``, ``"missing"`` or ``"unknown"``.
        imdb_ids: Sorted distinct IMDb title ids (``tt...``) linked anywhere on the page.
        description: Free-text description; not filled by :func:`parse_movie_detail`.
        user_comment_count: Number of user comments (rating images in ``table.script-comments``).
        page_meta: ``<title>``, meta description and keywords, when present.
        extra_fields: Unknown labels mapped to ``{"text", "full_text", "links"}``.
        warnings: Problems found while parsing.
    """

    url: str
    title: str | None
    writers: list[PersonLink] = field(default_factory=list)
    genres: list[str] = field(default_factory=list)
    directors: list[PersonLink] = field(default_factory=list)
    script_date: str | None = None
    movie_release_date: str | None = None
    release_year: int | None = None
    imsdb_opinion: str | None = None
    imsdb_rating: float | None = None
    imsdb_rating_text: str | None = None
    average_user_rating: float | None = None
    average_user_rating_text: str | None = None
    poster_url: str | None = None
    script_url: str | None = None
    script_link_text: str | None = None
    script_availability: str = "unknown"  # html | pdf | missing | unknown
    imdb_ids: list[str] = field(default_factory=list)
    description: str | None = None
    user_comment_count: int = 0
    page_meta: dict[str, str] = field(default_factory=dict)
    extra_fields: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert the detail record to a plain dict for storage.

        Returns:
            All fields as a dict; person links become nested dicts.
        """
        return asdict(self)


def _rating_from(nodes: list[Any]) -> tuple[float | None, str | None]:
    """Read a rating from a label block: a ``N-stars.gif`` image wins, else ``"N out of 10"``/``"N/10"`` text.

    Script, ad (``<ins>``), style and noscript nodes are ignored.

    Args:
        nodes: The nodes following a rating label (tags and text nodes).

    Returns:
        ``(rating, source)`` where source is the image ``src`` or the block text. When no rating is found
        the rating is None and the text (or None if empty) is still returned.
    """
    text_parts: list[str] = []
    for node in nodes:
        if isinstance(node, Tag) and node.name in {"script", "ins", "style", "noscript"}:
            continue
        if isinstance(node, Tag):
            imgs = [node] if node.name == "img" else node.find_all("img")
            for img in imgs:
                match = _RATING_IMG_RE.search(img.get("src", ""))
                if match:
                    return float(match.group("n")), img.get("src")
            text_parts.append(node.get_text(" "))
        else:
            text_parts.append(str(node))
    text = collapse_whitespace(" ".join(text_parts)).lstrip(":").strip()
    match = _RATING_TEXT_RE.search(text)
    if match:
        return float(match.group("n")), text
    return None, (text or None)


def _collect_label_blocks(cell: Tag) -> list[tuple[str, list[Any]]]:
    """Split a details cell into ``(label, nodes)`` pairs at each top-level ``<b>``.

    ``<strong>`` counts as a label too; empty labels are treated as content. Nodes before the first
    label are dropped. A trailing ``:`` is removed from labels.

    Args:
        cell: The ``<td>`` holding the ``<b>Label</b> value`` sequence.

    Returns:
        ``(label, nodes)`` pairs in document order; empty when the cell has no labels.
    """
    blocks: list[tuple[str, list[Any]]] = []
    current_label: str | None = None
    current: list[Any] = []
    for node in cell.children:
        if isinstance(node, Tag) and node.name in {"b", "strong"}:
            label = collapse_whitespace(node.get_text(" ")).rstrip(":").strip()
            if label:
                if current_label is not None:
                    blocks.append((current_label, current))
                current_label, current = label, []
                continue
        if current_label is not None:
            current.append(node)
    if current_label is not None:
        blocks.append((current_label, current))
    return blocks


def _is_script_link(node: Any) -> bool:
    """Check whether a node is the "Read ... Script" anchor.

    Args:
        node: Any BeautifulSoup node.

    Returns:
        True for an ``<a>`` whose ``href`` contains ``/scripts/``.
    """
    return isinstance(node, Tag) and node.name == "a" and "/scripts/" in (node.get("href") or "")


def _text_of(nodes: list[Any], *, first_line: bool = False) -> str:
    """Text of a label block. ``first_line`` stops at the first ``<br>`` after content
    (scalar fields such as ``Script Date : June 1994``).

    Script/ad/style nodes are skipped and reading always stops at the script link, so the
    "Read ... Script" anchor never leaks into a field value.

    Args:
        nodes: The nodes following a label.
        first_line: Stop at the first ``<br>`` once some text has been collected.

    Returns:
        Whitespace-collapsed text with a leading ``:`` removed; ``""`` when the block is empty.
    """
    parts: list[str] = []
    for node in nodes:
        if isinstance(node, Tag) and node.name in {"script", "ins", "style", "noscript"}:
            continue
        if _is_script_link(node):
            break
        if isinstance(node, Tag) and node.name == "br":
            if first_line and collapse_whitespace(" ".join(parts)).lstrip(":").strip():
                break
            parts.append(" ")
            continue
        parts.append(node.get_text(" ") if isinstance(node, Tag) else str(node))
    return collapse_whitespace(" ".join(parts)).lstrip(":").strip()


def _links_of(nodes: list[Any], base: str) -> list[PersonLink]:
    """Collect the named links in a label block.

    Args:
        nodes: The nodes following a label; anchors may be the nodes themselves or nested in them.
        base: Page URL used to resolve relative links.

    Returns:
        One ``PersonLink`` per anchor with non-empty text, in document order (URL normalised).
    """
    links: list[PersonLink] = []
    for node in nodes:
        if not isinstance(node, Tag):
            continue
        anchors = [node] if node.name == "a" else node.find_all("a", href=True)
        for anchor in anchors:
            name = collapse_whitespace(anchor.get_text(" "))
            if name:
                links.append(PersonLink(name=name, url=normalize_url(anchor.get("href"), base=base)))
    return links


def _writer_name_from_url(url: str | None) -> str | None:
    """Recover a writer's name from the ``w`` query parameter of a ``writer.php?w=...`` URL.

    Args:
        url: The writer link, or None.

    Returns:
        The URL-decoded name, or None when the URL or parameter is missing.
    """
    if not url:
        return None
    query = parse_qs(urlsplit(url).query)
    return unquote(query.get("w", [""])[0]) or None


def parse_movie_detail(html: str, url: str) -> MovieDetail:
    """Parse an IMSDb movie detail page.

    Fields are read from the ``table.script-details`` cell that contains ``<b>`` labels and links to
    genres, writers or scripts; known labels (``KNOWN_LABELS``) fill typed fields and all others go to
    ``extra_fields``. The script link is the first link to a script HTML/PDF page, searched in the table
    first and then page-wide. Missing pieces are recorded as warnings rather than raised.

    Args:
        html: Decoded page HTML.
        url: Page URL, used to resolve links and as a title fallback.

    Returns:
        The extracted metadata. For unparseable HTML most fields stay empty, the title comes from the
        URL, ``script_availability`` is ``"missing"`` and ``warnings`` explains what was not found.
    """
    soup = _soup(html)
    detail = MovieDetail(url=url, title=None)

    if soup.title:
        detail.page_meta["title"] = collapse_whitespace(soup.title.get_text(" "))
    for name in ("description", "keywords"):
        meta = soup.find("meta", attrs={"name": name})
        if meta and meta.get("content"):
            detail.page_meta[name] = collapse_whitespace(meta["content"])

    table = soup.find("table", class_="script-details")
    if table is None:
        detail.warnings.append("script-details table not found; using page-wide fallbacks")
        scope: Tag | BeautifulSoup = soup
    else:
        scope = table

    h1 = scope.find("h1") or soup.find("h1")
    if h1:
        # First text node only: an unclosed <h1> on malformed pages would otherwise swallow the page.
        heading = collapse_whitespace(next(h1.stripped_strings, ""))
        detail.title = re.sub(r"\s+Script$", "", heading, flags=re.IGNORECASE) or None
    if not detail.title:
        detail.title = detail_title_from_url(url)
        detail.warnings.append("title taken from URL")

    poster = scope.find("img", class_="avimg") or scope.find("img", src=re.compile(r"/posters/"))
    if poster and poster.get("src"):
        detail.poster_url = normalize_url(poster["src"], base=url)

    cell: Tag | None = None
    if table is not None:
        for td in table.find_all("td"):
            if td.find("b") and (td.find("a", href=re.compile(r"/genre/|writer\.php|/scripts/"))):
                cell = td
                break
    if cell is None:
        detail.warnings.append("details cell not found")
    else:
        for label, nodes in _collect_label_blocks(cell):
            key = KNOWN_LABELS.get(label.lower())
            if key == "writers":
                detail.writers = [link for link in _links_of(nodes, url)
                                  if link.url and "writer.php" in link.url] or \
                                 [PersonLink(n) for n in split_people(_text_of(nodes))]
            elif key == "genres":
                detail.genres = [link.name for link in _links_of(nodes, url)
                                 if link.url and "/genre/" in link.url] or split_people(_text_of(nodes))
            elif key == "directors":
                detail.directors = _links_of(nodes, url) or [PersonLink(n) for n in split_people(_text_of(nodes))]
            elif key == "script_date":
                detail.script_date = _text_of(nodes, first_line=True) or None
            elif key == "movie_release_date":
                detail.movie_release_date = _text_of(nodes, first_line=True) or None
            elif key == "imsdb_opinion":
                text = _text_of(nodes, first_line=True)
                detail.imsdb_opinion = None if not text or text.lower() == "none available" else text
            elif key == "imsdb_rating":
                detail.imsdb_rating, detail.imsdb_rating_text = _rating_from(nodes)
            elif key == "average_user_rating":
                detail.average_user_rating, detail.average_user_rating_text = _rating_from(nodes)
            else:
                detail.extra_fields[label] = {
                    "text": _text_of(nodes, first_line=True),
                    "full_text": _text_of(nodes),
                    "links": [asdict(link) for link in _links_of([n for n in nodes if not _is_script_link(n)], url)],
                }

    # Script link: "Read "X" Script" -> /scripts/X.html (or .pdf)
    script_anchor = None
    for anchor in scope.find_all("a", href=True):
        target = normalize_url(anchor["href"], base=url)
        if target and classify_url(target) in {PageType.SCRIPT_HTML, PageType.SCRIPT_PDF}:
            script_anchor = (anchor, target)
            break
    if script_anchor is None:
        for anchor in soup.find_all("a", href=True):
            target = normalize_url(anchor["href"], base=url)
            if target and classify_url(target) in {PageType.SCRIPT_HTML, PageType.SCRIPT_PDF}:
                script_anchor = (anchor, target)
                detail.warnings.append("script link found outside details table")
                break
    if script_anchor:
        anchor, target = script_anchor
        detail.script_url = target
        detail.script_link_text = collapse_whitespace(anchor.get_text(" "))
        detail.script_availability = "pdf" if classify_url(target) == PageType.SCRIPT_PDF else "html"
    else:
        cell_text = cell.get_text(" ") if cell else ""
        detail.script_availability = "missing"
        detail.warnings.append(
            "no script link found" + (" (page says not available)" if _NOT_AVAILABLE_RE.search(cell_text) else "")
        )

    # Dates -> year. Release date preferred; script date is a draft date, not a release.
    if detail.movie_release_date:
        match = _YEAR_RE.search(detail.movie_release_date)
        detail.release_year = int(match.group(1)) if match else None

    imdb_ids = sorted({m.group(1) for a in soup.find_all("a", href=True)
                       for m in [_IMDB_RE.search(a["href"])] if m})
    detail.imdb_ids = imdb_ids

    comments = soup.find("table", class_="script-comments")
    if comments:
        detail.user_comment_count = len(comments.find_all("img", src=re.compile(r"/images/rating/")))

    if not detail.writers:
        detail.warnings.append("no writers listed")
    if not detail.genres:
        detail.warnings.append("no genres listed")
    for link in detail.writers:
        if not link.name and link.url:
            link.name = _writer_name_from_url(link.url) or ""
    return detail
