"""In-memory film catalogue and title resolution (English titles, IMSDb's "X, The" form, Chinese aliases)."""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pymongo.database import Database

from src.database.models import Collections as C
from src.utils.text_processing import display_title, title_key

ALIAS_FILE = Path(__file__).parent / "data" / "title_aliases.json"
_CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_BRACKETS_RE = re.compile(r"^[\s《〈「『\"'“”‘’]+|[\s》〉」』\"'“”‘’]+$")
_YEAR_RE = re.compile(r"\s*\((\d{4})\)\s*$")


@dataclass
class Film:
    """One film from the ``movies`` collection, as held in the catalogue.

    Attributes:
        id: The movie document's ``_id``.
        title: IMSDb title as stored (may use the "X, The" form).
        display_title: Title for display ("The X").
        key: ``title_key`` of the display title; the main lookup key.
        year: Release year, or None.
        genres: Genre names (detail page genres, else listing genres).
        writers: Writer names (detail page writers, else listing writers).
        status: Collection status, e.g. ``"script_collected"``, ``"script_shared"``, ``"script_unavailable"``.
        script_id: ``_id`` of the screenplay in ``scripts``, or None.
        detail_url: The film's IMSDb detail page.
        aliases: Chinese/alternative titles from the alias file that map to this film.
    """

    id: str
    title: str
    display_title: str
    key: str
    year: int | None
    genres: list[str]
    writers: list[str]
    status: str
    script_id: str | None
    detail_url: str
    aliases: list[str] = field(default_factory=list)

    @property
    def has_script(self) -> bool:
        """Whether the screenplay text (its own or a shared one) is stored.

        Returns:
            True when the status is ``script_collected`` or ``script_shared`` and ``script_id`` is set.
        """
        return self.status in {"script_collected", "script_shared"} and bool(self.script_id)

    def label(self) -> str:
        """Display label with the year.

        Returns:
            ``"Title (year)"``, or just the title when the year is unknown.
        """
        return f"{self.display_title} ({self.year})" if self.year else self.display_title


def _clean(name: str) -> str:
    """Strip whitespace and surrounding quotes/title brackets (《》, 「」, quotes) from a title.

    Args:
        name: Title as written by the user or the planner.

    Returns:
        The title without the enclosing punctuation.
    """
    return _BRACKETS_RE.sub("", name.strip())


def alias_key(name: str) -> str:
    """Key for alias lookup: Chinese text is kept (whitespace/brackets removed); Latin text is slugified.

    Args:
        name: A title in English or Chinese.

    Returns:
        For text containing CJK characters, the text without spaces and punctuation; otherwise ``title_key``.
    """
    name = _clean(name)
    if _CJK_RE.search(name):
        return re.sub(r"[\s·・:：,，.。!！?？\-—]+", "", name)
    return title_key(name)


class FilmCatalog:
    """All discovered films, loaded once. ~1,300 entries, so lookups are done in memory."""

    def __init__(self, films: list[Film], aliases: dict[str, str] | None = None) -> None:
        """Index films by id and title key, and attach aliases.

        Each film is indexed under both its display-title key and its raw-title key. Sets ``film.aliases``
        on the given Film objects.

        Args:
            films: All films.
            aliases: Alias (e.g. a Chinese title) -> English title, or None.
        """
        self.films = films
        self.by_id = {f.id: f for f in films}
        self.by_key: dict[str, list[Film]] = {}
        for film in films:
            for key in {film.key, title_key(film.title)}:
                self.by_key.setdefault(key, []).append(film)
        self.alias_to_key: dict[str, str] = {}
        for alias, english in (aliases or {}).items():
            self.alias_to_key[alias_key(alias)] = title_key(english)
        for key, group in self.by_key.items():
            for film in group:
                film.aliases = sorted({a for a, k in (aliases or {}).items() if title_key(k) == key})
        self._keys = list(self.by_key)

    @classmethod
    def load(cls, db: Database, alias_file: Path | None = ALIAS_FILE) -> "FilmCatalog":
        """Build the catalogue from every document in the ``movies`` collection.

        Detail-page ``genres``/``writers`` are preferred over the ``listing_*`` fields. Alias file keys starting
        with ``_`` (comments) are ignored.

        Args:
            db: The MongoDB database.
            alias_file: JSON file of alias -> English title (``data/title_aliases.json``); None or a missing
                file means no aliases.

        Returns:
            The loaded catalogue.
        """
        films = []
        projection = {"title": 1, "display_title": 1, "release_year": 1, "genres": 1, "listing_genres": 1,
                      "writers": 1, "listing_writers": 1, "status": 1, "script_id": 1, "detail_url": 1}
        for doc in db[C.MOVIES].find({}, projection):
            title = doc.get("title") or ""
            shown = doc.get("display_title") or display_title(title)
            writers = [w["name"] for w in doc.get("writers") or [] if w.get("name")] or doc.get("listing_writers") or []
            films.append(Film(id=doc["_id"], title=title, display_title=shown, key=title_key(shown),
                              year=doc.get("release_year"), genres=doc.get("genres") or doc.get("listing_genres") or [],
                              writers=writers, status=doc.get("status", ""), script_id=doc.get("script_id"),
                              detail_url=doc.get("detail_url", "")))
        aliases: dict[str, str] = {}
        if alias_file is not None and alias_file.exists():
            raw = json.loads(alias_file.read_text(encoding="utf-8"))
            aliases = {k: v for k, v in raw.items() if not k.startswith("_")}
        return cls(films, aliases)

    def resolve(self, name: str, *, year: int | None = None, fuzzy: bool = True) -> list[Film]:
        """Films matching a title as written by a user or the planner. Best matches first; [] if none.

        Tries, in order: exact key, alias, with/without a leading article, franchise prefix ("Star Wars" ->
        "Star Wars: ..."), then fuzzy matching (Latin titles only). A trailing "(1999)" in the name is used
        as the year, which narrows multiple candidates.

        Args:
            name: Title in English or Chinese, possibly in brackets or with a year.
            year: Release year used to break ties, or None.
            fuzzy: Allow ``difflib`` close matches (cutoff 0.88) as a last resort.

        Returns:
            Matching films, those with a stored screenplay first, then by display title.
        """
        name = _clean(name or "")
        if not name:
            return []
        match = _YEAR_RE.search(name)
        if match:
            year = year or int(match.group(1))
            name = name[:match.start()]
        key = alias_key(name)
        candidates = list(self.by_key.get(key, []))
        if not candidates and key in self.alias_to_key:
            key = self.alias_to_key[key]
            candidates = list(self.by_key.get(key, []))
        if not candidates and not _CJK_RE.search(name):
            # IMSDb sometimes drops or keeps the article ("Godfather" vs "The Godfather").
            bare = re.sub(r"^(the|a|an)-", "", key)
            for variant in {bare, f"the-{bare}"} - {key}:
                candidates.extend(self.by_key.get(variant, []))
        if not candidates and not _CJK_RE.search(name) and len(key) >= 4:
            # Franchise titles: "Star Wars" -> "Star Wars: A New Hope", ...
            candidates = [f for f in self.films if f.key.startswith(key + "-")]
        if not candidates and fuzzy and not _CJK_RE.search(name):
            for close in difflib.get_close_matches(key, self._keys, n=3, cutoff=0.88):
                candidates.extend(self.by_key[close])
        if year and len(candidates) > 1:
            exact = [f for f in candidates if f.year == year]
            candidates = exact or candidates
        # Prefer films whose screenplay is stored.
        return sorted(candidates, key=lambda f: (not f.has_script, f.display_title))

    def search_titles(self, text: str, limit: int = 10) -> list[Film]:
        """Loose title search for listing questions (substring on normalized keys).

        Args:
            text: Part of a title.
            limit: Maximum number of films returned.

        Returns:
            Films whose key contains the text's key, shortest keys first.
        """
        key = title_key(text)
        hits = [f for f in self.films if key and key in f.key]
        return sorted(hits, key=lambda f: (len(f.key), f.key))[:limit]

    def to_summary(self, film: Film) -> dict[str, Any]:
        """Small JSON-serializable description of a film (sent to the web UI and CLI).

        Args:
            film: The film.

        Returns:
            A dict with ``id``, ``title``, ``year``, ``genres``, ``writers``, ``has_script`` and ``status``.
        """
        return {"id": film.id, "title": film.display_title, "year": film.year, "genres": film.genres,
                "writers": film.writers, "has_script": film.has_script, "status": film.status}
