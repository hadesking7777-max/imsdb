"""Exact answers from MongoDB: film details, characters and their lines, film lists, corpus totals.

These need no language model, so they are instant and always correct with respect to the stored data.
"""

from __future__ import annotations

import difflib
import re
from typing import Any

from pymongo.database import Database

from src.chatbot.catalog import Film, FilmCatalog
from src.database.models import Collections as C
from src.utils.text_processing import slugify

MAX_LINE_CHARS = 240
STATUS_NOTES = {
    "script_unavailable": "IMSDb does not offer this screenplay as a text page (PDF only or no script link).",
    "script_empty_at_source": "IMSDb's script page for this film exists but contains no text.",
    "script_failed": "The screenplay page could not be extracted.",
    "discovered": "Only the listing entry is stored; the detail page was not collected.",
    "detail_collected": "The screenplay has not been collected yet.",
}


def _short(text: str, limit: int = MAX_LINE_CHARS) -> str:
    """Collapse whitespace and truncate text with an ellipsis.

    Args:
        text: Text to shorten (None is treated as empty).
        limit: Maximum length in characters, including the ellipsis.

    Returns:
        The single-line text, at most ``limit`` characters long.
    """
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def film_facts(db: Database, catalog: FilmCatalog, film: Film, *, top_characters: int = 8) -> dict[str, Any]:
    """Collect the stored facts about one film.

    Reads the film's ``movies`` document (dates, draft, IMSDb page, shared-script owner), its ``scripts``
    document (``structure_stats``) and its most talkative ``characters``.

    Args:
        db: The MongoDB database.
        catalog: The film catalogue (used to name the owner of a shared screenplay).
        film: The film.
        top_characters: How many characters to include, by number of dialogue blocks.

    Returns:
        A dict with ``film``, ``year``, ``genres``, ``writers``, ``release_date``, ``script_date``, ``draft``,
        ``imsdb_page``; ``note`` when the screenplay is shared or unavailable; and, when a script is stored,
        ``scenes``, ``dialogue_blocks``, ``speaking_characters``, ``words``, ``pages`` and ``top_characters``
        (list of ``{"name", "lines", "words"}``).
    """
    movie = db[C.MOVIES].find_one({"_id": film.id}, {"script_date": 1, "listing_draft": 1, "movie_release_date": 1,
                                                    "shared_script_owner_id": 1, "detail_url": 1}) or {}
    facts: dict[str, Any] = {
        "film": film.display_title, "year": film.year, "genres": film.genres, "writers": film.writers,
        "release_date": movie.get("movie_release_date"), "script_date": movie.get("script_date"),
        "draft": (movie.get("listing_draft") or {}).get("text"), "imsdb_page": movie.get("detail_url"),
    }
    if film.status == "script_shared":
        owner = catalog.by_id.get(movie.get("shared_script_owner_id", ""))
        facts["note"] = (f"IMSDb links this film to the screenplay of {owner.display_title if owner else 'another film'}; "
                         "the scenes below come from that screenplay.")
    elif film.status in STATUS_NOTES:
        facts["note"] = STATUS_NOTES[film.status]
    if film.script_id:
        script = db[C.SCRIPTS].find_one({"_id": film.script_id}, {"structure_stats": 1, "title": 1}) or {}
        stats = script.get("structure_stats") or {}
        facts.update({"scenes": stats.get("scenes"), "dialogue_blocks": stats.get("dialogue_blocks"),
                      "speaking_characters": stats.get("characters"), "words": stats.get("words"),
                      "pages": stats.get("last_page")})
        facts["top_characters"] = [
            {"name": c["name"], "lines": c.get("dialogue_count", 0), "words": c.get("word_count", 0)}
            for c in db[C.CHARACTERS].find({"script_id": film.script_id}, {"name": 1, "dialogue_count": 1,
                                                                           "word_count": 1})
            .sort([("dialogue_count", -1), ("name", 1)]).limit(top_characters)]
    return facts


def find_character(db: Database, script_id: str, name: str) -> dict[str, Any] | None:
    """Character by name within one script: exact key, then prefix/contains, then fuzzy.

    The partial step matches any shared word of the slugified name (e.g. "JACK" -> "JACK DAWSON"), preferring
    the character with the most dialogue; the fuzzy step uses ``difflib`` with cutoff 0.8.

    Args:
        db: The MongoDB database (``characters`` collection).
        script_id: The screenplay's ``_id``.
        name: Character name as given by the user or planner.

    Returns:
        The full ``characters`` document, or None when nothing matches or the name is empty.
    """
    key = slugify(name or "", max_length=120)
    if not key:
        return None
    col = db[C.CHARACTERS]
    doc = col.find_one({"script_id": script_id, "name_key": key})
    if doc:
        return doc
    rows = list(col.find({"script_id": script_id}, {"name": 1, "name_key": 1, "dialogue_count": 1})
                .sort("dialogue_count", -1))
    parts = set(key.split("-"))
    partial = [r for r in rows if parts & set(r["name_key"].split("-"))]
    if partial:
        return col.find_one({"_id": partial[0]["_id"]})
    close = difflib.get_close_matches(key, [r["name_key"] for r in rows], n=1, cutoff=0.8)
    return col.find_one({"script_id": script_id, "name_key": close[0]}) if close else None


def character_facts(db: Database, film: Film, name: str, *, sample_lines: int = 4) -> dict[str, Any] | None:
    """Collect the stored facts about one character of a film.

    Uses ``find_character``, the ``scenes`` collection for the first appearance and ``dialogues`` (ordered by
    ``order``) for the first non-empty lines.

    Args:
        db: The MongoDB database.
        film: The film whose screenplay is searched.
        name: Character name.
        sample_lines: How many of the character's first lines to include.

    Returns:
        A dict with ``film``, ``character``, ``lines``, ``words``, ``scenes``, ``first_appearance``
        (``{"scene_number", "heading"}`` or None), ``cue_variants`` (up to 6) and ``first_lines``
        (``{"text", "scene_id"}``); None when the film has no script or the character is not found.
    """
    if not film.script_id:
        return None
    char = find_character(db, film.script_id, name)
    if char is None:
        return None
    scene_count = len(char.get("scene_ids") or [])
    first_scene = None
    if char.get("scene_ids"):
        scene = db[C.SCENES].find_one({"_id": char["scene_ids"][0]}, {"heading": 1, "scene_number": 1})
        if scene:
            first_scene = {"scene_number": scene.get("scene_number"), "heading": _short(scene.get("heading") or "", 120)}
    lines = []
    for d in db[C.DIALOGUES].find({"character_id": char["_id"], "text": {"$ne": ""}},
                                  {"text": 1, "scene_id": 1, "order": 1}).sort("order", 1).limit(sample_lines):
        lines.append({"text": _short(d["text"]), "scene_id": d.get("scene_id")})
    return {"film": film.display_title, "character": char["name"], "lines": char.get("dialogue_count", 0),
            "words": char.get("word_count", 0), "scenes": scene_count, "first_appearance": first_scene,
            "cue_variants": (char.get("variants") or [])[:6], "first_lines": lines}


def list_films(catalog: FilmCatalog, *, genre: str | None = None, writer: str | None = None,
               year: int | None = None, limit: int = 40) -> dict[str, Any]:
    """Filter the catalogue by genre, writer and/or year.

    Genre must match exactly (case-insensitive); the writer matches as a substring of slugified writer names.

    Args:
        catalog: The film catalogue.
        genre: Genre name, or None.
        writer: Writer name or part of it, or None.
        year: Release year, or None.
        limit: Maximum number of films listed.

    Returns:
        A dict with ``filters`` (the filters used), ``total``, ``films`` (labels sorted by key, at most
        ``limit``) and ``truncated``.
    """
    films = catalog.films
    if genre:
        g = genre.strip().lower()
        films = [f for f in films if any(x.lower() == g for x in f.genres)]
    if writer:
        w = slugify(writer)
        films = [f for f in films if any(w in slugify(x) for x in f.writers)]
    if year:
        films = [f for f in films if f.year == year]
    films = sorted(films, key=lambda f: f.key)
    return {"filters": {k: v for k, v in (("genre", genre), ("writer", writer), ("year", year)) if v},
            "total": len(films), "films": [f.label() for f in films[:limit]], "truncated": len(films) > limit}


def corpus_overview(db: Database, catalog: FilmCatalog) -> dict[str, Any]:
    """Totals for the whole archive.

    Scene and dialogue counts use ``estimated_document_count`` (fast collection metadata).

    Args:
        db: The MongoDB database.
        catalog: The film catalogue.

    Returns:
        A dict with ``films_listed``, ``screenplays_stored``, ``scenes``, ``dialogue_blocks``, ``genres``
        (name -> film count, largest first) and ``source``.
    """
    genres: dict[str, int] = {}
    for f in catalog.films:
        for g in f.genres:
            genres[g] = genres.get(g, 0) + 1
    return {"films_listed": len(catalog.films),
            "screenplays_stored": sum(1 for f in catalog.films if f.status == "script_collected"),
            "scenes": db[C.SCENES].estimated_document_count(),
            "dialogue_blocks": db[C.DIALOGUES].estimated_document_count(),
            "genres": dict(sorted(genres.items(), key=lambda kv: -kv[1])), "source": "IMSDb (imsdb.com)"}
