"""Search chunks: scene text split into passages small enough to show a language model.

Built from the ``scenes`` collection into ``chat_chunks`` with a MongoDB text index (English stemming).
Rebuilding is incremental: a script is re-chunked only when its content hash or the chunker version changes.
No model is needed, so this takes minutes, not hours.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any, Callable

from pymongo import ASCENDING, TEXT, IndexModel
from pymongo.database import Database

from src.database.models import Collections as C
from src.utils.clock import utcnow
from src.utils.text_processing import display_title

log = logging.getLogger("imsdb.chatbot.chunks")

CHUNKS = "chat_chunks"
CHUNK_STATE = "chat_chunk_state"
CHUNKER_VERSION = "3"
CHUNK_CHARS = 1200  # about 300 tokens: four or five passages fit comfortably in one answer prompt
TEXT_INDEX = "chunk_text"
_SPACES_RE = re.compile(r"[ \t ]+")
_BLANKS_RE = re.compile(r"\n{3,}")
_C1_RE = re.compile(r"[\x80-\x9f]")
# Shooting-script scene numbers around a heading: "99   EXT. TITANIC - DAY   99", "A12 INT. HALL A12".
_LEADING_NUM_RE = re.compile(r"^([A-Z]?\d+[A-Z]{0,2})\s+(?=\S)")


def clean_heading(heading: str) -> str:
    """Drop shooting-script numbers (which differ from our scene ordinal) but keep names like 'STUDIO 54'.

    A leading number is removed, and the same number at the end too ("99 EXT. TITANIC - DAY 99").

    Args:
        heading: Raw scene heading.

    Returns:
        The heading on one line, at most 200 characters.
    """
    text = normalize_scene_text(heading).replace("\n", " ")
    match = _LEADING_NUM_RE.match(text)
    if match:
        number = match.group(1)
        text = text[match.end():]
        if text.endswith(" " + number):
            text = text[: -len(number) - 1]
    return text.strip()[:200]


def _fix_c1(match: re.Match[str]) -> str:
    """Replace a C1 control character with the Windows-1252 character it was meant to be.

    Pages decoded as ISO-8859-1 that were really Windows-1252 leave C1 control characters behind
    (e.g. U+0096 for an en dash). Map them to the intended characters for readers and search.

    Args:
        match: A ``_C1_RE`` match of one character in U+0080..U+009F.

    Returns:
        The cp1252 character, or "" for the five bytes cp1252 leaves undefined.
    """
    try:
        return match.group(0).encode("latin-1").decode("cp1252")
    except UnicodeDecodeError:
        return ""


def normalize_scene_text(text: str) -> str:
    """Drop screenplay indentation (meaningless to a reader model) but keep line structure.

    Also fixes C1 control characters, collapses runs of spaces/tabs and limits blank lines to one.

    Args:
        text: Raw scene text (None is treated as empty).

    Returns:
        The normalized text.
    """
    text = _C1_RE.sub(_fix_c1, text or "")
    lines = [_SPACES_RE.sub(" ", line).strip() for line in text.split("\n")]
    return _BLANKS_RE.sub("\n\n", "\n".join(lines)).strip()


def split_chunks(text: str, limit: int = CHUNK_CHARS) -> list[str]:
    """Split on line boundaries into pieces of at most ``limit`` characters (long lines split on spaces).

    A line longer than ``limit`` is cut at the last space in its second half, or hard at ``limit``.

    Args:
        text: Normalized scene text.
        limit: Maximum characters per piece.

    Returns:
        The non-empty pieces, in order.
    """
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.split("\n"):
        while len(line) > limit:
            cut = line.rfind(" ", 0, limit)
            cut = cut if cut > limit // 2 else limit
            head, line = line[:cut], line[cut:].lstrip()
            if current:
                pieces.append("\n".join(current).strip())
                current, size = [], 0
            pieces.append(head.strip())
        if size + len(line) + 1 > limit and current:
            pieces.append("\n".join(current).strip())
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        pieces.append("\n".join(current).strip())
    return [p for p in pieces if p]


def ensure_chunk_indexes(db: Database) -> None:
    """Create the ``chat_chunks`` indexes if missing.

    ``script_order`` (script, scene, chunk) for positional/character queries, and the weighted English text
    index ``chunk_text`` over text (1), heading (2), characters (3) and film (2).

    Args:
        db: The MongoDB database.
    """
    db[CHUNKS].create_indexes([
        IndexModel([("script_id", ASCENDING), ("scene_number", ASCENDING), ("chunk_index", ASCENDING)],
                   name="script_order"),
        IndexModel([("text", TEXT), ("heading", TEXT), ("characters", TEXT), ("film", TEXT)],
                   weights={"text": 1, "heading": 2, "characters": 3, "film": 2},
                   default_language="english", language_override="text_language", name=TEXT_INDEX),
    ])


def chunk_script(db: Database, script: dict[str, Any], film: str, year: int | None) -> list[dict[str, Any]]:
    """Split every scene of one screenplay (from ``scenes``) into chunk documents. Writes nothing.

    Args:
        db: The MongoDB database.
        script: The ``scripts`` document (needs ``_id`` and ``movie_id``).
        film: Display title stored on each chunk.
        year: Release year stored on each chunk, or None.

    Returns:
        ``chat_chunks`` documents with ``_id`` ``"<scene_id>:kNN"``, ``script_id``, ``movie_id``, ``scene_id``,
        ``scene_number``, ``chunk_index``, ``chunk_count``, ``heading``, ``characters``, ``film``, ``year``,
        ``text`` and ``chars``; scenes with no text are skipped.
    """
    docs: list[dict[str, Any]] = []
    projection = {"scene_number": 1, "heading": 1, "characters": 1, "text": 1, "kind": 1}
    for scene in db[C.SCENES].find({"script_id": script["_id"]}, projection).sort("scene_number", ASCENDING):
        text = normalize_scene_text(scene.get("text") or "")
        if not text:
            continue
        parts = split_chunks(text)
        heading = clean_heading(scene.get("heading") or "") or (
            "Title page" if scene.get("scene_number") == 0 else "")
        for i, part in enumerate(parts):
            docs.append({
                "_id": f"{scene['_id']}:k{i:02d}", "script_id": script["_id"], "movie_id": script["movie_id"],
                "scene_id": scene["_id"], "scene_number": scene.get("scene_number"), "chunk_index": i,
                "chunk_count": len(parts), "heading": heading, "characters": scene.get("characters") or [],
                "film": film, "year": year, "text": part, "chars": len(part),
            })
    return docs


def build_chunks(db: Database, *, rebuild: bool = False, stop_event: threading.Event | None = None,
                 progress: Callable[[int, int, str], None] | None = None) -> dict[str, int]:
    """Create or update ``chat_chunks`` for every successfully parsed screenplay. Idempotent and resumable.

    A script is re-chunked when its ``content_hash`` or ``CHUNKER_VERSION`` differs from its ``chat_chunk_state``
    record; its old chunks are replaced. Chunks of scripts that are no longer parsed are removed unless the run
    was interrupted.

    Args:
        db: The MongoDB database.
        rebuild: Delete all chunks and state first.
        stop_event: When set, stops before the next script (``counts["interrupted"] = 1``).
        progress: Called as ``progress(n, total, film)`` after each re-chunked script.

    Returns:
        Counts: ``scripts``, ``rechunked``, ``unchanged``, ``removed``, ``chunks_written``, ``chunks_total``
        and, if stopped early, ``interrupted``.
    """
    ensure_chunk_indexes(db)
    if rebuild:
        db[CHUNKS].delete_many({})
        db[CHUNK_STATE].delete_many({})
    scripts = list(db[C.SCRIPTS].find({"status": {"$in": ["parsed", "parsed_with_warnings"]}},
                                      {"movie_id": 1, "title": 1, "content_hash": 1}).sort("_id", ASCENDING))
    movies = {m["_id"]: m for m in db[C.MOVIES].find({}, {"display_title": 1, "title": 1, "release_year": 1})}
    state = {s["_id"]: s for s in db[CHUNK_STATE].find({})}
    counts = {"scripts": len(scripts), "rechunked": 0, "unchanged": 0, "removed": 0, "chunks_written": 0}
    for n, script in enumerate(scripts, 1):
        if stop_event is not None and stop_event.is_set():
            counts["interrupted"] = 1
            break
        prior = state.get(script["_id"])
        if prior and prior.get("content_hash") == script.get("content_hash") \
                and prior.get("chunker_version") == CHUNKER_VERSION:
            counts["unchanged"] += 1
            continue
        movie = movies.get(script["movie_id"], {})
        film = movie.get("display_title") or display_title(movie.get("title") or script.get("title") or "")
        docs = chunk_script(db, script, film, movie.get("release_year"))
        db[CHUNKS].delete_many({"script_id": script["_id"]})
        if docs:
            db[CHUNKS].insert_many(docs, ordered=False)
        db[CHUNK_STATE].replace_one({"_id": script["_id"]}, {
            "_id": script["_id"], "content_hash": script.get("content_hash"), "chunker_version": CHUNKER_VERSION,
            "chunks": len(docs), "built_at": utcnow()}, upsert=True)
        counts["rechunked"] += 1
        counts["chunks_written"] += len(docs)
        if progress:
            progress(n, len(scripts), film)
    live = {s["_id"] for s in scripts}
    stale = [sid for sid in state if sid not in live]
    if stale and "interrupted" not in counts:
        db[CHUNKS].delete_many({"script_id": {"$in": stale}})
        db[CHUNK_STATE].delete_many({"_id": {"$in": stale}})
        counts["removed"] = len(stale)
    counts["chunks_total"] = db[CHUNKS].estimated_document_count()
    return counts


def chunk_index_status(db: Database) -> dict[str, Any]:
    """Report how far the chunk index is built.

    Args:
        db: The MongoDB database.

    Returns:
        A dict with ``scripts_parsed``, ``scripts_indexed`` (at the current chunker version), ``chunks``,
        ``text_index`` (whether the text index exists), ``ready`` (index exists and something is indexed)
        and ``complete`` (every parsed script is indexed).
    """
    parsed = db[C.SCRIPTS].count_documents({"status": {"$in": ["parsed", "parsed_with_warnings"]}})
    indexed = db[CHUNK_STATE].count_documents({"chunker_version": CHUNKER_VERSION})
    has_text_index = any(ix["name"] == TEXT_INDEX for ix in db[CHUNKS].list_indexes())
    return {"scripts_parsed": parsed, "scripts_indexed": indexed, "chunks": db[CHUNKS].estimated_document_count(),
            "text_index": has_text_index, "ready": has_text_index and indexed > 0,
            "complete": has_text_index and indexed >= parsed}
