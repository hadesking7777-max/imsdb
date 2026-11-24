"""Collection creation, server-side validators and indexes (idempotent)."""

from __future__ import annotations

from typing import Any

from pymongo import ASCENDING, DESCENDING, IndexModel
from pymongo.database import Database
from pymongo.errors import CollectionInvalid, OperationFailure

from src.database.models import Collections as C
from src.utils.logging import get_logger

log = get_logger("indexes")

# Unique constraints are the backbone of idempotency: re-running ingestion can only
# update existing documents, never duplicate them.
INDEXES: dict[str, list[IndexModel]] = {
    C.MOVIES: [
        IndexModel([("detail_url", ASCENDING)], unique=True, name="uniq_detail_url"),
        IndexModel([("title_key", ASCENDING)], name="title_key"),
        IndexModel([("status", ASCENDING)], name="status"),
        IndexModel([("genres", ASCENDING)], name="genres"),
        IndexModel([("writers.name", ASCENDING)], name="writer_names"),
        IndexModel([("release_year", ASCENDING)], name="release_year"),
        IndexModel([("script_url", ASCENDING)], name="script_url", sparse=True),
        IndexModel([("last_seen_at", DESCENDING)], name="last_seen"),
    ],
    C.SCRIPTS: [
        IndexModel([("script_url", ASCENDING)], unique=True, name="uniq_script_url"),
        IndexModel([("movie_id", ASCENDING)], name="movie_id"),
        IndexModel([("content_hash", ASCENDING)], name="content_hash"),
        IndexModel([("status", ASCENDING)], name="status"),
        IndexModel([("parsed_at", DESCENDING)], name="parsed_at"),
    ],
    C.SCRIPT_VERSIONS: [
        IndexModel([("script_id", ASCENDING), ("version", ASCENDING)], unique=True, name="uniq_script_version"),
        IndexModel([("content_hash", ASCENDING)], name="content_hash"),
    ],
    C.SCENES: [
        IndexModel([("script_id", ASCENDING), ("scene_number", ASCENDING)], unique=True, name="uniq_script_scene"),
        IndexModel([("movie_id", ASCENDING)], name="movie_id"),
        IndexModel([("script_version_hash", ASCENDING)], name="script_version"),
    ],
    C.DIALOGUES: [
        IndexModel([("script_id", ASCENDING), ("order", ASCENDING)], unique=True, name="uniq_script_order"),
        IndexModel([("scene_id", ASCENDING), ("order_in_scene", ASCENDING)], name="scene_order"),
        IndexModel([("character_id", ASCENDING)], name="character_id"),
        IndexModel([("movie_id", ASCENDING)], name="movie_id"),
        IndexModel([("speaker", ASCENDING)], name="speaker"),
        IndexModel([("script_version_hash", ASCENDING)], name="script_version"),
    ],
    C.CHARACTERS: [
        IndexModel([("script_id", ASCENDING), ("name_key", ASCENDING)], unique=True, name="uniq_script_character"),
        IndexModel([("movie_id", ASCENDING)], name="movie_id"),
        IndexModel([("name", ASCENDING)], name="name"),
    ],
    C.CRAWL_QUEUE: [
        IndexModel([("url", ASCENDING)], unique=True, name="uniq_url"),
        IndexModel([("status", ASCENDING), ("page_type", ASCENDING), ("priority", DESCENDING),
                    ("next_attempt_at", ASCENDING)], name="claim_order"),
        IndexModel([("status", ASCENDING), ("lease_expires_at", ASCENDING)], name="leases"),
        IndexModel([("movie_id", ASCENDING)], name="movie_id", sparse=True),
        IndexModel([("page_type", ASCENDING), ("completed_at", ASCENDING)], name="refresh"),
    ],
    C.CRAWL_LOGS: [
        IndexModel([("ts", DESCENDING)], name="ts"),
        IndexModel([("job_id", ASCENDING), ("ts", ASCENDING)], name="job_ts"),
        IndexModel([("event", ASCENDING), ("ts", DESCENDING)], name="event_ts"),
        IndexModel([("level", ASCENDING)], name="level"),
    ],
    C.CRAWL_JOBS: [
        IndexModel([("status", ASCENDING), ("started_at", DESCENDING)], name="status_started"),
    ],
    C.RAW_ARTIFACTS: [
        IndexModel([("url", ASCENDING), ("sha256", ASCENDING)], unique=True, name="uniq_url_sha"),
        IndexModel([("sha256", ASCENDING)], name="sha256"),
        IndexModel([("page_type", ASCENDING)], name="page_type"),
        IndexModel([("last_fetched_at", DESCENDING)], name="last_fetched"),
    ],
    C.SOURCE_PERMISSIONS: [],
}


def _schema(required: list[str], props: dict[str, Any]) -> dict[str, Any]:
    """Build a ``$jsonSchema`` collection validator.

    Args:
        required: Field names every document must have.
        props: BSON type constraints per field name.

    Returns:
        The validator document for ``create_collection``/``collMod``.
    """
    return {"$jsonSchema": {"bsonType": "object", "required": required, "properties": props}}


_STR = {"bsonType": "string"}
_INT = {"bsonType": ["int", "long"]}
_DATE = {"bsonType": "date"}

VALIDATORS: dict[str, dict[str, Any]] = {
    C.MOVIES: _schema(["_id", "source", "detail_url", "status", "first_discovered_at"], {
        "detail_url": _STR, "status": _STR, "first_discovered_at": _DATE,
    }),
    C.SCRIPTS: _schema(["_id", "movie_id", "script_url", "status", "version", "text", "raw",
                        "parser_version", "rights"], {
        "movie_id": _STR, "script_url": _STR, "status": _STR, "version": _INT, "text": _STR,
    }),
    C.SCRIPT_VERSIONS: _schema(["_id", "script_id", "version", "archived_at"], {
        "script_id": _STR, "version": _INT, "archived_at": _DATE,
    }),
    C.SCENES: _schema(["_id", "script_id", "movie_id", "scene_number", "char_start", "char_end"], {
        "script_id": _STR, "movie_id": _STR, "scene_number": _INT,
    }),
    C.DIALOGUES: _schema(["_id", "script_id", "movie_id", "order", "speaker", "text", "original_text"], {
        "script_id": _STR, "movie_id": _STR, "order": _INT, "text": _STR, "original_text": _STR,
    }),
    C.CHARACTERS: _schema(["_id", "script_id", "movie_id", "name", "name_key"], {
        "script_id": _STR, "name": _STR,
    }),
    C.CRAWL_QUEUE: _schema(["_id", "url", "page_type", "status", "next_attempt_at"], {
        "url": _STR, "status": {"enum": ["pending", "in_progress", "done", "failed", "blocked", "skipped"]},
    }),
    C.CRAWL_LOGS: _schema(["ts", "level", "event"], {"ts": _DATE, "event": _STR}),
    C.CRAWL_JOBS: _schema(["_id", "status", "started_at"], {"status": _STR}),
    C.SOURCE_PERMISSIONS: _schema(["_id", "level", "version"], {
        "level": {"enum": ["denied", "discovery_only", "sample", "bulk"]}, "version": _INT,
    }),
    C.RAW_ARTIFACTS: _schema(["_id", "url", "sha256", "path", "size"], {
        "sha256": _STR, "path": _STR, "size": _INT,
    }),
}


def ensure_schema(db: Database) -> dict[str, list[str]]:
    """Create collections, apply validators and build indexes. Safe to re-run.

    Missing collections are created with their validator (``validationLevel`` moderate: existing invalid
    documents are only checked when they are updated); existing collections get the validator re-applied with
    ``collMod`` (a failure is logged as a warning). The indexes in ``INDEXES`` are then created; creating an
    identical existing index is a no-op.

    Args:
        db: Application database.

    Returns:
        For each collection in ``Collections.ALL``, the names of its indexes from ``INDEXES`` (empty when it only
        has the default ``_id`` index).
    """
    existing = set(db.list_collection_names())
    created: dict[str, list[str]] = {}
    for name in C.ALL:
        validator = VALIDATORS.get(name)
        if name not in existing:
            try:
                db.create_collection(name, validator=validator, validationLevel="moderate",
                                     validationAction="error")
            except CollectionInvalid:
                pass  # created concurrently
        elif validator:
            try:
                db.command({"collMod": name, "validator": validator, "validationLevel": "moderate",
                            "validationAction": "error"})
            except OperationFailure as exc:  # pragma: no cover - permissions on locked-down servers
                log.warning("could not update validator for %s: %s", name, exc)
        specs = INDEXES.get(name, [])
        created[name] = db[name].create_indexes(specs) if specs else []
    return created
