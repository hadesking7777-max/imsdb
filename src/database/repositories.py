"""Repositories: all MongoDB reads/writes go through these classes.

Idempotency rules
-----------------
* Movies are keyed by canonical ``detail_url`` (unique index); listing data uses
  ``$setOnInsert``/``$addToSet`` so re-discovery never duplicates or clobbers.
* Scripts are keyed by canonical ``script_url``. Re-ingesting identical content is a
  no-op; changed content archives the previous document into ``script_versions`` first.
* Scenes/dialogues/characters have deterministic ``_id`` values and are written with
  ``ReplaceOne(upsert=True)``. Every parse carries a ``parse_id``; leftovers from an
  older parse of the same script are removed after the new set is fully written.
* Queue items are keyed by canonical URL; enqueueing is ``$setOnInsert`` only.
"""

from __future__ import annotations

import os
import socket
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, get_args

from pymongo import DeleteMany, ReplaceOne, ReturnDocument, UpdateOne
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import BulkWriteError, DuplicateKeyError

from src.database.models import (
    CharacterDoc,
    Collections as C,
    CrawlJob,
    CrawlLogEvent,
    DialogueDoc,
    MovieDetailFields,
    PermissionLevel,
    QueueItem,
    RawArtifactDoc,
    SceneDoc,
    ScriptDoc,
    SourcePermissionDoc,
    movie_id_for,
)
from src.errors import CrawlerBusyError
from src.utils.clock import as_utc, utcnow
from src.utils.hashing import short_hash
from src.utils.text_processing import display_title, title_key

BULK_BATCH = 1000


def _chunks(seq: Sequence[Any], size: int = BULK_BATCH) -> Iterable[Sequence[Any]]:
    """Split a sequence into consecutive slices for batched bulk writes.

    Args:
        seq: Items to split (e.g. bulk write operations).
        size: Maximum slice length.

    Yields:
        Slices of ``seq`` with at most ``size`` items, in order.
    """
    for start in range(0, len(seq), size):
        yield seq[start:start + size]


# ====================================================================== movies
class MovieRepository:
    """Reads and writes the ``movies`` collection (one document per canonical IMSDb detail URL)."""
    def __init__(self, db: Database) -> None:
        """Bind the repository to a database.

        Args:
            db: Database that holds the ``movies`` collection.
        """
        self.col: Collection = db[C.MOVIES]

    def upsert_from_listing(self, entries: Sequence[Any], *, source_label: str, listing_url: str,
                            now: datetime | None = None) -> dict[str, int]:
        """Insert or update the movies found on one listing page.

        One ``UpdateOne(upsert=True)`` per entry, keyed by ``detail_url`` and sent in unordered batches of
        ``BULK_BATCH``. Identity fields (``_id``, titles, status ``discovered``, first-seen data) are set only on
        insert; ``listing_*`` fields, ``last_seen_at`` and ``updated_at`` are overwritten every time; the source
        label, the raw href and (for ``genre:`` labels) the genre are added to sets. Detail-page data is never touched.

        Args:
            entries: ``ListingEntry`` objects (``detail_url``, ``title``, ``draft_text``/``draft_date``/
                ``draft_label``, ``writers`` and optionally ``href``).
            source_label: Listing that produced the entries, e.g. ``genre:Action``, ``alphabetical:A`` or
                ``all-scripts``.
            listing_url: URL of the listing page (stored as ``first_listing_url`` on new movies).
            now: Timestamp to record; defaults to the current UTC time.

        Returns:
            ``{"inserted": n, "matched": m}``: number of movies created and of existing movies matched.
        """
        from src.crawler.url_manager import detail_title_from_url  # local import avoids cycle

        now = now or utcnow()
        ops: list[UpdateOne] = []
        for entry in entries:
            url_title = detail_title_from_url(entry.detail_url) or entry.title
            add_to_set: dict[str, Any] = {"listing_sources": source_label}
            if getattr(entry, "href", None):
                add_to_set["source_hrefs"] = entry.href
            if source_label.startswith("genre:"):
                add_to_set["listing_genres"] = source_label.split(":", 1)[1]
            ops.append(UpdateOne(
                {"detail_url": entry.detail_url},
                {
                    "$setOnInsert": {
                        "_id": movie_id_for(entry.detail_url, display_title(url_title)),
                        "source": "imsdb",
                        "detail_url": entry.detail_url,
                        "title": entry.title,
                        "display_title": display_title(entry.title),
                        "title_key": title_key(entry.title),
                        "status": "discovered",
                        "first_discovered_at": now,
                        "created_at": now,
                        "first_listing_url": listing_url,
                    },
                    "$set": {
                        "listing_title": entry.title,
                        "listing_draft": {"text": entry.draft_text, "date": entry.draft_date,
                                          "label": entry.draft_label},
                        "listing_writers": entry.writers,
                        "last_seen_at": now,
                        "updated_at": now,
                    },
                    "$addToSet": add_to_set,
                },
                upsert=True,
            ))
        inserted = matched = 0
        for batch in _chunks(ops):
            result = self.col.bulk_write(list(batch), ordered=False)
            inserted += result.upserted_count
            matched += result.matched_count
        return {"inserted": inserted, "matched": matched}

    def ensure_from_detail_url(self, detail_url: str, title: str, now: datetime | None = None) -> str:
        """Return the id of the movie for a detail URL, creating a minimal ``discovered`` record if missing.

        Used when a detail page is reached without having been seen on a listing. An existing document is
        returned unchanged.

        Args:
            detail_url: Canonical detail-page URL.
            title: Title to store if the movie has to be created.
            now: Timestamp for a new record; defaults to the current UTC time.

        Returns:
            The movie ``_id`` (existing or newly inserted).
        """
        now = now or utcnow()
        doc = self.col.find_one_and_update(
            {"detail_url": detail_url},
            {"$setOnInsert": {
                "_id": movie_id_for(detail_url, display_title(title)),
                "source": "imsdb", "detail_url": detail_url, "title": title,
                "display_title": display_title(title), "title_key": title_key(title),
                "status": "discovered", "first_discovered_at": now, "created_at": now,
                "last_seen_at": now, "updated_at": now,
            }},
            upsert=True, return_document=ReturnDocument.AFTER, projection={"_id": 1},
        )
        return doc["_id"]

    def get(self, movie_id: str) -> dict[str, Any] | None:
        """Fetch a movie by ``_id``.

        Args:
            movie_id: Movie ``_id``.

        Returns:
            The movie document, or None if there is none.
        """
        return self.col.find_one({"_id": movie_id})

    def get_by_detail_url(self, detail_url: str) -> dict[str, Any] | None:
        """Fetch a movie by its canonical detail-page URL.

        Args:
            detail_url: Canonical detail-page URL.

        Returns:
            The movie document, or None if there is none.
        """
        return self.col.find_one({"detail_url": detail_url})

    def apply_detail(self, movie_id: str, fields: MovieDetailFields, *, status: str,
                     now: datetime | None = None) -> bool:
        """Store parsed detail-page metadata on a movie and set its status.

        Every field of ``fields`` is written with ``$set``, together with ``status``, ``detail_checked_at`` and
        ``updated_at``. When the stored ``detail_raw.sha256`` differs from the new one, the previous ``detail_raw``
        is pushed onto ``detail_versions`` (with ``superseded_at``) and ``detail_fetched_at`` is set; when the
        content is unchanged, ``detail_fetched_at`` keeps its old value. The movie must already exist (no upsert).

        Args:
            movie_id: ``_id`` of the movie to update.
            fields: Parsed detail-page metadata.
            status: New movie status, e.g. ``detail_collected`` (see ``MovieStatus``).
            now: Timestamp to record; defaults to the current UTC time.

        Returns:
            True if the detail page is new or its content hash changed, False if it is identical to the stored one.
        """
        now = now or utcnow()
        existing = self.col.find_one({"_id": movie_id}, {"detail_raw": 1})
        payload = fields.to_mongo()
        update: dict[str, Any] = {"$set": {**payload, "status": status, "detail_fetched_at": now,
                                           "detail_checked_at": now, "updated_at": now}}
        changed = True
        if existing and existing.get("detail_raw"):
            previous = existing["detail_raw"]
            changed = previous.get("sha256") != fields.detail_raw.sha256
            if changed:
                update["$push"] = {"detail_versions": {**previous, "superseded_at": now}}
            else:
                update["$set"].pop("detail_fetched_at")
        self.col.update_one({"_id": movie_id}, update)
        return changed

    def touch_checked(self, movie_id: str, now: datetime | None = None) -> None:
        """Record that a movie's detail page was re-checked (sets ``detail_checked_at`` only).

        Args:
            movie_id: Movie ``_id``.
            now: Check time; defaults to the current UTC time.
        """
        now = now or utcnow()
        self.col.update_one({"_id": movie_id}, {"$set": {"detail_checked_at": now}})

    def set_status(self, movie_id: str, status: str, *, error: str | None = None,
                   now: datetime | None = None) -> None:
        """Set a movie's ``status`` and ``updated_at``.

        Args:
            movie_id: Movie ``_id``.
            status: New status (see ``MovieStatus``).
            error: Message stored in ``last_error``; None leaves any previous error in place.
            now: Timestamp to record; defaults to the current UTC time.
        """
        now = now or utcnow()
        update: dict[str, Any] = {"status": status, "updated_at": now}
        if error is not None:
            update["last_error"] = error
        self.col.update_one({"_id": movie_id}, {"$set": update})

    def count_records_collected(self) -> int:
        """Count movies whose detail page has been collected (the unit counted by the sample cap).

        Returns:
            Number of movies that have a ``detail_fetched_at`` field.
        """
        return self.col.count_documents({"detail_fetched_at": {"$exists": True}})


# ====================================================================== scripts
@dataclass(slots=True)
class SaveOutcome:
    """Result of ``ScriptRepository.save``.

    Attributes:
        action: ``created`` (new script), ``unchanged`` (same content, parser version and status), ``updated``
            (content changed, previous version archived) or ``reparsed`` (same content, new parser version or
            status).
        version: Script version after the save (starts at 1, incremented on every content change).
        parse_id: Parse run id stamped on the document.
    """
    action: str  # created | unchanged | updated | reparsed
    version: int
    parse_id: str | None


class ScriptRepository:
    """Reads and writes ``scripts`` and archives superseded script versions into ``script_versions``."""
    def __init__(self, db: Database) -> None:
        """Bind the repository to a database.

        Args:
            db: Database that holds the ``scripts`` and ``script_versions`` collections.
        """
        self.col: Collection = db[C.SCRIPTS]
        self.versions: Collection = db[C.SCRIPT_VERSIONS]

    def get(self, script_id: str, projection: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Fetch a script by ``_id``.

        Args:
            script_id: Script ``_id``.
            projection: MongoDB projection selecting the fields to return, or None for the whole document.

        Returns:
            The script document, or None if there is none.
        """
        return self.col.find_one({"_id": script_id}, projection)

    def needs_parse(self, script_id: str, content_hash: str | None, parser_version: str) -> bool:
        """Decide whether a script has to be (re)parsed and saved.

        Args:
            script_id: Script ``_id``.
            content_hash: SHA-256 of the newly extracted text, or None when no text was extracted.
            parser_version: Current parser version.

        Returns:
            False only when the stored script has the same content hash and parser version and a successful status
            (``parsed``, ``parsed_with_warnings`` or ``pdf_only``); True otherwise, including when none is stored.
        """
        existing = self.col.find_one({"_id": script_id}, {"content_hash": 1, "parser_version": 1, "status": 1})
        if existing is None:
            return True
        return not (existing.get("content_hash") == content_hash
                    and existing.get("parser_version") == parser_version
                    and existing.get("status") in {"parsed", "parsed_with_warnings", "pdf_only"})

    def touch_checked(self, script_id: str, raw_ref: dict[str, Any] | None = None,
                      now: datetime | None = None) -> None:
        """Record that a script page was re-checked without being re-parsed.

        Sets ``last_checked_at`` and, when given, replaces the ``raw`` reference.

        Args:
            script_id: Script ``_id``.
            raw_ref: ``RawRef`` dict of the latest fetch, or None/empty to keep the stored one.
            now: Check time; defaults to the current UTC time.
        """
        now = now or utcnow()
        update: dict[str, Any] = {"last_checked_at": now}
        if raw_ref:
            update["raw"] = raw_ref
        self.col.update_one({"_id": script_id}, {"$set": update})

    def save(self, doc: ScriptDoc, *, parse_id: str, now: datetime | None = None) -> SaveOutcome:
        """Write a parsed script, keeping its version history.

        * No stored script: the document is inserted (``created``).
        * Same ``content_hash``: the document is replaced in place, keeping ``created_at``, ``version`` and
          ``previous_versions`` (``reparsed`` if the parser version or status changed, otherwise ``unchanged``).
        * Different content: the complete stored document is first archived into ``script_versions`` as
          ``<id>:v<version>`` (an archive left by an interrupted attempt is kept), then replaced with
          ``version + 1`` (``updated``).

        Args:
            doc: The new script document.
            parse_id: Id of this parse run, stored on the document so derived records can be matched to it.
            now: Archive time, and fallback ``created_at`` when the stored document lacks one; defaults to the
                current UTC time.

        Returns:
            The action taken, the resulting version and ``parse_id``.
        """
        now = now or utcnow()
        data = doc.to_mongo()
        data["parse_id"] = parse_id
        existing = self.col.find_one({"_id": doc.id}, {"elements": 0, "source_markup": 0,
                                                        "formatting_spans": 0, "text": 0})
        if existing is None:
            self.col.replace_one({"_id": doc.id}, data, upsert=True)
            return SaveOutcome("created", data["version"], parse_id)

        data["created_at"] = existing.get("created_at", now)
        same_content = existing.get("content_hash") == doc.content_hash
        if same_content:
            data["version"] = existing.get("version", 1)
            data["previous_versions"] = existing.get("previous_versions", 0)
            self.col.replace_one({"_id": doc.id}, data)
            action = "reparsed" if (existing.get("parser_version") != doc.parser_version
                                    or existing.get("status") != doc.status) else "unchanged"
            return SaveOutcome(action, data["version"], parse_id)

        # Content changed: archive the complete previous document before replacing it.
        full_previous = self.col.find_one({"_id": doc.id})
        assert full_previous is not None
        version = int(full_previous.get("version", 1))
        archived = {k: v for k, v in full_previous.items() if k != "_id"}
        try:
            self.versions.insert_one({
                "_id": f"{doc.id}:v{version}",
                "script_id": doc.id,
                "version": version,
                "archived_at": now,
                "reason": "source content changed",
                "content_hash": full_previous.get("content_hash"),
                "superseded_by_hash": doc.content_hash,
                "document": archived,
            })
        except DuplicateKeyError:
            pass  # archived by an earlier, interrupted attempt
        data["version"] = version + 1
        data["previous_versions"] = int(full_previous.get("previous_versions", 0)) + 1
        self.col.replace_one({"_id": doc.id}, data)
        return SaveOutcome("updated", data["version"], parse_id)


class DerivedRepository:
    """Scenes, dialogues and characters for one script, replaced as a set per parse.

    The new set is written first and leftovers of older parses are removed afterwards; this is not a
    MongoDB transaction.
    """

    def __init__(self, db: Database) -> None:
        """Bind the repository to a database.

        Args:
            db: Database that holds the ``scenes``, ``dialogues`` and ``characters`` collections.
        """
        self.scenes: Collection = db[C.SCENES]
        self.dialogues: Collection = db[C.DIALOGUES]
        self.characters: Collection = db[C.CHARACTERS]

    def replace_for_script(self, script_id: str, parse_id: str, *, scenes: Sequence[SceneDoc],
                           dialogues: Sequence[DialogueDoc], characters: Sequence[CharacterDoc]) -> dict[str, int]:
        """Write the scenes, dialogues and characters of one parse and delete those of older parses.

        Each document is upserted by ``_id`` with ``parse_id`` stamped on it (unordered bulk writes in batches of
        ``BULK_BATCH``). Only after all three sets are written are this script's documents with a different
        ``parse_id`` deleted, so an interrupted run never leaves the script with fewer records than before.

        Args:
            script_id: ``_id`` of the script the records belong to.
            parse_id: Id of the current parse; this script's records carrying any other id are stale.
            scenes: All scenes of the parse.
            dialogues: All dialogue blocks of the parse.
            characters: All characters of the parse.

        Returns:
            Counts keyed ``scenes_written``, ``dialogues_written``, ``characters_written``, ``scenes_removed_stale``,
            ``dialogues_removed_stale`` and ``characters_removed_stale``.
        """
        written: dict[str, int] = {}
        for name, col, docs in (("scenes", self.scenes, scenes), ("dialogues", self.dialogues, dialogues),
                                ("characters", self.characters, characters)):
            ops = []
            for doc in docs:
                data = doc.to_mongo()
                data["parse_id"] = parse_id
                ops.append(ReplaceOne({"_id": data["_id"]}, data, upsert=True))
            for batch in _chunks(ops):
                col.bulk_write(list(batch), ordered=False)
            written[name] = len(ops)
        # Only after the new set is complete, remove leftovers of older parses.
        removed = {}
        for name, col in (("scenes", self.scenes), ("dialogues", self.dialogues),
                          ("characters", self.characters)):
            result = col.bulk_write([DeleteMany({"script_id": script_id, "parse_id": {"$ne": parse_id}})])
            removed[name] = result.deleted_count
        return {**{f"{k}_written": v for k, v in written.items()},
                **{f"{k}_removed_stale": v for k, v in removed.items()}}


# ====================================================================== queue
class QueueRepository:
    """Reads and writes ``crawl_queue``: URLs to fetch with their lease, retry and change-tracking state.

    Every state-changing call on a claimed item filters on ``lease_owner`` so a worker whose lease was reset
    cannot overwrite another worker's result.
    """
    def __init__(self, db: Database) -> None:
        """Bind the repository to a database.

        Args:
            db: Database that holds the ``crawl_queue`` collection.
        """
        self.col: Collection = db[C.CRAWL_QUEUE]

    def enqueue_many(self, items: Sequence[QueueItem]) -> int:
        """Add URLs to the queue, ignoring those already queued.

        Each item is upserted by ``url`` with ``$setOnInsert`` only, so existing items keep their state.
        Duplicate-key errors caused by concurrent inserts are ignored.

        Args:
            items: Queue items to add.

        Returns:
            Number of items newly inserted (0 for an empty list).
        """
        if not items:
            return 0
        ops = [UpdateOne({"url": item.url}, {"$setOnInsert": item.to_mongo()}, upsert=True) for item in items]
        inserted = 0
        for batch in _chunks(ops):
            try:
                inserted += self.col.bulk_write(list(batch), ordered=False).upserted_count
            except BulkWriteError as exc:  # concurrent upsert races surface as duplicate keys
                if any(err.get("code") != 11000 for err in exc.details.get("writeErrors", [])):
                    raise
                inserted += exc.details.get("nUpserted", 0)  # the batch's other upserts did succeed
        return inserted

    def get_by_url(self, url: str) -> dict[str, Any] | None:
        """Fetch a queue item by URL.

        Args:
            url: Canonical URL.

        Returns:
            The queue document, or None if the URL is not queued.
        """
        return self.col.find_one({"url": url})

    def claim_next(self, *, job_id: str, worker_id: str, page_types: Sequence[str], lease_seconds: int,
                   now: datetime | None = None) -> dict[str, Any] | None:
        """Atomically lease the next ready item to a worker.

        Picks a ``pending`` item of one of ``page_types`` whose ``next_attempt_at`` has passed, by highest
        ``priority``, then oldest ``discovered_at``, then ``position`` and ``_id``. It becomes ``in_progress``,
        leased to ``worker_id`` for ``lease_seconds``, tagged with ``job_id``, and its ``attempts`` is incremented.

        Args:
            job_id: Job claiming the item.
            worker_id: Lease owner; later updates must present the same id.
            page_types: Page types this worker handles.
            lease_seconds: Lease duration in seconds; afterwards ``reset_expired_leases`` may return the item.
            now: Reference time; defaults to the current UTC time.

        Returns:
            The claimed item as it is after the update, or None when nothing is ready.
        """
        now = now or utcnow()
        return self.col.find_one_and_update(
            {"status": "pending", "page_type": {"$in": list(page_types)}, "next_attempt_at": {"$lte": now}},
            {"$set": {"status": "in_progress", "lease_owner": worker_id,
                      "lease_expires_at": now + timedelta(seconds=lease_seconds),
                      "job_id": job_id, "last_attempt_at": now},
             "$inc": {"attempts": 1}},
            sort=[("priority", -1), ("discovered_at", 1), ("position", 1), ("_id", 1)],
            return_document=ReturnDocument.AFTER,
        )

    def complete(self, item_id: str, worker_id: str, *, status_code: int | None, content_hash: str | None,
                 changed: bool, etag: str | None = None, last_modified: str | None = None,
                 movie_id: str | None = None, now: datetime | None = None) -> bool:
        """Mark a leased item as successfully fetched.

        Sets status ``done``, clears the lease, ``last_error``, the attempt counter and any refresh flag, records
        the HTTP status, increments ``fetch_count`` (and ``change_count`` when ``changed``) and stores the optional
        values that are given.

        Args:
            item_id: Queue item ``_id``.
            worker_id: Must match the item's ``lease_owner``, otherwise nothing is written.
            status_code: HTTP status of the final response, or None.
            content_hash: SHA-256 of the fetched body; None/empty keeps the stored value.
            changed: Whether the content differs from the previous fetch.
            etag: ``ETag`` response header, if any.
            last_modified: ``Last-Modified`` response header, if any.
            movie_id: Movie the page belongs to, if known.
            now: Completion time; defaults to the current UTC time.

        Returns:
            True if the item was updated, False if the lease was lost (reset or owned by another worker).
        """
        now = now or utcnow()
        update: dict[str, Any] = {
            "$set": {"status": "done", "completed_at": now, "lease_owner": None, "lease_expires_at": None,
                     "last_error": None, "last_status_code": status_code, "attempts": 0},
            "$inc": {"fetch_count": 1, "change_count": 1 if changed else 0},
            "$unset": {"refresh_requested_at": ""},
        }
        if content_hash:
            update["$set"]["content_hash"] = content_hash
        if etag:
            update["$set"]["etag"] = etag
        if last_modified:
            update["$set"]["last_modified"] = last_modified
        if movie_id:
            update["$set"]["movie_id"] = movie_id
        result = self.col.update_one({"_id": item_id, "lease_owner": worker_id}, update)
        return result.modified_count == 1

    def fail(self, item: dict[str, Any], worker_id: str, *, error: str, status_code: int | None,
             retryable: bool, blocked: bool, retry_after: float | None, max_attempts: int,
             now: datetime | None = None) -> str:
        """Record a failed fetch and decide whether the item will be retried.

        * ``blocked``: status ``blocked`` (never retried automatically).
        * ``retryable`` and ``attempts < max_attempts``: back to ``pending`` after an exponential back-off of
          60 s * 2^(attempts - 1), capped at 6 hours, but never shorter than ``retry_after``.
        * Otherwise: status ``failed``.

        The lease is cleared, the error (truncated to 2000 characters) and status code are stored, and
        ``completed_at`` is set for ``failed``/``blocked``. The write applies only while ``worker_id`` holds the lease.

        Args:
            item: The claimed queue document (its ``attempts`` already counts this attempt).
            worker_id: Current lease owner.
            error: Error message.
            status_code: HTTP status, or None for network errors.
            retryable: Whether the error is transient.
            blocked: Whether the server denied access (401/403/451).
            retry_after: Delay requested by the server (``Retry-After``) in seconds, or None.
            max_attempts: Attempts allowed before giving up.
            now: Failure time; defaults to the current UTC time.

        Returns:
            The new status: ``pending``, ``failed`` or ``blocked``; or ``lease_lost`` when ``worker_id`` no longer
            holds the lease (it expired and the item was recovered), in which case nothing is written.
        """
        now = now or utcnow()
        attempts = int(item.get("attempts", 1))
        if blocked:
            status = "blocked"
            next_at = now
        elif retryable and attempts < max_attempts:
            status = "pending"
            delay = max(retry_after or 0.0, min(6 * 3600.0, 60.0 * (2 ** (attempts - 1))))
            next_at = now + timedelta(seconds=delay)
        else:
            status = "failed"
            next_at = now
        result = self.col.update_one(
            {"_id": item["_id"], "lease_owner": worker_id},
            {"$set": {"status": status, "last_error": error[:2000], "last_status_code": status_code,
                      "next_attempt_at": next_at, "lease_owner": None, "lease_expires_at": None,
                      "completed_at": now if status in {"failed", "blocked"} else None}},
        )
        return status if result.matched_count else "lease_lost"

    def skip(self, item_id: str, worker_id: str, reason: str, now: datetime | None = None) -> None:
        """Mark a leased item as ``skipped`` (completed without being processed) and clear its lease.

        Args:
            item_id: Queue item ``_id``.
            worker_id: Must match the item's ``lease_owner``, otherwise nothing is written.
            reason: Why it was skipped (stored in ``last_error``).
            now: Completion time; defaults to the current UTC time.
        """
        now = now or utcnow()
        self.col.update_one({"_id": item_id, "lease_owner": worker_id},
                            {"$set": {"status": "skipped", "last_error": reason, "completed_at": now,
                                      "lease_owner": None, "lease_expires_at": None}})

    def release(self, item_id: str, worker_id: str) -> None:
        """Return a claimed item without counting the attempt (graceful shutdown).

        Args:
            item_id: Queue item ``_id``; only affected while ``in_progress``.
            worker_id: Must match the item's ``lease_owner``.
        """
        self.col.update_one({"_id": item_id, "lease_owner": worker_id, "status": "in_progress"},
                            {"$set": {"status": "pending", "lease_owner": None, "lease_expires_at": None},
                             "$inc": {"attempts": -1}})

    def release_job(self, job_id: str) -> int:
        """Return every ``in_progress`` item claimed by a job to ``pending`` (attempt counters are kept).

        Args:
            job_id: Job whose items are released.

        Returns:
            Number of items released.
        """
        result = self.col.update_many({"status": "in_progress", "job_id": job_id},
                                      {"$set": {"status": "pending", "lease_owner": None,
                                                "lease_expires_at": None}})
        return result.modified_count

    def reset_expired_leases(self, now: datetime | None = None) -> int:
        """Return ``in_progress`` items whose lease has expired (crashed or stuck workers) to ``pending``.

        Args:
            now: Reference time; defaults to the current UTC time.

        Returns:
            Number of items reset.
        """
        now = now or utcnow()
        result = self.col.update_many({"status": "in_progress", "lease_expires_at": {"$lt": now}},
                                      {"$set": {"status": "pending", "lease_owner": None,
                                                "lease_expires_at": None}})
        return result.modified_count

    def requeue(self, *, page_types: Sequence[str], statuses: Sequence[str],
                completed_before: datetime | None = None, now: datetime | None = None) -> int:
        """Put items back to ``pending`` for an immediate retry.

        Resets ``attempts`` to 0, sets ``next_attempt_at`` to ``now`` and clears any lease.

        Args:
            page_types: Page types to requeue.
            statuses: Current statuses to requeue, e.g. ``["failed"]``.
            completed_before: Only items completed before this time, or None for all.
            now: Time the items become ready; defaults to the current UTC time.

        Returns:
            Number of items requeued.
        """
        now = now or utcnow()
        query: dict[str, Any] = {"page_type": {"$in": list(page_types)}, "status": {"$in": list(statuses)}}
        if completed_before is not None:
            query["completed_at"] = {"$lt": completed_before}
        result = self.col.update_many(query, {"$set": {"status": "pending", "next_attempt_at": now,
                                                       "attempts": 0, "lease_owner": None,
                                                       "lease_expires_at": None}})
        return result.modified_count

    def request_refresh(self, *, page_types: Sequence[str], completed_before: datetime,
                        now: datetime | None = None) -> int:
        """Re-queue completed items older than ``completed_before`` and flag them as refresh work.

        Matching ``done`` items become ``pending`` with ``attempts`` 0, ``next_attempt_at`` and
        ``refresh_requested_at`` set to ``now``; ``complete`` clears the flag again.

        Args:
            page_types: Page types to refresh.
            completed_before: Only items completed before this time are refreshed.
            now: Time the items become ready; defaults to the current UTC time.

        Returns:
            Number of items re-queued.
        """
        now = now or utcnow()
        result = self.col.update_many(
            {"page_type": {"$in": list(page_types)}, "status": "done", "completed_at": {"$lt": completed_before}},
            {"$set": {"status": "pending", "next_attempt_at": now, "attempts": 0, "refresh_requested_at": now,
                      "lease_owner": None, "lease_expires_at": None}},
        )
        return result.modified_count

    def count(self, query: dict[str, Any]) -> int:
        """Count queue items matching a filter.

        Args:
            query: MongoDB filter document.

        Returns:
            Number of matching items.
        """
        return self.col.count_documents(query)

    def next_ready_at(self, page_types: Sequence[str]) -> datetime | None:
        """Find when the next pending item of the given page types becomes claimable.

        Args:
            page_types: Page types to consider.

        Returns:
            The earliest ``next_attempt_at`` (UTC) among pending items, or None when there are none.
        """
        doc = self.col.find_one({"status": "pending", "page_type": {"$in": list(page_types)}},
                                {"next_attempt_at": 1}, sort=[("next_attempt_at", 1)])
        return as_utc(doc["next_attempt_at"]) if doc else None

    def status_counts(self) -> dict[str, dict[str, int]]:
        """Count queue items per page type and status.

        Returns:
            ``{page_type: {status: count}}``.
        """
        out: dict[str, dict[str, int]] = {}
        for row in self.col.aggregate([{"$group": {"_id": {"t": "$page_type", "s": "$status"}, "n": {"$sum": 1}}}]):
            out.setdefault(row["_id"]["t"], {})[row["_id"]["s"]] = row["n"]
        return out


# ====================================================================== logs / jobs
class LogRepository:
    """Writes structured crawl events to ``crawl_logs``."""
    def __init__(self, db: Database) -> None:
        """Bind the repository to a database.

        Args:
            db: Database that holds the ``crawl_logs`` collection.
        """
        self.col: Collection = db[C.CRAWL_LOGS]

    def log(self, event: str, *, level: str = "INFO", job_id: str | None = None, url: str | None = None,
            page_type: str | None = None, record_id: str | None = None, message: str | None = None,
            details: dict[str, Any] | None = None) -> None:
        """Insert one crawl log event, timestamped with the current UTC time.

        Args:
            event: Short event name, e.g. ``robots_fetch`` or ``parse_warning``.
            level: ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR`` or ``CRITICAL``.
            job_id: Job the event belongs to, or None.
            url: URL concerned, or None.
            page_type: Page type concerned, or None.
            record_id: Id of the affected record (movie, script...), or None.
            message: Human-readable message, or None.
            details: Extra structured data; None stores an empty dict.
        """
        doc = CrawlLogEvent(ts=utcnow(), level=level, event=event, job_id=job_id, url=url,  # type: ignore[arg-type]
                            page_type=page_type, record_id=record_id, message=message,
                            details=details or {})
        self.col.insert_one(doc.to_mongo())


LOCK_ID = "lock:crawler"


def pid_alive(pid: int) -> bool:
    """Whether a local process exists. Never signals the process (os.kill(pid, 0) terminates on Windows).

    Args:
        pid: Process id.

    Returns:
        True if the process is running (or exists but belongs to another user); False if it does not exist,
        has exited, or ``pid`` is not positive.
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return kernel32.GetLastError() == 5  # access denied => exists (another user's process)
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _owner_is_dead(owner: str | None) -> bool:
    """Tell whether the crawler-lock owner is a process on this host that no longer runs.

    Lock owners look like ``host:pid:rand``; only same-host owners can be checked.

    Args:
        owner: Owner string from the lock document, or None.

    Returns:
        True only when the owner is on this host and its pid is not alive; False when the owner is empty,
        malformed, on another host or still running.
    """
    if not owner:
        return False
    parts = owner.split(":")
    if len(parts) < 3 or parts[0] != socket.gethostname() or not parts[1].isdigit():
        return False
    return not pid_alive(int(parts[1]))


class JobRepository:
    """Reads and writes ``crawl_jobs``: one document per CLI job run, plus the crawler lock ``lock:crawler``."""
    def __init__(self, db: Database) -> None:
        """Bind the repository to a database.

        Args:
            db: Database that holds the ``crawl_jobs`` collection.
        """
        self.col: Collection = db[C.CRAWL_JOBS]

    def start(self, command: str, params: dict[str, Any], *, resumed_from: str | None = None) -> CrawlJob:
        """Create and insert a new ``running`` job record.

        Args:
            command: CLI command that started the job, e.g. ``crawl``.
            params: Command parameters, stored so the job can be resumed.
            resumed_from: Id of the interrupted job this one continues, or None.

        Returns:
            The inserted job (id ``job-<UTC timestamp>-<6 hex>``, with this host name and process id).
        """
        now = utcnow()
        job = CrawlJob(_id=f"job-{now:%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}", command=command, params=params,
                       status="running", started_at=now, heartbeat_at=now, host=socket.gethostname(),
                       pid=os.getpid(), resumed_from=resumed_from)
        self.col.insert_one(job.to_mongo())
        return job

    def heartbeat(self, job_id: str, counters: dict[str, int]) -> None:
        """Refresh a job's ``heartbeat_at`` and replace its stored counters.

        Args:
            job_id: Job ``_id``.
            counters: Current progress counters.
        """
        self.col.update_one({"_id": job_id}, {"$set": {"heartbeat_at": utcnow(), "counters": counters}})

    def finish(self, job_id: str, status: str, counters: dict[str, int], error: str | None = None) -> None:
        """Mark a job as finished (sets ``status``, ``finished_at``, final counters and error).

        Args:
            job_id: Job ``_id``.
            status: Final status: ``completed``, ``failed``, ``interrupted`` or ``stopped``.
            counters: Final progress counters.
            error: Error message, or None on success.
        """
        now = utcnow()
        self.col.update_one({"_id": job_id}, {"$set": {"status": status, "finished_at": now, "heartbeat_at": now,
                                                        "counters": counters, "error": error}})

    def get(self, job_id: str) -> dict[str, Any] | None:
        """Fetch a job by ``_id``.

        Args:
            job_id: Job ``_id``.

        Returns:
            The job document, or None if there is none.
        """
        return self.col.find_one({"_id": job_id})

    def latest_resumable(self, stale_after: timedelta = timedelta(minutes=2)) -> dict[str, Any] | None:
        """Find the most recent job that can be resumed.

        Looks at the 5 most recently started ``running``/``interrupted`` jobs. An ``interrupted`` job is returned
        as is. A ``running`` job whose heartbeat is older than ``stale_after`` is treated as crashed: it is marked
        ``interrupted`` in MongoDB and returned. Running jobs with a fresh heartbeat are skipped.

        Args:
            stale_after: Heartbeat age after which a running job counts as crashed.

        Returns:
            The job document (with status ``interrupted``), or None when there is nothing to resume.
        """
        now = utcnow()
        for job in self.col.find({"status": {"$in": ["running", "interrupted"]}, "command": {"$exists": True}},
                                 sort=[("started_at", -1)], limit=5):
            if job["status"] == "interrupted":
                return job
            if as_utc(job["heartbeat_at"]) < now - stale_after:  # crashed process
                self.col.update_one({"_id": job["_id"], "status": "running"},
                                    {"$set": {"status": "interrupted", "error": "stale heartbeat (process died)"}})
                job["status"] = "interrupted"
                return job
        return None

    def mark_superseded(self, job_id: str, by_job: str) -> None:
        """Mark an old job as ``stopped`` because another job resumed it.

        Args:
            job_id: The superseded job.
            by_job: Id of the job that took over (stored as ``resumed_by``).
        """
        self.col.update_one({"_id": job_id}, {"$set": {"status": "stopped", "resumed_by": by_job}})

    # ---------------------------------------------------------------- crawl lock
    def acquire_lock(self, owner: str, ttl_seconds: int = 300) -> None:
        """Take the crawler lock so that only one crawl runs at a time.

        The lock is the ``lock:crawler`` document in ``crawl_jobs``. It is taken (or created) when it is free,
        already owned by ``owner`` or expired. When another owner holds it but is a dead process on this host, the
        lock is taken over atomically and the old owner is recorded in ``taken_over_from``.

        Args:
            owner: Lock owner id, ``host:pid:rand``.
            ttl_seconds: Lock lifetime in seconds; keep it alive with ``refresh_lock``.

        Raises:
            CrawlerBusyError: Another live process holds the lock.
        """
        now = utcnow()
        try:
            self.col.find_one_and_update(
                {"_id": LOCK_ID, "$or": [{"owner": None}, {"owner": owner}, {"expires_at": {"$lt": now}}]},
                {"$set": {"owner": owner, "expires_at": now + timedelta(seconds=ttl_seconds),
                          "status": "lock", "started_at": now}},
                upsert=True,
            )
        # A held lock does not match the filter, so the upsert tries to insert a second ``lock:crawler``.
        except DuplicateKeyError as exc:
            holder = self.col.find_one({"_id": LOCK_ID}) or {}
            if _owner_is_dead(holder.get("owner")):
                # The holder crashed on this machine: take the lock over atomically.
                taken = self.col.find_one_and_update(
                    {"_id": LOCK_ID, "owner": holder.get("owner")},
                    {"$set": {"owner": owner, "expires_at": now + timedelta(seconds=ttl_seconds),
                              "status": "lock", "started_at": now, "taken_over_from": holder.get("owner")}},
                )
                if taken is not None:
                    return
            raise CrawlerBusyError(
                f"Another crawler run holds the lock (owner {holder.get('owner')}, expires "
                f"{holder.get('expires_at')}). Wait for it to finish or let the lock expire."
            ) from exc

    def refresh_lock(self, owner: str, ttl_seconds: int = 300) -> None:
        """Extend the crawler lock to expire ``ttl_seconds`` from now, if ``owner`` still holds it.

        Args:
            owner: Lock owner id.
            ttl_seconds: New lifetime in seconds.
        """
        self.col.update_one({"_id": LOCK_ID, "owner": owner},
                            {"$set": {"expires_at": utcnow() + timedelta(seconds=ttl_seconds)}})

    def release_lock(self, owner: str) -> None:
        """Free the crawler lock if ``owner`` holds it (owner and expiry set to None; the document stays).

        Args:
            owner: Lock owner id.
        """
        self.col.update_one({"_id": LOCK_ID, "owner": owner}, {"$set": {"owner": None, "expires_at": None}})


# ====================================================================== raw artifacts
class RawArtifactRepository:
    """Indexes the raw HTTP responses stored on disk in ``raw_artifacts``, one document per (URL, body SHA-256)."""
    def __init__(self, db: Database) -> None:
        """Bind the repository to a database.

        Args:
            db: Database that holds the ``raw_artifacts`` collection.
        """
        self.col: Collection = db[C.RAW_ARTIFACTS]

    @staticmethod
    def artifact_id(url: str, sha256: str) -> str:
        """Build the deterministic ``_id`` of a raw artifact.

        Args:
            url: Requested URL.
            sha256: SHA-256 hex digest of the body.

        Returns:
            ``raw-`` followed by a 24-character hash of both values.
        """
        return "raw-" + short_hash(url, sha256, length=24)

    def record(self, doc: RawArtifactDoc) -> None:
        """Insert or update the index entry of a fetched response.

        Keyed by (``url``, ``sha256``): identity fields (``_id``, size, path, compression, page type, first fetch
        time) are set only on insert; response metadata (final URL, status, headers, HTTP version, encoding, last
        fetch time) is overwritten; ``fetch_count`` is incremented.

        Args:
            doc: Artifact metadata of the fetch.
        """
        data = doc.to_mongo()
        insert_only = {k: data[k] for k in ("_id", "url", "sha256", "size", "path", "compressed",
                                             "page_type", "first_fetched_at")}
        self.col.update_one(
            {"url": doc.url, "sha256": doc.sha256},
            {"$setOnInsert": insert_only,
             "$set": {"final_url": doc.final_url, "status_code": doc.status_code, "headers": doc.headers,
                      "http_version": doc.http_version, "encoding": doc.encoding,
                      "last_fetched_at": doc.last_fetched_at},
             "$inc": {"fetch_count": 1}},
            upsert=True,
        )


# ====================================================================== permissions
class PermissionRepository:
    """Reads and writes ``source_permissions``: each source's permission level with its evidence and history."""
    def __init__(self, db: Database) -> None:
        """Bind the repository to a database.

        Args:
            db: Database that holds the ``source_permissions`` collection.
        """
        self.col: Collection = db[C.SOURCE_PERMISSIONS]

    def get(self, source_id: str) -> SourcePermissionDoc | None:
        """Load a source's permission record.

        Args:
            source_id: Source id, e.g. ``imsdb``.

        Returns:
            The validated record, or None if the source has not been seeded.
        """
        doc = self.col.find_one({"_id": source_id})
        return SourcePermissionDoc.model_validate(doc) if doc else None

    def seed(self, doc: SourcePermissionDoc) -> bool:
        """Insert the default permission record if the source has none (an existing record is never changed).

        Args:
            doc: Default record to insert.

        Returns:
            True if it was inserted, False if a record already existed.
        """
        result = self.col.update_one({"_id": doc.id}, {"$setOnInsert": doc.to_mongo()}, upsert=True)
        return result.upserted_id is not None

    def change_level(self, source_id: str, *, level: str, sample_limit: int | None, granted_by: str,
                     evidence: str, note: str | None = None) -> SourcePermissionDoc:
        """Change a source's permission level and record who authorised it and why.

        Sets ``level`` (and ``sample_limit`` when given), increments ``version``, and pushes a snapshot of the
        previous state onto ``history`` and the statement onto ``evidence``.

        Args:
            source_id: Source id, e.g. ``imsdb``.
            level: New level: ``denied``, ``discovery_only``, ``sample`` or ``bulk``.
            sample_limit: Total records allowed at ``sample`` level, or None to keep the current limit.
            granted_by: Who authorised the change.
            evidence: Description or location of the authorisation evidence.
            note: Optional free-text note.

        Returns:
            The updated permission record.

        Raises:
            ValueError: ``level`` is not a known permission level, or ``sample_limit`` is negative.
            KeyError: The source has no permission record.
        """
        if level not in get_args(PermissionLevel):
            raise ValueError(f"unknown permission level {level!r}; use one of {', '.join(get_args(PermissionLevel))}")
        if sample_limit is not None and sample_limit < 0:
            raise ValueError("sample_limit must not be negative")
        now = utcnow()
        current = self.col.find_one({"_id": source_id})
        if current is None:
            raise KeyError(source_id)
        snapshot = {"version": current["version"], "level": current["level"],
                    "sample_limit": current.get("sample_limit"), "changed_at": now,
                    "changed_by": granted_by, "new_level": level, "evidence": evidence, "note": note}
        update: dict[str, Any] = {"level": level, "updated_at": now, "version": current["version"] + 1}
        if sample_limit is not None:
            update["sample_limit"] = sample_limit
        self.col.update_one({"_id": source_id}, {
            "$set": update,
            "$push": {"history": snapshot,
                      "evidence": {"recorded_at": now, "recorded_by": granted_by, "statement": evidence,
                                   "level": level, "note": note}},
        })
        result = self.get(source_id)
        assert result is not None
        return result

    def record_inspection(self, source_id: str, *, robots: dict[str, Any], policy_documents: list[dict[str, Any]],
                          assessment: dict[str, Any]) -> None:
        """Store the result of a source policy inspection.

        Overwrites ``robots``, ``policy_documents`` and ``assessment``, sets ``inspected_at``/``updated_at`` and
        appends a compact entry (robots status and hash, each document's URL, hash and status) to ``inspections``.

        Args:
            source_id: Source id, e.g. ``imsdb``.
            robots: Parsed robots.txt summary (its ``status`` and ``sha256`` are also logged).
            policy_documents: Fetched policy pages, each with ``url``, ``sha256`` and ``status_code``.
            assessment: Result of the policy text assessment.
        """
        now = utcnow()
        self.col.update_one({"_id": source_id}, {
            "$set": {"robots": robots, "policy_documents": policy_documents, "assessment": assessment,
                     "inspected_at": now, "updated_at": now},
            "$push": {"inspections": {"at": now, "robots_status": robots.get("status"),
                                      "robots_sha256": robots.get("sha256"),
                                      "documents": [{"url": d.get("url"), "sha256": d.get("sha256"),
                                                     "status": d.get("status_code")} for d in policy_documents]}},
        })

    def update_robots(self, source_id: str, robots: dict[str, Any]) -> None:
        """Replace the stored robots.txt summary of a source.

        Args:
            source_id: Source id, e.g. ``imsdb``.
            robots: New parsed robots.txt summary.
        """
        self.col.update_one({"_id": source_id}, {"$set": {"robots": robots, "updated_at": utcnow()}})


class Repositories:
    """Convenience bundle of all repositories for one database."""

    def __init__(self, db: Database) -> None:
        """Create every repository on the same database.

        Args:
            db: Application database.
        """
        self.db = db
        self.movies = MovieRepository(db)
        self.scripts = ScriptRepository(db)
        self.derived = DerivedRepository(db)
        self.queue = QueueRepository(db)
        self.logs = LogRepository(db)
        self.jobs = JobRepository(db)
        self.raw = RawArtifactRepository(db)
        self.permissions = PermissionRepository(db)
