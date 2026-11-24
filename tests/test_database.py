"""MongoDB integration tests (isolated throwaway database; skipped if MongoDB is unreachable)."""

from __future__ import annotations

from datetime import timedelta

import pytest
from pymongo.errors import DuplicateKeyError, WriteError

from src.crawler.discovery import make_queue_item
from src.database.connection import assert_local_uri
from src.database.indexes import ensure_schema
from src.database.models import Collections as C
from src.database.repositories import Repositories
from src.errors import ConfigError, CrawlerBusyError
from src.parsers.movie_parser import parse_listing_page
from src.parsers.screenplay_parser import extract_screenplay, parse_screenplay
from src.database.models import RawRef
from src.services.documents import build_script_bundle, make_rights
from src.utils.clock import utcnow
from src.utils.text_processing import decode_html_bytes
from tests.conftest import BASE_URL, fixture_text

pytestmark = pytest.mark.mongodb


# ====================================================================== connection rules (no server needed)
@pytest.mark.parametrize("uri", ["mongodb://127.0.0.1:27017/", "mongodb://localhost/", "mongodb://[::1]:27017/",
                                 "mongodb://user:pw@127.0.0.1:27017/?authSource=admin"])
def test_local_uris_are_accepted(uri: str) -> None:
    """Loopback hosts (IPv4, ``localhost``, IPv6, with or without credentials) pass the local-only check.

    Args:
        uri: A MongoDB connection string.
    """
    assert_local_uri(uri)


@pytest.mark.parametrize("uri", ["mongodb://db.example.org:27017/", "mongodb+srv://cluster0.example.mongodb.net/",
                                 "mongodb://127.0.0.1,10.0.0.5/"])
def test_remote_uris_are_rejected(uri: str) -> None:
    """Remote hosts, ``mongodb+srv`` and multi-host URIs with any remote member raise ``ConfigError``.

    Args:
        uri: A MongoDB connection string.
    """
    with pytest.raises(ConfigError):
        assert_local_uri(uri)


# ====================================================================== schema
def test_schema_is_idempotent_and_complete(mongo) -> None:
    """Applying the schema twice works and creates every collection and the unique indexes.

    Args:
        mongo: Connection to a throwaway database with the project schema.
    """
    ensure_schema(mongo.db)
    ensure_schema(mongo.db)
    assert set(C.ALL) <= set(mongo.db.list_collection_names())
    assert "uniq_detail_url" in mongo.db[C.MOVIES].index_information()
    assert "uniq_script_order" in mongo.db[C.DIALOGUES].index_information()


def test_server_side_validator_rejects_malformed_documents(mongo) -> None:
    """MongoDB's JSON-schema validators reject an invalid queue status and a script missing required fields.

    Args:
        mongo: Connection to a throwaway database with the project schema.
    """
    with pytest.raises(WriteError):
        mongo.db[C.CRAWL_QUEUE].insert_one({"_id": "x", "url": "u", "page_type": "home", "status": "bogus",
                                            "next_attempt_at": utcnow()})
    with pytest.raises(WriteError):
        mongo.db[C.SCRIPTS].insert_one({"_id": "s", "movie_id": "m"})  # missing required fields


# ====================================================================== movies
def _listing_entries():
    """Parse the all-scripts fixture page.

    Returns:
        The five movie entries of the fixture listing.
    """
    return parse_listing_page(fixture_text("all_scripts.html"), BASE_URL("/all-scripts.html")).entries


def test_listing_upsert_is_idempotent(mongo) -> None:
    """Re-upserting listing entries inserts nothing new and merges listing sources and genres.

    Args:
        mongo: Connection to a throwaway database with the project schema.
    """
    repos = Repositories(mongo.db)
    entries = _listing_entries()
    first = repos.movies.upsert_from_listing(entries, source_label="all-scripts", listing_url=BASE_URL("/all-scripts.html"))
    second = repos.movies.upsert_from_listing(entries, source_label="all-scripts", listing_url=BASE_URL("/all-scripts.html"))
    third = repos.movies.upsert_from_listing(entries[:2], source_label="genre:Drama", listing_url=BASE_URL("/genre/Drama"))
    assert first["inserted"] == 5 and second["inserted"] == 0 and third["inserted"] == 0
    assert mongo.db[C.MOVIES].count_documents({}) == 5
    lighthouse = repos.movies.get_by_detail_url(entries[1].detail_url)
    assert lighthouse["display_title"] == "The Lighthouse Keeper"
    assert set(lighthouse["listing_sources"]) == {"all-scripts", "genre:Drama"}
    assert lighthouse["listing_genres"] == ["Drama"]
    assert lighthouse["_id"].startswith("mv-the-lighthouse-keeper-")


def test_unique_detail_url_constraint(mongo) -> None:
    """A second movie with the same detail URL violates the unique index.

    Args:
        mongo: Connection to a throwaway database with the project schema.
    """
    repos = Repositories(mongo.db)
    url = BASE_URL("/Movie Scripts/X Script.html")
    repos.movies.ensure_from_detail_url(url, "X")
    with pytest.raises(DuplicateKeyError):
        mongo.db[C.MOVIES].insert_one({"_id": "other", "source": "imsdb", "detail_url": url, "status": "discovered",
                                       "first_discovered_at": utcnow()})


# ====================================================================== scripts + derived records
def _bundle(mongo, html: str, processed_dir, *, movie_id: str = "mv-the-lighthouse-keeper-test"):
    """Parse screenplay HTML into a script bundle for the test movie (the movie document is upserted first).

    Args:
        mongo: Connection used to upsert the movie document.
        html: The screenplay page HTML.
        processed_dir: Where oversized element arrays would be offloaded (unused for this small fixture).
        movie_id: ``_id`` of the movie that owns the script.

    Returns:
        The ``(script, scenes, dialogues, characters)`` tuple from ``build_script_bundle``.
    """
    movie = {"_id": movie_id, "title": "Lighthouse Keeper, The"}
    mongo.db[C.MOVIES].update_one({"_id": movie_id}, {"$setOnInsert": {
        "source": "imsdb", "detail_url": BASE_URL("/Movie Scripts/Lighthouse Keeper, The Script.html"),
        "status": "detail_collected", "first_discovered_at": utcnow(), "title": "Lighthouse Keeper, The"}},
        upsert=True)
    decoded = decode_html_bytes(html.encode("utf-8"), "text/html; charset=UTF-8")
    extracted = extract_screenplay(decoded.text)
    parsed = parse_screenplay(extracted)
    raw = RawRef(artifact_id="raw-test", sha256="0" * 64, path="x/y.html", size=1, fetched_at=utcnow(), status_code=200)
    return build_script_bundle(movie=movie, script_url="https://imsdb.com/scripts/Lighthouse-Keeper,-The.html",
                               raw=raw, decoded=decoded, extracted=extracted, parsed=parsed,
                               rights=make_rights("sample", 1), fetched_at=utcnow(), now=utcnow(),
                               processed_dir=processed_dir)


def _save(repos: Repositories, bundle, parse_id: str):
    """Save a script bundle the way ingestion does: the script first, then its derived records.

    Args:
        repos: Repositories on the test database.
        bundle: ``(script, scenes, dialogues, characters)`` from ``_bundle``.
        parse_id: Id stamped on this parse's derived records.

    Returns:
        The script save outcome (``action`` and ``version``).
    """
    script, scenes, dialogues, characters = bundle
    outcome = repos.scripts.save(script, parse_id=parse_id)
    repos.derived.replace_for_script(script.id, parse_id, scenes=scenes, dialogues=dialogues, characters=characters)
    return outcome


def test_script_save_is_idempotent_and_relationships_hold(mongo, settings) -> None:
    """Saving identical content twice changes nothing, and dialogue/scene/character links and text offsets hold.

    Args:
        mongo: Connection to a throwaway database with the project schema.
        settings: Test settings (provides ``processed_dir``).
    """
    repos = Repositories(mongo.db)
    html = fixture_text("script_lighthouse.html")
    first = _save(repos, _bundle(mongo, html, settings.processed_dir), "p1")
    counts = {c: mongo.db[c].count_documents({}) for c in (C.SCRIPTS, C.SCENES, C.DIALOGUES, C.CHARACTERS)}
    second = _save(repos, _bundle(mongo, html, settings.processed_dir), "p2")
    assert first.action == "created" and second.action == "unchanged" and second.version == 1
    assert {c: mongo.db[c].count_documents({}) for c in counts} == counts  # no duplicates
    assert counts == {C.SCRIPTS: 1, C.SCENES: 5, C.DIALOGUES: 8, C.CHARACTERS: 2}
    dialogue = mongo.db[C.DIALOGUES].find_one({"order": 0})
    assert mongo.db[C.SCENES].find_one({"_id": dialogue["scene_id"]}) is not None
    character = mongo.db[C.CHARACTERS].find_one({"_id": dialogue["character_id"]})
    assert character["name"] == "MARGARET" and dialogue["_id"] in character["dialogue_ids"]
    script = mongo.db[C.SCRIPTS].find_one({})
    assert script["text"][dialogue["char_start"]:dialogue["char_end"]] == dialogue["original_text"]


def test_changed_content_archives_previous_version_and_cleans_stale_records(mongo, settings) -> None:
    """Changed text bumps the version, archives the old document and removes derived records of the old parse.

    Args:
        mongo: Connection to a throwaway database with the project schema.
        settings: Test settings (provides ``processed_dir``).
    """
    repos = Repositories(mongo.db)
    html = fixture_text("script_lighthouse.html")
    _save(repos, _bundle(mongo, html, settings.processed_dir), "p1")
    # Remove the last scene (and its dialogue-free content) from the source -> fewer scenes.
    changed = html.replace("Calm sea. Gulls wheel overhead.", "Calm sea at last.").replace(
        "<b>3    EXT. CLIFF PATH - DAWN                                         3\n</b>", "")
    outcome = _save(repos, _bundle(mongo, changed, settings.processed_dir), "p2")
    assert outcome.action == "updated" and outcome.version == 2
    archived = mongo.db[C.SCRIPT_VERSIONS].find_one({"script_id": outcome_script_id(mongo)})
    assert archived["version"] == 1 and "Gulls wheel overhead" in archived["document"]["text"]
    assert mongo.db[C.SCENES].count_documents({}) == 4  # stale scene 4 removed
    assert mongo.db[C.SCENES].count_documents({"parse_id": {"$ne": "p2"}}) == 0
    assert mongo.db[C.SCRIPTS].find_one({})["previous_versions"] == 1


def outcome_script_id(mongo) -> str:
    """Return the ``_id`` of the (single) script in the test database.

    Args:
        mongo: Connection to the test database.

    Returns:
        The script's ``_id``.
    """
    return mongo.db[C.SCRIPTS].find_one({}, {"_id": 1})["_id"]


# ====================================================================== queue
def test_queue_dedup_claim_complete(mongo) -> None:
    """The queue de-duplicates by canonical URL, leases an item to one worker only, and completion clears the lease.

    Args:
        mongo: Connection to a throwaway database with the project schema.
    """
    repos = Repositories(mongo.db)
    item = make_queue_item(BASE_URL("/genre/Drama"))
    assert repos.queue.enqueue_many([item, item]) == 1
    assert repos.queue.enqueue_many([make_queue_item("https://www.imsdb.com/genre/Drama/")]) == 0  # same canonical URL
    claimed = repos.queue.claim_next(job_id="j", worker_id="w1", page_types=["listing_genre"], lease_seconds=60)
    assert claimed["status"] == "in_progress" and claimed["attempts"] == 1
    assert repos.queue.claim_next(job_id="j", worker_id="w2", page_types=["listing_genre"], lease_seconds=60) is None
    assert repos.queue.complete(claimed["_id"], "w1", status_code=200, content_hash="h", changed=True)
    done = repos.queue.get_by_url(item.url)
    assert done["status"] == "done" and done["content_hash"] == "h" and done["lease_owner"] is None


def test_queue_failure_backoff_and_permanent_failure(mongo) -> None:
    """Retryable failures go back to ``pending`` until ``max_attempts``, then the item is ``failed``.

    Args:
        mongo: Connection to a throwaway database with the project schema.
    """
    repos = Repositories(mongo.db)
    repos.queue.enqueue_many([make_queue_item(BASE_URL("/genre/Drama"))])
    for attempt in range(1, 4):
        item = repos.queue.claim_next(job_id="j", worker_id="w", page_types=["listing_genre"], lease_seconds=60,
                                      now=utcnow() + timedelta(days=attempt))
        status = repos.queue.fail(item, "w", error="HTTP 503", status_code=503, retryable=True, blocked=False,
                                  retry_after=None, max_attempts=3)
        assert status == ("pending" if attempt < 3 else "failed")
    final = repos.queue.get_by_url(BASE_URL("/genre/Drama"))
    assert final["status"] == "failed" and final["last_status_code"] == 503


def test_blocked_items_are_not_retried(mongo) -> None:
    """A failure marked as blocked moves the item straight to ``blocked``.

    Args:
        mongo: Connection to a throwaway database with the project schema.
    """
    repos = Repositories(mongo.db)
    repos.queue.enqueue_many([make_queue_item(BASE_URL("/genre/Drama"))])
    item = repos.queue.claim_next(job_id="j", worker_id="w", page_types=["listing_genre"], lease_seconds=60)
    assert repos.queue.fail(item, "w", error="403", status_code=403, retryable=False, blocked=True, retry_after=None,
                            max_attempts=5) == "blocked"


def test_failure_after_lost_lease_writes_nothing(mongo) -> None:
    """A worker whose lease was taken over gets ``lease_lost`` and cannot overwrite the item.

    Args:
        mongo: Connection to a throwaway database with the project schema.
    """
    repos = Repositories(mongo.db)
    repos.queue.enqueue_many([make_queue_item(BASE_URL("/genre/Drama"))])
    item = repos.queue.claim_next(job_id="j", worker_id="w1", page_types=["listing_genre"], lease_seconds=60)
    repos.queue.col.update_one({"_id": item["_id"]}, {"$set": {"lease_owner": "w2"}})  # recovered by another worker
    assert repos.queue.fail(item, "w1", error="503", status_code=503, retryable=False, blocked=True,
                            retry_after=None, max_attempts=5) == "lease_lost"
    assert repos.queue.get_by_url(BASE_URL("/genre/Drama"))["status"] == "in_progress"


def test_enqueue_counts_inserts_in_a_batch_with_duplicates(mongo, monkeypatch: pytest.MonkeyPatch) -> None:
    """When a batch partly fails with duplicate keys (a concurrent insert), its successful upserts still count.

    Args:
        mongo: Connection to a throwaway database with the project schema.
        monkeypatch: Makes ``bulk_write`` raise a duplicate-key ``BulkWriteError`` after 2 upserts.
    """
    from pymongo.errors import BulkWriteError

    repos = Repositories(mongo.db)

    def racing_bulk_write(ops: list, ordered: bool = True) -> None:
        """Pretend 2 of the upserts succeeded and one lost a race to a concurrent insert.

        Args:
            ops: The write operations.
            ordered: Ignored.

        Raises:
            BulkWriteError: Always, with one duplicate-key error and ``nUpserted`` 2.
        """
        raise BulkWriteError({"writeErrors": [{"index": 2, "code": 11000, "errmsg": "E11000 duplicate key"}],
                              "nUpserted": 2, "writeConcernErrors": []})

    monkeypatch.setattr(repos.queue.col, "bulk_write", racing_bulk_write)
    items = [make_queue_item(BASE_URL(f"/genre/G{i}")) for i in range(3)]
    assert repos.queue.enqueue_many(items) == 2


def test_permission_level_is_validated(mongo) -> None:
    """Unknown permission levels and negative sample limits are refused before anything is written.

    Args:
        mongo: Connection to a throwaway database with the project schema.
    """
    from src.services.permissions import default_imsdb_permission

    repos = Repositories(mongo.db)
    repos.permissions.seed(default_imsdb_permission())
    with pytest.raises(ValueError, match="unknown permission level"):
        repos.permissions.change_level("imsdb", level="everything", sample_limit=None, granted_by="t", evidence="e")
    with pytest.raises(ValueError, match="negative"):
        repos.permissions.change_level("imsdb", level="sample", sample_limit=-1, granted_by="t", evidence="e")
    assert repos.permissions.get("imsdb").version == 1


def test_expired_leases_are_recovered(mongo) -> None:
    """Only leases past their expiry are reset to ``pending``.

    Args:
        mongo: Connection to a throwaway database with the project schema.
    """
    repos = Repositories(mongo.db)
    repos.queue.enqueue_many([make_queue_item(BASE_URL("/genre/Drama"))])
    repos.queue.claim_next(job_id="crashed", worker_id="w", page_types=["listing_genre"], lease_seconds=60)
    assert repos.queue.reset_expired_leases(now=utcnow()) == 0
    assert repos.queue.reset_expired_leases(now=utcnow() + timedelta(minutes=5)) == 1
    assert repos.queue.get_by_url(BASE_URL("/genre/Drama"))["status"] == "pending"


def test_crawl_lock_is_exclusive(mongo) -> None:
    """The crawl lock refuses a second owner, is re-entrant for its owner, and is free after release.

    Args:
        mongo: Connection to a throwaway database with the project schema.
    """
    repos = Repositories(mongo.db)
    repos.jobs.acquire_lock("owner-a")
    with pytest.raises(CrawlerBusyError):
        repos.jobs.acquire_lock("owner-b")
    repos.jobs.acquire_lock("owner-a")  # re-entrant for the same owner
    repos.jobs.release_lock("owner-a")
    repos.jobs.acquire_lock("owner-b")


def test_permission_changes_keep_history(mongo) -> None:
    """Seeding never overwrites, and each level change bumps the version and keeps history and evidence.

    Args:
        mongo: Connection to a throwaway database with the project schema.
    """
    from src.services.permissions import default_imsdb_permission

    repos = Repositories(mongo.db)
    assert repos.permissions.seed(default_imsdb_permission())
    assert not repos.permissions.seed(default_imsdb_permission())  # never overwrites
    record = repos.permissions.change_level("imsdb", level="discovery_only", sample_limit=None, granted_by="tester",
                                            evidence="lowered for test")
    record = repos.permissions.change_level("imsdb", level="bulk", sample_limit=None, granted_by="tester",
                                            evidence="written permission (test)")
    assert record.level == "bulk" and record.version == 3
    assert [h["level"] for h in record.history] == ["sample", "discovery_only"]
    assert len(record.evidence) == 3


def test_lock_held_by_dead_local_process_is_taken_over(mongo) -> None:
    """A lock held by a dead process on this host is taken over; one held by a live process is not.

    Args:
        mongo: Connection to a throwaway database with the project schema.
    """
    import os
    import socket

    from src.database.repositories import LOCK_ID, pid_alive

    assert pid_alive(os.getpid())
    assert not pid_alive(2_000_000_000)  # no such process
    repos = Repositories(mongo.db)
    dead_owner = f"{socket.gethostname()}:2000000000:abc123"
    repos.jobs.acquire_lock(dead_owner)  # simulates a crashed crawler that never released the lock
    repos.jobs.acquire_lock("survivor")
    lock = mongo.db[C.CRAWL_JOBS].find_one({"_id": LOCK_ID})
    assert lock["owner"] == "survivor" and lock["taken_over_from"] == dead_owner
    live_owner = f"{socket.gethostname()}:{os.getpid()}:live"
    repos.jobs.release_lock("survivor")
    repos.jobs.acquire_lock(live_owner)
    with pytest.raises(CrawlerBusyError):
        repos.jobs.acquire_lock("intruder")  # a live holder is never displaced


def test_canonicalization_merges_uncollected_aliases(mongo) -> None:
    """Movies stored under a legacy double-encoded URL merge into the canonical record (idempotently).

    The alias movie is deleted, its listing sources and URL move to the canonical one, and its queue item is skipped.

    Args:
        mongo: Connection to a throwaway database with the project schema.
    """
    from src.crawler.url_manager import url_id
    from src.services.deduplication import canonicalize_movie_urls, duplicate_movie_titles

    db = mongo.db
    canonical = "https://imsdb.com/Movie%20Scripts/Kate%20&%20Leopold%20Script.html"
    legacy = "https://imsdb.com/Movie%20Scripts/Kate%20%2526%20Leopold%20Script.html"  # pre-fix canonical form
    now = utcnow()
    base = {"source": "imsdb", "status": "discovered", "first_discovered_at": now, "title_key": "kate-leopold"}
    db[C.MOVIES].insert_many([
        {"_id": "mv-good", "detail_url": canonical, "listing_sources": ["all-scripts"], **base},
        {"_id": "mv-legacy", "detail_url": legacy, "listing_sources": ["alphabetical:K"],
         "source_hrefs": ["/Movie Scripts/Kate %2526 Leopold Script.html"], **base},
    ])
    for url in (canonical, legacy):
        item = make_queue_item(url, now=now)
        item = item.model_copy(update={"url": url, "id": url_id(url)})  # as stored before the fix
        db[C.CRAWL_QUEUE].insert_one(item.to_mongo())
    assert len(duplicate_movie_titles(db)) == 1
    result = canonicalize_movie_urls(db)
    assert [m["from_id"] for m in result["merged"]] == ["mv-legacy"]
    good = db[C.MOVIES].find_one({"_id": "mv-good"})
    assert set(good["listing_sources"]) == {"all-scripts", "alphabetical:K"} and legacy in good["url_aliases"]
    assert db[C.MOVIES].find_one({"_id": "mv-legacy"}) is None
    assert db[C.CRAWL_QUEUE].find_one({"url": legacy})["status"] == "skipped"
    assert duplicate_movie_titles(db) == []
    assert canonicalize_movie_urls(db)["merged"] == []  # idempotent
