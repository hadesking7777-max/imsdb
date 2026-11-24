"""Duplicate detection.

Hard duplicates are prevented by unique indexes (canonical URLs, deterministic ids).
This module finds *soft* duplicates that indexes cannot catch, e.g. the same movie
listed under two URLs or two script pages with identical text. Duplicates are
reported, never merged or deleted automatically - the one exception is
``canonicalize_movie_urls``, which merges *uncollected* movie records left over from an
older URL normalization into their canonical twin.
"""

from __future__ import annotations

from typing import Any

from pymongo.database import Database

from src.database.models import Collections as C


def duplicate_movie_titles(db: Database, limit: int = 100) -> list[dict[str, Any]]:
    """Movies sharing a normalized title (and release year when known) under different URLs.

    Args:
        db: Project database (reads ``movies``).
        limit: Maximum number of groups to return.

    Returns:
        Groups, largest first: ``_id`` (``title_key`` and ``year``), ``count``, ``ids`` and ``urls``.
    """
    pipeline = [
        {"$group": {"_id": {"title_key": "$title_key", "year": "$release_year"},
                    "count": {"$sum": 1}, "ids": {"$push": "$_id"}, "urls": {"$push": "$detail_url"}}},
        {"$match": {"count": {"$gt": 1}, "_id.title_key": {"$ne": None}}},
        {"$sort": {"count": -1}},
        {"$limit": limit},
    ]
    return list(db[C.MOVIES].aggregate(pipeline, allowDiskUse=True))


def duplicate_script_contents(db: Database, limit: int = 100) -> list[dict[str, Any]]:
    """Different script URLs whose extracted text is byte-identical.

    Args:
        db: Project database (reads ``scripts``; scripts without a ``content_hash`` are ignored).
        limit: Maximum number of groups to return.

    Returns:
        Groups (unordered): ``_id`` (the content hash), ``count``, ``ids`` (script ids) and ``movies``
        (their movie ids).
    """
    pipeline = [
        {"$match": {"content_hash": {"$ne": None}}},
        {"$group": {"_id": "$content_hash", "count": {"$sum": 1}, "ids": {"$push": "$_id"},
                    "movies": {"$push": "$movie_id"}}},
        {"$match": {"count": {"$gt": 1}}},
        {"$limit": limit},
    ]
    return list(db[C.SCRIPTS].aggregate(pipeline, allowDiskUse=True))


def duplicate_key_violations(db: Database) -> dict[str, int]:
    """Defensive check that natural keys are unique (should always be zero).

    Args:
        db: Project database.

    Returns:
        For each checked collection (movies, scripts, scenes, dialogues, characters, crawl_queue), the
        number of natural-key values that occur more than once.
    """
    checks = {
        C.MOVIES: ["detail_url"],
        C.SCRIPTS: ["script_url"],
        C.SCENES: ["script_id", "scene_number"],
        C.DIALOGUES: ["script_id", "order"],
        C.CHARACTERS: ["script_id", "name_key"],
        C.CRAWL_QUEUE: ["url"],
    }
    out: dict[str, int] = {}
    for collection, fields in checks.items():
        group_id = {f: f"${f}" for f in fields}
        rows = list(db[collection].aggregate([
            {"$group": {"_id": group_id, "n": {"$sum": 1}}},
            {"$match": {"n": {"$gt": 1}}},
            {"$count": "groups"},
        ], allowDiskUse=True))
        out[collection] = rows[0]["groups"] if rows else 0
    return out


def canonicalize_movie_urls(db: Database) -> dict[str, Any]:
    """Merge *uncollected* movie records whose ``detail_url`` is not canonical under the current
    normalization rules into their canonical twin (provenance kept in ``url_aliases``).

    Collected records are never merged automatically - they are flagged for review instead.
    Superseded queue items are marked ``skipped`` with the reason; nothing is deleted silently:
    every merge is returned (and logged by the caller).

    For each non-canonical uncollected movie: if no canonical twin exists its ``detail_url`` is rewritten;
    otherwise its listing sources, genres and hrefs are added to the twin and it is deleted. The canonical
    URL is enqueued if missing, and the old URL's pending queue item is skipped. Flagged collected records
    get a ``needs_review`` message.

    Args:
        db: Project database (updates ``movies`` and ``crawl_queue``).

    Returns:
        ``merged``: list of ``{from_id, from_url, into_id, canonical_url}``; ``flagged_for_review``: ids of
        collected movies with a non-canonical URL.
    """
    from src.crawler.discovery import make_queue_item
    from src.crawler.url_manager import normalize_url
    from src.utils.clock import utcnow

    merged: list[dict[str, str]] = []
    flagged: list[str] = []
    now = utcnow()
    for movie in db[C.MOVIES].find({}, {"detail_url": 1, "listing_sources": 1, "listing_genres": 1,
                                         "source_hrefs": 1, "detail_fetched_at": 1}):
        old = movie["detail_url"]
        canonical = normalize_url(old)
        if not canonical or canonical == old:
            continue
        if "detail_fetched_at" in movie:
            db[C.MOVIES].update_one({"_id": movie["_id"]}, {"$set": {"needs_review": f"non-canonical URL; canonical form "
                                                                                      f"is {canonical}"}})
            flagged.append(movie["_id"])
            continue
        twin = db[C.MOVIES].find_one({"detail_url": canonical}, {"_id": 1})
        aliases = {"url_aliases": old}
        if twin is None:
            db[C.MOVIES].update_one({"_id": movie["_id"]}, {"$set": {"detail_url": canonical, "updated_at": now},
                                                             "$addToSet": aliases})
            target_id = movie["_id"]
        else:
            db[C.MOVIES].update_one({"_id": twin["_id"]}, {
                "$addToSet": {"listing_sources": {"$each": movie.get("listing_sources", [])},
                              "listing_genres": {"$each": movie.get("listing_genres", [])},
                              "source_hrefs": {"$each": movie.get("source_hrefs", [])}, **aliases},
                "$set": {"updated_at": now}})
            db[C.MOVIES].delete_one({"_id": movie["_id"], "detail_fetched_at": {"$exists": False}})
            target_id = twin["_id"]
        if db[C.CRAWL_QUEUE].count_documents({"url": canonical}) == 0:
            item = make_queue_item(canonical, discovered_from=old, movie_id=target_id, now=now)
            if item is not None:
                db[C.CRAWL_QUEUE].update_one({"url": canonical}, {"$setOnInsert": item.to_mongo()}, upsert=True)
        db[C.CRAWL_QUEUE].update_one({"url": old, "status": "pending"},
                                     {"$set": {"status": "skipped", "completed_at": now,
                                               "last_error": f"superseded by canonical URL {canonical}"}})
        merged.append({"from_id": movie["_id"], "from_url": old, "into_id": target_id, "canonical_url": canonical})
    return {"merged": merged, "flagged_for_review": flagged}


def duplicate_summary(db: Database) -> dict[str, Any]:
    """Summarize soft duplicates and natural-key violations (read-only).

    Args:
        db: Project database.

    Returns:
        ``movie_title_groups``/``movie_records_in_groups`` and ``script_content_groups``/
        ``script_records_in_groups`` (from the first 100 groups of each), ``key_violations`` (see
        ``duplicate_key_violations``), and up to 10 example groups each as ``movie_examples`` and
        ``script_examples``.
    """
    movies = duplicate_movie_titles(db)
    scripts = duplicate_script_contents(db)
    return {
        "movie_title_groups": len(movies),
        "movie_records_in_groups": sum(g["count"] for g in movies),
        "script_content_groups": len(scripts),
        "script_records_in_groups": sum(g["count"] for g in scripts),
        "key_violations": duplicate_key_violations(db),
        "movie_examples": movies[:10],
        "script_examples": scripts[:10],
    }
